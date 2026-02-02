# modules/analysis.py

import geopandas as gpd
import numpy as np
import pandas as pd
import pymannkendall as mk
import rasterio
import requests
import streamlit as st
from rasterio.mask import mask
from rasterio.warp import Resampling, reproject
from scipy import stats
from scipy.stats import loglaplace, norm
import warnings

from modules.config import Config

# Intentamos importar scipy con manejo de error
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# --- FUNCIÓN AUXILIAR DE DETECCIÓN (EL CEREBRO DE COMPATIBILIDAD) ---
def get_safe_cols(df):
    """
    Detecta automáticamente las columnas de fecha y valor de precipitación,
    sin importar si vienen de la BD nueva ('valor') o vieja ('precipitation').
    """
    if df is None or df.empty: return None, None
    
    # 1. Candidatos para fecha
    c_date = next((c for c in ['fecha', Config.DATE_COL, 'ds', 'fecha_mes_año', 'Date', 'time'] if c in df.columns), None)
    
    # 2. Candidatos para valor (Lluvia)
    c_val = next((c for c in ['valor', Config.PRECIPITATION_COL, 'precipitation', 'Pptn', 'lluvia', 'rain'] if c in df.columns), None)
    
    return c_date, c_val

@st.cache_data
def calculate_spi(df, col=None, window=12):
    """
    Calcula el Índice Estandarizado de Precipitación (SPI).
    CORREGIDO: Detecta columna automáticamente si 'col' no se especifica.
    """
    if df is None or df.empty:
        return df

    df_spi = df.copy()
    
    # Detección automática de columnas
    c_date, c_val = get_safe_cols(df_spi)
    
    # Si nos pasan una columna específica, la usamos, si no, usamos la detectada
    target_col = col if col and col in df_spi.columns else c_val
    
    if not target_col:
        return pd.Series(dtype=float) # No se encontró columna de datos

    # Asegurar índice de fecha
    if c_date and not isinstance(df_spi.index, pd.DatetimeIndex):
        try:
            df_spi[c_date] = pd.to_datetime(df_spi[c_date])
            df_spi = df_spi.set_index(c_date).sort_index()
        except: pass # Si falla, asumimos que el índice ya es correcto o fallará abajo

    # 1. Calcular acumulado móvil (Rolling Sum)
    df_spi["rolling_precip"] = (
        df_spi[target_col].rolling(window=window, min_periods=window).sum()
    )

    # 2. Calcular SPI (Aproximación Log-Normal / Z-Score)
    try:
        # Filtrar ceros y nulos para el logaritmo
        valid_mask = (df_spi["rolling_precip"] > 0) & (df_spi["rolling_precip"].notna())

        # Transformación Logarítmica
        log_precip = np.log(df_spi.loc[valid_mask, "rolling_precip"])

        mean = log_precip.mean()
        std = log_precip.std()

        df_spi["spi"] = np.nan 

        if std == 0:
            df_spi["spi"] = 0
        else:
            # Calcular Z-score
            df_spi.loc[valid_mask, "spi"] = (log_precip - mean) / std

            # Manejo de Sequía Extrema (Valores 0)
            min_spi = df_spi["spi"].min()
            if pd.isna(min_spi):
                min_spi = -3.0
            
            # Asignar mínimo a los valores que eran 0
            df_spi.loc[
                ~valid_mask
                & df_spi["rolling_precip"].notna()
                & (df_spi["rolling_precip"] == 0),
                "spi",
            ] = min_spi

    except Exception:
        df_spi["spi"] = np.nan

    return df_spi["spi"]


@st.cache_data
def calculate_spei(precip_series, et_series, window=12):
    """
    Calcula el SPEI usando la distribución Log-Laplace.
    """
    scale = int(window) 

    # Validación básica
    if (precip_series is None or et_series is None or precip_series.empty or et_series.empty):
        return pd.Series(dtype=float)

    # Alineación por índice
    df = pd.DataFrame({"precip": precip_series, "et": et_series}).sort_index()
    # Asumimos frecuencia mensual para llenar huecos correctamente
    if not df.index.freq:
        try: df = df.asfreq("MS")
        except: pass

    # Limpieza
    df.dropna(subset=["precip"], inplace=True)
    df["et"] = df["et"].fillna(method="ffill").fillna(method="bfill")
    df.dropna(subset=["et"], inplace=True)

    # --- AJUSTE INTELIGENTE DE UNIDADES ---
    # Si 'et' parece ser Temperatura (< 40 promedio), lo convertimos a ET aprox.
    if df["et"].mean() < 40:
        df["et"] = df["et"] * 4.5 

    if len(df) < scale * 2:
        return pd.Series(dtype=float)

    # Balance Hídrico (D)
    water_balance = df["precip"] - df["et"]

    # Acumulación
    rolling_balance = water_balance.rolling(window=scale, min_periods=scale).sum()

    # Ajuste Log-Laplace
    data_for_fit = rolling_balance.dropna()
    data_for_fit = data_for_fit[np.isfinite(data_for_fit)]

    spei = pd.Series(np.nan, index=rolling_balance.index)

    if not data_for_fit.empty and len(data_for_fit.unique()) > 1:
        try:
            # Ajuste de parámetros
            params = loglaplace.fit(
                data_for_fit,
                floc=data_for_fit.min() - 1e-5 if data_for_fit.min() <= 0 else 0,
            )
            # CDF
            cdf = loglaplace.cdf(rolling_balance.dropna(), *params)
            cdf_series = pd.Series(cdf, index=rolling_balance.dropna().index)
            # Clipping y Z-Score
            cdf_clipped = np.clip(cdf_series.values, 1e-7, 1 - 1e-7)
            spei.loc[cdf_series.index] = norm.ppf(cdf_clipped)
        except Exception:
            pass 

    spei.replace([np.inf, -np.inf], np.nan, inplace=True)
    return spei


def calculate_monthly_anomalies(df_monthly, df_long_full):
    """
    Calcula anomalías mensuales.
    CORREGIDO: Usa nombres consistentes para evitar KeyError.
    """
    if df_monthly.empty or df_long_full.empty:
        return pd.DataFrame()
    
    # Detectar columnas en ambos dataframes
    _, c_val_m = get_safe_cols(df_monthly)
    _, c_val_full = get_safe_cols(df_long_full)
    
    if not c_val_m or not c_val_full: return pd.DataFrame()

    # Calcular Climatología (Promedio por mes)
    # Usamos Config.MONTH_COL que debe ser 'mes'
    climatology = (
        df_long_full.groupby(Config.MONTH_COL)[c_val_full]
        .mean()
        .reset_index()
    )
    
    # Renombramos para el merge
    col_clima_name = "climatology_ppt"
    climatology = climatology.rename(columns={c_val_full: col_clima_name})
    
    # Merge
    merged = pd.merge(df_monthly, climatology, on=Config.MONTH_COL, how="left")
    
    # Cálculo: Valor Actual - Promedio Histórico
    merged["anomalia"] = merged[c_val_m] - merged[col_clima_name]
    
    return merged


def estimate_temperature(altitude_m):
    """Estimación de temperatura basada en gradiente térmico vertical (Andes)."""
    if pd.isna(altitude_m):
        return 25.0
    # Gradiente típico: 28°C a nivel del mar, disminuye 0.6°C por cada 100m
    temp = 28.0 - (0.006 * float(altitude_m))
    return max(temp, 1.0) 


def calculate_water_balance_turc(precip_mm, temp_c):
    """
    Estima Evapotranspiración Real (ETR) y Escorrentía (Q) usando la fórmula de Turc.
    """
    if pd.isna(precip_mm) or pd.isna(temp_c):
        return 0, 0

    # Fórmula de Turc para ETR
    L = 300 + 25 * temp_c + 0.05 * (temp_c**3)
    if L == 0: L = 0.001
    
    etr = precip_mm / np.sqrt(0.9 + (precip_mm / L) ** 2)

    # Limitar ETR a la precipitación
    etr = min(etr, precip_mm)

    # Escorrentía = Precipitación - ETR
    q = precip_mm - etr
    return etr, q


def classify_holdridge_point(precip_mm, altitude_m):
    """Clasificación simple de Holdridge."""
    if pd.isna(precip_mm) or pd.isna(altitude_m):
        return "N/A"
    
    alt = float(altitude_m)
    ppt = float(precip_mm)
    
    if alt < 1000:
        piso = "Tropical"
    elif alt < 2000:
        piso = "Premontano"
    elif alt < 3000:
        piso = "Montano Bajo"
    elif alt < 3500:
        piso = "Montano-Paramo"
    else:
        piso = "Montano"

    if ppt < 1000:
        prov = "Bosque Seco"
    elif ppt < 2000:
        prov = "Bosque Húmedo"
    elif ppt < 4000:
        prov = "Bosque Muy Húmedo"
    else:
        prov = "Bosque Pluvial"

    return f"{prov} {piso}"


# ==============================================================================
# PARTE 2: ANÁLISIS CLIMATOLÓGICO Y BALANCE HÍDRICO (Blindado)
# ==============================================================================

def calculate_morphometry(gdf_basin):
    """
    Calcula morfometría básica de una cuenca.
    (Función auxiliar necesaria para el balance).
    """
    res = {"area_km2": 0, "perimetro_km": 0}
    if gdf_basin is None or gdf_basin.empty:
        return res
    
    try:
        # Proyectar temporalmente a Magna Sirgas (metros) para cálculo
        if gdf_basin.crs and gdf_basin.crs.to_string() != "EPSG:3116":
            gdf_temp = gdf_basin.to_crs("EPSG:3116")
        else:
            gdf_temp = gdf_basin.copy()
            
        area_m2 = gdf_temp.geometry.area.sum()
        perim_m = gdf_temp.geometry.length.sum()
        
        res["area_km2"] = area_m2 / 1e6
        res["perimetro_km"] = perim_m / 1000.0
    except:
        pass
    return res

def calculate_percentiles_and_extremes(df_long, station_name, p_lower=10, p_upper=90):
    """
    Calcula umbrales de percentiles y clasifica eventos extremos.
    """
    if df_long.empty: return pd.DataFrame(), pd.DataFrame()
    
    # Detección de columnas
    c_date, c_val = get_safe_cols(df_long)
    
    # Detectar columna de nombre
    c_name = next((c for c in ['nombre', Config.STATION_NAME_COL, 'nom_est', 'station'] if c in df_long.columns), None)
    
    # Detectar columna de mes
    c_mes = next((c for c in ['mes', Config.MONTH_COL, 'month'] if c in df_long.columns), None)
    
    if not c_val or not c_name: return pd.DataFrame(), pd.DataFrame()

    # Si no existe columna mes, la creamos
    df_work = df_long.copy()
    if not c_mes:
        if c_date:
            df_work['mes'] = pd.to_datetime(df_work[c_date]).dt.month
            c_mes = 'mes'
        else:
            return pd.DataFrame(), pd.DataFrame()

    # Filtrar estación
    df_station_full = df_work[df_work[c_name] == station_name].copy()
    
    if df_station_full.empty: return pd.DataFrame(), pd.DataFrame()

    # Calcular umbrales
    df_thresholds = (
        df_station_full.groupby(c_mes)[c_val]
        .agg(
            p_lower=lambda x: np.nanpercentile(x.dropna(), p_lower),
            p_upper=lambda x: np.nanpercentile(x.dropna(), p_upper),
            mean_monthly="mean",
        )
        .reset_index()
    )
    
    # Merge y Clasificación
    df_station_extremes = pd.merge(df_station_full, df_thresholds, on=c_mes, how="left")
    
    df_station_extremes["event_type"] = "Normal"
    
    is_dry = df_station_extremes[c_val] < df_station_extremes["p_lower"]
    df_station_extremes.loc[is_dry, "event_type"] = f"Sequía Extrema (< P{p_lower}%)"
    
    is_wet = df_station_extremes[c_val] > df_station_extremes["p_upper"]
    df_station_extremes.loc[is_wet, "event_type"] = f"Húmedo Extremo (> P{p_upper}%)"
    
    return df_station_extremes.dropna(subset=[c_val]), df_thresholds


@st.cache_data
def calculate_climatological_anomalies(_df_monthly_filtered, _df_long, baseline_start, baseline_end):
    """
    Calcula anomalías con respecto a un período base.
    """
    df_monthly_filtered = _df_monthly_filtered.copy()
    df_long = _df_long.copy()

    # Detección de columnas
    _, c_val = get_safe_cols(df_long)
    c_name = next((c for c in ['nombre', Config.STATION_NAME_COL, 'nom_est'] if c in df_long.columns), None)
    
    # Columnas de tiempo
    c_year = next((c for c in ['año', Config.YEAR_COL, 'year'] if c in df_long.columns), None)
    c_month = next((c for c in ['mes', Config.MONTH_COL, 'month'] if c in df_long.columns), None)
    
    if not c_val or not c_name or not c_year or not c_month:
        return df_monthly_filtered

    # Filtrar periodo base
    baseline_df = df_long[
        (df_long[c_year] >= baseline_start) & (df_long[c_year] <= baseline_end)
    ]

    # Calcular Climatología
    df_climatology = (
        baseline_df.groupby([c_name, c_month])[c_val]
        .mean()
        .reset_index()
        .rename(columns={c_val: "precip_promedio_climatologico"})
    )

    # Merge
    df_anomalias = pd.merge(
        df_monthly_filtered,
        df_climatology,
        on=[c_name, c_month],
        how="left",
    )

    df_anomalias["anomalia"] = df_anomalias[c_val] - df_anomalias["precip_promedio_climatologico"]
    return df_anomalias


@st.cache_data
def analyze_events(index_series, threshold, event_type="drought"):
    """
    Identifica eventos de sequía/humedad (Rachas).
    """
    if index_series.empty: return pd.DataFrame()

    if event_type == "drought":
        is_event = index_series < threshold
    else:  # 'wet'
        is_event = index_series > threshold

    event_blocks = (is_event.diff() != 0).cumsum()
    active_events = is_event[is_event]
    
    if active_events.empty: return pd.DataFrame()

    events = []
    for event_id, group in active_events.groupby(event_blocks):
        start_date = group.index.min()
        end_date = group.index.max()
        duration = len(group)
        event_values = index_series.loc[start_date:end_date]

        events.append({
            "Fecha Inicio": start_date,
            "Fecha Fin": end_date,
            "Duración (meses)": duration,
            "Magnitud": event_values.sum(),
            "Intensidad": event_values.mean(),
            "Pico": event_values.min() if event_type == "drought" else event_values.max(),
        })

    if not events: return pd.DataFrame()
    return pd.DataFrame(events).sort_values(by="Fecha Inicio").reset_index(drop=True)


@st.cache_data
def calculate_basin_stats(_gdf_stations, _gdf_basins, _df_monthly, basin_name, basin_col_name):
    """
    Calcula estadísticas para estaciones dentro de una cuenca.
    """
    if _gdf_basins is None or basin_col_name not in _gdf_basins.columns:
        return pd.DataFrame(), [], "GeoDataFrame de cuencas inválido."

    target_basin = _gdf_basins[_gdf_basins[basin_col_name] == basin_name]
    if target_basin.empty:
        return pd.DataFrame(), [], f"No se encontró la cuenca '{basin_name}'."

    # Spatial Join
    # Aseguramos proyecciones iguales
    if _gdf_stations.crs != target_basin.crs:
        target_basin = target_basin.to_crs(_gdf_stations.crs)
        
    stations_in_basin = gpd.sjoin(_gdf_stations, target_basin, how="inner", predicate="within")
    
    # Detectar columna de nombre en estaciones
    c_name_est = next((c for c in ['nombre', Config.STATION_NAME_COL, 'nom_est'] if c in stations_in_basin.columns), None)
    
    if not c_name_est: return pd.DataFrame(), [], "Error: Columna nombre no encontrada."
    
    station_names = stations_in_basin[c_name_est].unique().tolist()

    if not station_names:
        return pd.DataFrame(), [], None

    # Filtrar datos de lluvia
    # Detectar columna nombre en datos mensuales
    c_name_data = next((c for c in ['nombre', Config.STATION_NAME_COL, 'nom_est'] if c in _df_monthly.columns), None)
    _, c_val = get_safe_cols(_df_monthly)
    
    if not c_name_data or not c_val: return pd.DataFrame(), [], "Error estructura datos."

    df_basin_precip = _df_monthly[_df_monthly[c_name_data].isin(station_names)]
    
    if df_basin_precip.empty:
        return pd.DataFrame(), station_names, "No hay datos de precipitación en esta zona."

    stats = df_basin_precip[c_val].describe().reset_index()
    stats.columns = ["Métrica", "Valor"]
    stats["Valor"] = stats["Valor"].round(2)

    return stats, station_names, None


@st.cache_data
def get_mean_altitude_for_basin(_basin_geometry):
    """
    Consulta API Open-Elevation.
    """
    try:
        simplified_geom = _basin_geometry.simplify(tolerance=0.01)
        exterior_coords = list(simplified_geom.exterior.coords)
        
        # Limitar puntos para no saturar la API
        if len(exterior_coords) > 50:
            exterior_coords = exterior_coords[::int(len(exterior_coords)/50)]

        locations = [{"latitude": lat, "longitude": lon} for lon, lat in exterior_coords]

        response = requests.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json={"locations": locations},
            timeout=5 # Timeout rápido para no bloquear la app
        )
        if response.status_code != 200: return None, "API Error"

        results = response.json().get("results", [])
        if not results: return None, "No data"
        
        elevations = [res["elevation"] for res in results]
        return np.mean(elevations), None
    except Exception as e:
        return None, str(e)


def calculate_hydrological_balance(precip_mm, alt_m, gdf_basin, delta_temp_c=0):
    """
    Balance Hídrico (Turc) con cálculo de volumen.
    """
    # Valores por defecto seguros
    res_vacio = {k:0 for k in ["P","ET","Q","Q_mm","Vol","Q_m3_año","Alt","Area"]}
    
    if precip_mm is None or pd.isna(precip_mm) or precip_mm <= 0:
        return res_vacio

    # Temperatura estimada
    t = estimate_temperature(alt_m) + delta_temp_c

    # Turc
    L = 300 + 25 * t + 0.05 * t**3
    if L == 0: L = 0.001
    denom = np.sqrt(0.9 + (precip_mm / L) ** 2)
    etr = precip_mm / denom if denom != 0 else precip_mm
    etr = min(etr, precip_mm)

    q = precip_mm - etr # Escorrentía mm

    # Morfometría (Área)
    morph = calculate_morphometry(gdf_basin)
    area = morph.get("area_km2", 0)

    # Volumen (mm * km2 * 1000 = m3) -> NO, mm * km2 / 1000 = Mm3?
    # 1 mm = 1 L/m2
    # 1 km2 = 1,000,000 m2
    # 1 mm en 1 km2 = 1,000,000 Litros = 1,000 m3
    # Vol (Mm3) = Q(mm) * Area(km2) / 1000
    
    vol_mm3 = (q * area) / 1000.0 

    return {
        "P": precip_mm,
        "P_media_anual_mm": precip_mm,
        "ET": etr,
        "ET_media_anual_mm": etr,
        "Q": q,
        "Q_mm": q,
        "Vol": vol_mm3,
        "Q_m3_año": vol_mm3 * 1e6, # m3 totales
        "Alt": alt_m,
        "Area": area,
    }

# ==============================================================================
# PARTE 3: MORFOMETRÍA, CURVA HIPSOMÉTRICA Y TENDENCIAS (Blindado)
# ==============================================================================

def calculate_morphometry(gdf_basin, dem_path=None):
    """
    Calcula métricas morfométricas básicas y avanzadas (si hay DEM).
    """
    if gdf_basin is None or gdf_basin.empty:
        return {"area_km2":0, "perimetro_km":0, "alt_max_m":0, "alt_min_m":0, "alt_prom_m":0, "pendiente_prom":0}

    # Proyección para área (EPSG:3116 Colombia Magna Sirgas u otro métrico)
    try: 
        gdf_proj = gdf_basin.to_crs(epsg=3116)
    except: 
        gdf_proj = gdf_basin # Si falla, usamos el que traiga (esperando que sea métrico)

    area_km2 = gdf_proj.area.sum() / 1e6
    perimetro_km = gdf_proj.length.sum() / 1000
    indice_forma = (0.28 * perimetro_km) / np.sqrt(area_km2) if area_km2 > 0 else 0

    # --- CÁLCULO REAL DE ALTITUDES CON DEM ---
    alt_max, alt_min, alt_med, pendiente_prom = 0, 0, 0, 0
    
    if dem_path:
        try:
            import rasterio
            from rasterio.mask import mask # Importación local segura
            
            with rasterio.open(dem_path) as src:
                # Asegurar CRS
                if gdf_basin.crs != src.crs:
                    geom_for_mask = gdf_basin.to_crs(src.crs).geometry
                else:
                    geom_for_mask = gdf_basin.geometry

                # Máscara
                out_image, _ = mask(src, geom_for_mask, crop=True, nodata=src.nodata)
                data = out_image[0]
                validos = data[data != src.nodata]
                validos = validos[validos > -100] # Filtro ruido

                if validos.size > 0:
                    alt_max = float(np.max(validos))
                    alt_min = float(np.min(validos))
                    alt_med = float(np.mean(validos))
                    
                    # Pendiente aproximada (Relief Ratio)
                    longitud_aprox = np.sqrt(area_km2 * 1e6) 
                    if longitud_aprox > 0:
                        pendiente_prom = ((alt_max - alt_min) / longitud_aprox) * 100
        except Exception as e:
            # print(f"Error procesando DEM en morphometry: {e}")
            # Fallback a simulación
            alt_max, alt_min, alt_med = 2800, 800, 1800
    else:
        # Fallback si no hay ruta DEM
        alt_max, alt_min, alt_med = 2800, 800, 1800

    return {
        "area_km2": area_km2,
        "perimetro_km": perimetro_km,
        "indice_forma": indice_forma,
        "alt_max_m": alt_max,
        "alt_min_m": alt_min,
        "alt_prom_m": alt_med,
        "pendiente_prom": pendiente_prom,
    }


def calculate_hypsometric_curve(gdf_basin, dem_path=None):
    """
    Genera la curva hipsométrica REAL leyendo el DEM y recortando por la cuenca.
    """
    if gdf_basin is None or gdf_basin.empty:
        return None

    elevations = []
    
    # 1. INTENTAR LEER DEL DEM SI EXISTE
    if dem_path:
        try:
            import rasterio
            from rasterio.mask import mask
            
            with rasterio.open(dem_path) as src:
                if gdf_basin.crs != src.crs:
                    gdf_basin_proj = gdf_basin.to_crs(src.crs)
                else:
                    gdf_basin_proj = gdf_basin

                out_image, _ = mask(
                    src, 
                    gdf_basin_proj.geometry, 
                    crop=True, 
                    nodata=src.nodata
                )
                
                data = out_image[0]
                elevations = data[data != src.nodata]
                # Filtro valores absurdos
                elevations = elevations[(elevations > -100) & (elevations < 6000)]

        except Exception as e:
            # print(f"Error leyendo DEM: {e}")
            elevations = []

    # 2. SI FALLA EL DEM, USAR FALLBACK SINTÉTICO
    if len(elevations) == 0:
        morph = calculate_morphometry(gdf_basin)
        min_z, max_z = morph["alt_min_m"], morph["alt_max_m"]
        n_points = 1000
        elevations = np.linspace(min_z, max_z, n_points)

    # 3. CALCULAR CURVA
    elevations_sorted = np.sort(elevations)[::-1] 
    n_pixels = len(elevations_sorted)
    
    area_percent = np.arange(1, n_pixels + 1) / n_pixels * 100
    
    # Reducir resolución para graficar (optimización web)
    if n_pixels > 200:
        indices = np.linspace(0, n_pixels - 1, 200, dtype=int)
        elevations_plot = elevations_sorted[indices]
        area_plot = area_percent[indices]
    else:
        elevations_plot = elevations_sorted
        area_plot = area_percent

    # 4. Ajuste de Ecuación
    try:
        coeffs = np.polyfit(area_plot, elevations_plot, 3)
        poly_model = np.poly1d(coeffs)
        
        eq_str = (
            f"H = {coeffs[0]:.2e}A³ "
            f"{'+' if coeffs[1]>=0 else '-'} {abs(coeffs[1]):.2e}A² "
            f"{'+' if coeffs[2]>=0 else '-'} {abs(coeffs[2]):.2e}A "
            f"{'+' if coeffs[3]>=0 else '-'} {abs(coeffs[3]):.2f}"
        )
    except:
        eq_str = "N/A"
        poly_model = lambda x: x

    return {
        "elevations": elevations_plot,  # Eje Y
        "area_percent": area_plot,      # Eje X
        "equation": eq_str,
        "poly_model": poly_model,
        "source": "DEM Real" if dem_path and len(elevations) > 0 else "Simulado"
    }

def calculate_all_station_trends(df_anual, gdf_stations):
    """
    Calcula la tendencia de Mann-Kendall para todas las estaciones.
    BLINDADO: Detecta nombres de columnas automáticamente.
    """
    if df_anual.empty: return gpd.GeoDataFrame()

    # --- DETECCIÓN DE COLUMNAS (La clave de la compatibilidad) ---
    _, c_val = get_safe_cols(df_anual)
    
    # Buscar columna de nombre en datos anuales
    c_name_data = next((c for c in ['nombre', Config.STATION_NAME_COL, 'nom_est', 'station'] if c in df_anual.columns), None)
    
    if not c_val or not c_name_data: return gpd.GeoDataFrame()

    trend_results = []
    stations_with_data = df_anual[c_name_data].unique()

    for station in stations_with_data:
        station_data = df_anual[df_anual[c_name_data] == station].dropna(subset=[c_val])
        
        # Mínimo 10 años para tendencia
        if len(station_data) >= 10:
            try:
                mk_result = mk.original_test(station_data[c_val])
                trend_results.append({
                    c_name_data: station, # Usamos el nombre de col que encontramos
                    "slope_sen": mk_result.slope,
                    "p_value": mk_result.p,
                })
            except Exception:
                continue

    if not trend_results:
        return gpd.GeoDataFrame()

    df_trends = pd.DataFrame(trend_results)

    # --- UNIÓN CON GEOMETRÍA ---
    # Buscar columna de nombre en geo
    c_name_geo = next((c for c in ['nombre', Config.STATION_NAME_COL, 'nom_est'] if c in gdf_stations.columns), None)
    
    if not c_name_geo: return gpd.GeoDataFrame()

    # Merge flexible
    gdf_trends = pd.merge(
        gdf_stations, 
        df_trends, 
        left_on=c_name_geo, 
        right_on=c_name_data
    )

    return gpd.GeoDataFrame(gdf_trends)

# ==============================================================================
# PARTE 4: RASTERIZACIÓN Y ZONIFICACIÓN CLIMÁTICA (FINAL)
# ==============================================================================

def generate_life_zone_raster(dem_path, ppt_path, mask_geom=None, downscale_factor=1):
    """
    Genera una matriz de Zonas de Vida cruzando Raster de Altura (DEM) y Precipitación.
    Optimizado para memoria y manejo de CRS.
    """
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.io import MemoryFile
        from rasterio.mask import mask

        # 1. Abrir DEM (Maestro)
        with rasterio.open(dem_path) as src_dem:
            # Calcular nueva resolución (Downscale para rendimiento en web)
            dst_height = src_dem.height // downscale_factor
            dst_width = src_dem.width // downscale_factor
            
            # Ajustar transformación
            dst_transform = src_dem.transform * src_dem.transform.scale(
                (src_dem.width / dst_width), (src_dem.height / dst_height)
            )

            # Leer y redimensionar DEM
            dem_arr = np.empty((dst_height, dst_width), dtype=np.float32)
            reproject(
                source=rasterio.band(src_dem, 1),
                destination=dem_arr,
                src_transform=src_dem.transform,
                src_crs=src_dem.crs,
                dst_transform=dst_transform,
                dst_crs=src_dem.crs,
                resampling=Resampling.bilinear,
            )

            # Perfil para el output
            profile = src_dem.profile.copy()
            profile.update({
                "height": dst_height,
                "width": dst_width,
                "transform": dst_transform,
                "dtype": "int16",
                "nodata": 0,
                "count": 1
            })

        # 2. Abrir PPT y alinearlo al DEM
        with rasterio.open(ppt_path) as src_ppt:
            ppt_arr = np.empty((dst_height, dst_width), dtype=np.float32)
            reproject(
                source=rasterio.band(src_ppt, 1),
                destination=ppt_arr,
                src_transform=src_ppt.transform,
                src_crs=src_ppt.crs,
                dst_transform=dst_transform,  # Usamos la rejilla del DEM
                dst_crs=src_dem.crs,          # Usamos el CRS del DEM
                resampling=Resampling.bilinear,
            )

        # 3. Clasificación Vectorizada (Holdridge Simplificado)
        lz_arr = np.zeros_like(dem_arr, dtype=np.int16)

        # Máscara de datos válidos (evitar nodata del DEM y PPT)
        valid = (dem_arr > -500) & (ppt_arr >= 0) & np.isfinite(dem_arr) & np.isfinite(ppt_arr)

        if np.any(valid):
            alt = dem_arr[valid]
            ppt = ppt_arr[valid]
            zones = np.zeros_like(alt, dtype=np.int16)

            # --- Lógica Holdridge Vectorizada ---
            # 1: Bs-T, 2: Bh-T, ... (Códigos internos para visualización)
            
            # Tropical (<1000m)
            mask_T = alt < 1000
            zones[mask_T & (ppt < 1000)] = 1  # b-s-T
            zones[mask_T & (ppt >= 1000) & (ppt < 2000)] = 2  # b-h-T
            zones[mask_T & (ppt >= 2000) & (ppt < 4000)] = 3  # b-mh-T
            zones[mask_T & (ppt >= 4000)] = 4  # b-pl-T

            # Premontano (1000-2000m)
            mask_P = (alt >= 1000) & (alt < 2000)
            zones[mask_P & (ppt < 1000)] = 5
            zones[mask_P & (ppt >= 1000) & (ppt < 2000)] = 6
            zones[mask_P & (ppt >= 2000) & (ppt < 4000)] = 7
            zones[mask_P & (ppt >= 4000)] = 8

            # Montano Bajo (2000-3000m)
            mask_MB = (alt >= 2000) & (alt < 3000)
            zones[mask_MB & (ppt < 1000)] = 9
            zones[mask_MB & (ppt >= 1000) & (ppt < 2000)] = 10
            zones[mask_MB & (ppt >= 2000) & (ppt < 4000)] = 11
            zones[mask_MB & (ppt >= 4000)] = 12

            # Montano (>3000m)
            mask_M = alt >= 3000
            zones[mask_M & (ppt < 1000)] = 13
            zones[mask_M & (ppt >= 1000)] = 14

            lz_arr[valid] = zones

        # 4. Aplicar Máscara de Cuenca (Si existe)
        if mask_geom is not None:
            with MemoryFile() as memfile:
                with memfile.open(**profile) as dataset:
                    dataset.write(lz_arr, 1)

                    # Reproyectar geometría máscara si es necesario
                    if hasattr(mask_geom, "crs") and mask_geom.crs != src_dem.crs:
                        mask_geom_proj = mask_geom.to_crs(src_dem.crs)
                    else:
                        mask_geom_proj = mask_geom

                    try:
                        out_img, _ = mask(
                            dataset, 
                            mask_geom_proj.geometry, 
                            crop=True, 
                            nodata=0
                        )
                        lz_arr = out_img[0]
                        # Nota: Si recortamos, el shape cambia, pero para visualización rápida
                        # a veces es mejor mantener el canvas original y solo poner ceros.
                        # Aquí devolvemos el recortado para ahorrar proceso.
                    except Exception:
                        pass # Si falla el recorte, devolvemos el cuadro completo

        return lz_arr, dst_transform, src_dem.crs

    except Exception as e:
        # print(f"Error generando LZ Raster: {e}")
        return None, None, None


def calculate_climatic_indices(series_mensual, alt_m):
    """
    Calcula Índices de Aridez (Martonne) y Erosividad (Fournier).
    Robusto ante series incompletas.
    """
    if series_mensual is None or series_mensual.empty:
        return {}

    # Datos base
    # Asumimos que 'series_mensual' es una serie temporal larga o un ciclo de 12 meses.
    # P_total_anual_promedio = promedio_mensual * 12
    mean_monthly = series_mensual.mean()
    P_total = mean_monthly * 12
    
    # Temperatura estimada
    T_media = estimate_temperature(alt_m)

    # 1. Índice de Martonne (Aridez)
    # I_M = P / (T + 10)
    # Evitar división por cero
    denom = T_media + 10
    im = P_total / denom if denom != 0 else 0

    if im < 5: im_cat = "Desierto Absoluto"
    elif im < 10: im_cat = "Árido"
    elif im < 20: im_cat = "Semiárido"
    elif im < 30: im_cat = "Mediterráneo / Semihúmedo"
    elif im < 60: im_cat = "Húmedo"
    else: im_cat = "Perhúmedo"

    # 2. Índice de Fournier Modificado (Erosividad - MFI)
    # MFI = Sum(pi^2) / P_total
    try:
        # Si tiene índice fecha, agrupamos por mes para obtener el ciclo promedio
        if isinstance(series_mensual.index, pd.DatetimeIndex):
            monthly_means = series_mensual.groupby(series_mensual.index.month).mean()
        else:
            # Si no es fecha, asumimos que ya son valores mensuales representativos
            monthly_means = series_mensual 

        sum_p2 = (monthly_means**2).sum()
        mfi = sum_p2 / P_total if P_total > 0 else 0

        if mfi < 60: mfi_cat = "Muy Baja"
        elif mfi < 90: mfi_cat = "Baja"
        elif mfi < 120: mfi_cat = "Moderada"
        elif mfi < 160: mfi_cat = "Alta"
        else: mfi_cat = "Muy Alta"
    except:
        mfi = 0
        mfi_cat = "N/A"

    return {
        "martonne_val": im,
        "martonne_class": im_cat,
        "fournier_val": mfi,
        "fournier_class": mfi_cat,
        "temp_media": T_media,
    }


# ==============================================================================
# PARTE 5: ESTADÍSTICAS EXTREMAS Y RETORNO (FINAL)
# ==============================================================================

def calculate_return_periods(df_long, station_name):
    """
    Calcula períodos de retorno (Gumbel) sobre máximos anuales.
    Blindado contra cambios de nombres de columna.
    """
    if df_long.empty: return None, None

    # --- Detección de Columnas ---
    _, c_val = get_safe_cols(df_long)
    c_name = next((c for c in ['nombre', Config.STATION_NAME_COL, 'nom_est'] if c in df_long.columns), None)
    c_year = next((c for c in ['año', Config.YEAR_COL, 'year'] if c in df_long.columns), None)

    if not c_val or not c_name or not c_year: return None, "Error: Columnas no identificadas."

    # 1. Filtrar datos de la estación
    df_station = df_long[df_long[c_name] == station_name].copy()

    if df_station.empty: return None, None

    # 2. Obtener Máximos Anuales
    annual_max = df_station.groupby(c_year)[c_val].max().dropna()

    if len(annual_max) < 10:  # Mínimo estadístico
        return None, "Insuficientes datos anuales (<10 años) para ajuste de Gumbel."

    # 3. Ajuste Distribución Gumbel
    try:
        # params = (loc, scale)
        params = stats.gumbel_r.fit(annual_max)

        # 4. Calcular Valores para Tr específicos
        tr_list = [2, 5, 10, 25, 50, 100]
        probs = [1 - (1 / tr) for tr in tr_list]

        precip_values = stats.gumbel_r.ppf(probs, *params)

        df_results = pd.DataFrame({
            "Período de Retorno (Tr)": tr_list,
            "Probabilidad Excedencia": [f"{1/tr:.1%}" for tr in tr_list],
            "Ppt Máxima Esperada (mm)": precip_values,
        })

        return df_results, {"params": params, "data": annual_max}
    except Exception as e:
        return None, f"Error en ajuste estadístico: {e}"


def calculate_percentiles_extremes(df_long, station_name, p_low=10, p_high=90):
    """Identifica eventos por encima/debajo de percentiles."""
    if df_long.empty: return None

    # Detección
    _, c_val = get_safe_cols(df_long)
    c_name = next((c for c in ['nombre', Config.STATION_NAME_COL, 'nom_est'] if c in df_long.columns), None)
    
    if not c_val or not c_name: return None

    df_station = df_long[df_long[c_name] == station_name].copy()
    if df_station.empty: return None

    val = df_station[c_val].dropna()
    if val.empty: return None

    # Calcular umbrales
    thresh_low = np.percentile(val, p_low)
    thresh_high = np.percentile(val, p_high)

    # Filtrar eventos
    df_station["Tipo Evento"] = "Normal"
    df_station.loc[df_station[c_val] <= thresh_low, "Tipo Evento"] = f"Bajo (<P{p_low})"
    df_station.loc[df_station[c_val] >= thresh_high, "Tipo Evento"] = f"Alto (>P{p_high})"

    return df_station, thresh_low, thresh_high


def calculate_duration_curve(series_mensual, runoff_coeff, area_km2, q_base_m3s=0):
    """
    Calcula Curva de Duración (FDC) usando Caudal Total (Directo + Base).
    """
    if series_mensual is None or series_mensual.empty:
        return None

    # 1. Caudal Directo (Rápido)
    # Factor = Area(m2) / Tiempo(s)
    # 30.4375 días promedio por mes * 86400 seg/día
    factor = (area_km2 * 1000) / (30.4375 * 86400)
    q_rapid = series_mensual * runoff_coeff * factor
    
    # 2. Caudal Total (Modelo Aditivo)
    q_total = q_rapid + q_base_m3s

    # 3. Ordenar
    sorted_q = q_total.sort_values(ascending=False)
    n = len(sorted_q)
    if n < 5: return None

    # 4. Probabilidades (Weibull)
    probs = np.arange(1, n + 1) / (n + 1) * 100

    try:
        # Ajuste Logarítmico suele ajustar mejor que el polinómico para FDC,
        # pero mantenemos el polinómico si da buenos resultados visuales.
        coeffs = np.polyfit(probs, sorted_q.values, 3)
        
        eq_str = (
            f"Q = {coeffs[0]:.2e}P³ "
            f"{'+' if coeffs[1]>=0 else '-'} {abs(coeffs[1]):.2e}P² "
            f"{'+' if coeffs[2]>=0 else '-'} {abs(coeffs[2]):.2e}P "
            f"{'+' if coeffs[3]>=0 else '-'} {abs(coeffs[3]):.2e}"
        )
    except:
        eq_str = "N/A"

    return {
        "data": pd.DataFrame({"Probabilidad": probs, "Caudal": sorted_q.values}),
        "equation": eq_str
    }


def calculate_bias_correction_metrics(df_stations, df_satellite):
    """
    Calcula el sesgo (Bias) entre estaciones y satélite.
    """
    try:
        # Detectar columnas de nombre para el merge
        c_name_st = next((c for c in ['nombre', Config.STATION_NAME_COL, 'nom_est'] if c in df_stations.columns), None)
        c_name_sat = next((c for c in ['nombre', Config.STATION_NAME_COL, 'nom_est'] if c in df_satellite.columns), None)
        
        # Detectar columna de valor en estaciones
        _, c_val_st = get_safe_cols(df_stations)
        
        if not c_name_st or not c_name_sat or not c_val_st: return None

        # Unir por estación
        df_merge = pd.merge(
            df_stations[[c_name_st, c_val_st, "geometry"]],
            df_satellite[[c_name_sat, "ppt_sat"]],
            left_on=c_name_st,
            right_on=c_name_sat
        )

        if df_merge.empty: return None

        # Calcular métricas
        # Bias Factor (Multiplicativo)
        df_merge["bias_factor"] = df_merge[c_val_st] / df_merge["ppt_sat"].replace(0, 0.01)

        # Bias Diff (Aditivo)
        df_merge["bias_diff"] = df_merge[c_val_st] - df_merge["ppt_sat"]

        # Limpieza de outliers (Factores absurdos > 10 o < 0.1)
        df_merge = df_merge[df_merge["bias_factor"].between(0.1, 10)]

        return df_merge
    except Exception as e:
        # print(f"Error cálculo sesgo: {e}")
        return None


def calculate_hydrological_statistics(series_mensual, runoff_coeff, area_km2, q_base_m3s=0):
    """
    Calcula estadísticas hidrológicas extremas (Qmax, Qmin) 
    usando un MODELO ADITIVO (Superficial + Base).
    Incluye lógica de 'Suelo Físico' para sequías extremas.
    """
    stats_dict = {}
    
    # 1. Validación
    if series_mensual is None or series_mensual.empty:
        return {}
    
    # Asegurar índice datetime
    if not isinstance(series_mensual.index, pd.DatetimeIndex):
        try:
            series_mensual.index = pd.to_datetime(series_mensual.index)
        except:
            return {} 

    # 2. CONSTRUCCIÓN DE LA SERIE DE CAUDAL TOTAL
    # Q = Q_rápido + Q_base
    factor_conv = (area_km2 * 1000) / (30.4375 * 86400)
    q_rapid = series_mensual * runoff_coeff * factor_conv
    q_total_series = q_rapid + q_base_m3s

    # 3. ESTADÍSTICAS BÁSICAS
    try:
        stats_dict["Q_Medio"] = q_total_series.mean()
        stats_dict["Desviacion_Std"] = q_total_series.std()
        
        # Q95 (Caudal Ecológico)
        stats_dict["Q_Ecologico_Q95"] = q_total_series.quantile(0.05) 
        
        stats_dict["Q_Max_Historico"] = q_total_series.max()
        stats_dict["Q_Min_Historico"] = q_total_series.min()
    except:
        return {}

    # 4. PROYECCIONES PROBABILÍSTICAS
    if not HAS_SCIPY: return stats_dict

    try:
        # Agrupación Anual
        q_max_anual = q_total_series.resample('A').max().dropna()
        q_min_anual = q_total_series.resample('A').min().dropna()
        
        tr_list = [2.33, 5, 10, 25, 50, 100]

        # --- A. MÁXIMOS (Crecientes) -> GUMBEL ---
        if len(q_max_anual) >= 3:
            loc_max, scale_max = stats.gumbel_r.fit(q_max_anual)
            for tr in tr_list:
                prob = 1 - (1/tr)
                val = stats.gumbel_r.ppf(prob, loc_max, scale_max)
                stats_dict[f"Q_Max_{tr}a"] = max(0, val)
        else:
            for tr in tr_list: stats_dict[f"Q_Max_{tr}a"] = 0

        # --- B. MÍNIMOS (Sequías) -> LOG-NORMAL ---
        if len(q_min_anual) >= 3:
            # Factor de Agotamiento de Acuífero:
            # Incluso en sequía de 100 años, asumimos que el acuífero retiene 
            # al menos un % de su capacidad de descarga base.
            
            # Protección para logaritmo
            q_min_safe = q_min_anual.clip(lower=0.001) 
            
            log_q = np.log(q_min_safe)
            mu_log, sigma_log = stats.norm.fit(log_q)
            
            for tr in tr_list:
                prob = 1 / tr
                val_log = stats.norm.ppf(prob, mu_log, sigma_log)
                prediccion = np.exp(val_log)
                
                # SUELO FÍSICO (CONTROL DE REALIDAD)
                suelo_fisico = q_base_m3s * 0.20 
                
                stats_dict[f"Q_Min_{tr}a"] = max(prediccion, suelo_fisico)

        else:
            for tr in tr_list: stats_dict[f"Q_Min_{tr}a"] = 0

    except Exception:
        pass

    return stats_dict