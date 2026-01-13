import pandas as pd
import numpy as np

def calcular_recarga_simple(df_lluvia, coef_infiltracion):
    """
    Calcula la recarga potencial basada en un coeficiente de infiltración sobre datos mensuales.
    
    Args:
        df_lluvia (pd.DataFrame): DataFrame con columna 'valor' (lluvia mensual).
        coef_infiltracion (float): Porcentaje de agua que entra al suelo (0.0 a 1.0).
        
    Returns:
        pd.DataFrame: DataFrame con la columna 'recarga_estimada'.
    """
    df_resultado = df_lluvia.copy()
    
    # En datos mensuales, asumimos que el coeficiente aplica al total del mes
    # Ejemplo: Si llovió 100mm en el mes y el coeficiente es 0.20 (20%), recargan 20mm.
    df_resultado['recarga_estimada'] = df_resultado['valor'] * coef_infiltracion
    
    return df_resultado

def obtener_clasificacion_suelo(tipo_suelo):
    """
    Devuelve un coeficiente de infiltración aproximado según el tipo de suelo.
    """
    suelos = {
        "Arenoso (Alta Infiltración)": 0.50,
        "Franco (Media Infiltración)": 0.30,
        "Arcilloso (Baja Infiltración)": 0.10,
        "Urbano/Impermeable": 0.05
    }
    return suelos.get(tipo_suelo, 0.20)