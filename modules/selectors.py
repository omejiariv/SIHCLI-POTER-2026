# ... (dentro de la función render_selector_espacial) ...
    elif modo == "⛰️ Por Cuenca":
        # ... (código de selección de nombre de cuenca igual que antes) ...
        
        if sel:
            gdf_cuenca_sel = gdf_subcuencas[gdf_subcuencas[col_nom] == sel]
            gdf_area_out = gdf_cuenca_sel
            
            # --- NUEVO: SLIDER DE BUFFER (Radio de Búsqueda) ---
            # Permite incluir estaciones vecinas para mejorar la interpolación en los bordes
            buffer_km = st.sidebar.slider("Radio de Búsqueda (Buffer km):", 0, 50, 15, help="Incluye estaciones vecinas para mejorar la interpolación en los bordes.")
            
            # Convertir geometría a proyectada para buffer en metros (aprox)
            # Usamos una proyección métrica temporal (Web Mercator o Magna Sirgas)
            gdf_buffer = gdf_cuenca_sel.to_crs("EPSG:3857").buffer(buffer_km * 1000).to_crs("EPSG:4326")
            
            # CRUCE ESPACIAL CON EL BUFFER
            if gdf_stations.crs != gdf_buffer.crs:
                gdf_stations = gdf_stations.to_crs(gdf_buffer.crs)
                
            # Usamos el buffer para filtrar las estaciones
            estaciones_dentro = gpd.sjoin(gdf_stations, gpd.GeoDataFrame(geometry=gdf_buffer), predicate='within')
            
            if not estaciones_dentro.empty:
                col_id = 'id_estacion' if 'id_estacion' in estaciones_dentro.columns else 'codigo'
                ids_out = estaciones_dentro[col_id].tolist()
                altitud_out = estaciones_dentro[Config.ALTITUDE_COL].mean()
                
                # Feedback visual
                n_total = len(ids_out)
                st.sidebar.success(f"✅ {n_total} estaciones (Vecinas incl.)")
            else:
                st.sidebar.warning("⚠️ Sin estaciones en el radio de búsqueda.")
            
            nombre_out = f"Cuenca {sel}"