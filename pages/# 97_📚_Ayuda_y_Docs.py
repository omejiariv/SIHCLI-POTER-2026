# 97_📚_Ayuda_y_Docs.py

import streamlit as st
import os

st.set_page_config(page_title="Ayuda y Documentación", page_icon="📚", layout="wide")

st.title("📚 Centro de Documentación y Ayuda")
st.markdown("---")

tab1, tab2 = st.tabs(["📘 Documentación Técnica", "📖 Manual de Usuario"])

with tab1:
    st.header("Documentación Técnica del Sistema")
    st.info("Este documento detalla la arquitectura, tecnologías y estructura de datos de SIHCLI-POTER.")
    
    try:
        # Busca el archivo en la raíz
        with open("DOCUMENTACION_TECNICA.md", "r", encoding="utf-8") as f:
            doc_content = f.read()
            
        with st.expander("👁️ Ver Documentación en Pantalla", expanded=True):
            st.markdown(doc_content)
            
        st.download_button(
            label="📥 Descargar Documentación Técnica (.md)",
            data=doc_content,
            file_name="SIHCLI_POTER_Docs_Tecnica.md",
            mime="text/markdown"
        )
    except FileNotFoundError:
        st.warning("⚠️ El archivo 'DOCUMENTACION_TECNICA.md' no se encontró en la carpeta raíz.")

with tab2:
    st.header("Guía de Usuario")
    st.write("Bienvenido al manual de operación de SIHCLI-POTER.")
    st.info("🚧 El Manual PDF detallado está en construcción.")
    
    st.subheader("Preguntas Frecuentes")
    with st.expander("¿Cómo subir nuevos datos de lluvia?"):
        st.write("""
        1. Ve al menú lateral: **'👑 Panel Administracion'**.
        2. Ingresa tus credenciales seguras.
        3. Pestaña **'Estaciones & Lluvias'** -> **'Carga Masiva (CSV)'**.
        4. Sube el archivo y sigue el asistente inteligente.
        """)

