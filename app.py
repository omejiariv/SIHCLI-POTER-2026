# app.py
import streamlit as st

# Configuración de página (debe ser la primera línea)
st.set_page_config(
    page_title="SIHCLI-POTER 2026",
    page_icon="🌊",
    layout="wide"
)

# Título y Bienvenida
st.title("🌊 SIHCLI-POTER 2026")
st.markdown("### Sistema de Información Hidro-Climatológica y Eco-Hidrológica")
st.markdown("**Corporación CuencaVerde | Fondo de Agua de Medellín y la Región Central**")

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.info("👋 **Bienvenido al nuevo sistema integrado.**")
    st.markdown("""
    Esta plataforma ha evolucionado para integrar nuevos módulos estratégicos.
    
    **👈 Usa el menú lateral para navegar entre:**
    
    * **01 🌦️ Clima e Hidrología:** Tu tablero de monitoreo actual.
    * **02 💧 Aguas Subterráneas:** (Nuevo) Recarga y acuíferos.
    * **03 🍃 Biodiversidad:** (Nuevo) Ecosistemas.
    * **04 📊 Toma de Decisiones:** (Nuevo) Gestión.
    """)

with col2:
    st.success("🎯 **Objetivo 2026**")
    st.markdown("""
    > *"Gestionar integralmente el recurso hídrico entendiendo la cuenca 
    > como un sistema vivo."*
    """)
    st.warning("⚠️ Nota: Si no ves el menú lateral de páginas, haz clic en la flecha pequeña `>` en la esquina superior izquierda.")

st.divider()