import os
import sys

# 1. TRUCO PARA WINDOWS: Forzar la consola a usar UTF-8
try:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Definir la estructura
folders = [
    "pages",
    "modules",
    "data/climate",
    "data/hydrogeology",
    "data/biodiversity",
    "assets"
]

files = {
    "pages/01_🌦️_Clima_e_Hidrologia.py": "# Código movido del dashboard original\nimport streamlit as st\nst.title('Clima e Hidrología')",
    "pages/02_💧_Aguas_Subterraneas.py": "# Módulo de Aguas Subterráneas\nimport streamlit as st\nst.title('Aguas Subterráneas y Recarga')",
    "pages/03_🍃_Biodiversidad.py": "# Módulo de Biodiversidad\nimport streamlit as st\nst.title('Biodiversidad y Salud Ecosistémica')",
    "pages/04_📊_Toma_de_Decisiones.py": "# Módulo de Soporte a Decisiones\nimport streamlit as st\nst.title('Tablero de Control - GIRH')",
    "modules/__init__.py": "",
    "modules/hydrogeo_utils.py": "# Funciones para cálculos hidrogeológicos",
    "modules/bio_utils.py": "# Funciones para cálculos de biodiversidad"
}

print("--- Iniciando creación de estructura ---")

# Crear carpetas
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"[OK] Carpeta revisada: {folder}")

# Crear archivos vacíos
for path, content in files.items():
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        # Imprimimos un mensaje seguro sin el emoji en la consola para evitar errores
        print(f"[OK] Archivo creado: {path.encode('ascii', 'replace').decode()}") 
    else:
        print(f"[YA EXISTE] El archivo: {path.encode('ascii', 'replace').decode()}")

print("\n¡Estructura lista para SIHCLI-POTER 2026!")