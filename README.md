# SIHCLI-POTER

Sistema de Información Hidroclimática — Interfaz y utilidades para procesamiento y visualización de datos hidrometeorológicos.

Resumen
- Proyecto: SIHCLI-POTER
- Lenguaje: Python
- Licencia: GPL-3.0
- Objetivo: Proveer una aplicación (Streamlit) para visualizar y procesar información hidroclimática y herramientas de migración/ETL.

Estado actual
- app.py y migracion.py como puntos de entrada.
- modules/ con lógica del proyecto.
- Configuración de entorno mínima (config.yaml, .devcontainer, .streamlit).
- Falta documentación operativa, tests automatizados y pipelines de CI/CD (este commit añade un primer pipeline y herramientas de desarrollo).

Requisitos
- Python 3.10+
- (Opcional) Docker para despliegue en contenedor

Instalación rápida (local)
1. Clonar y crear entorno virtual:
```bash
git clone https://github.com/omejiariv/SIHCLI-POTER.git
cd SIHCLI-POTER
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Ejecutar la aplicación (Streamlit):
```bash
streamlit run app.py
```

Comandos útiles de desarrollo
- Formateo (Black):
```bash
black .
```
- Linter (Ruff):
```bash
ruff .
```
- Ejecutar tests:
```bash
pytest
```

CI / Integración continua
- Se añade un workflow básico en .github/workflows/ci.yml que:
  - Instala dependencias
  - Ejecuta ruff (lint)
  - Ejecuta black --check (formato)
  - Ejecuta pytest y genera reporte de cobertura
  - Ejecuta pre-commit (comprobación)

Buenas prácticas recomendadas
- Ejecuta pre-commit localmente antes de crear PRs:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
- No subir credenciales al repositorio. Usa secretos de CI para valores sensibles.

Estructura sugerida (futura refactorización)
- sihcli/
  - __init__.py
  - cli.py
  - api.py
  - db.py
  - etl.py
  - ui/ (streamlit)
- tests/
- .github/workflows/ (CI)

Cómo contribuir
1. Abre un issue describiendo el cambio.
2. Crea una rama feature/xxx desde main.
3. Añade tests cuando agregues comportamiento.
4. Abre PR apuntando a main; el pipeline CI se ejecutará automáticamente.

Licencia
Este proyecto está licenciado bajo GPL-3.0. Consulta el archivo LICENSE para más detalles.
