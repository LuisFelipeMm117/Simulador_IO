# pages/1_Simulacion_de_Shocks.py
"""
Re-export de la página principal (app.py).
Streamlit requiere que las páginas estén en /pages/ con prefijo numérico.
La lógica real vive en app.py; esta página la importa y re-ejecuta.
"""
from app import main

main()

# Esta página no es necesaria si app.py ya es la raíz.
# Streamlit usa app.py como página 1 automáticamente.
