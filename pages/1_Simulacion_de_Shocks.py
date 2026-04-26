# pages/1_Simulacion_de_Shocks.py
"""
Re-export de la página principal (app.py).
Streamlit requiere que las páginas estén en /pages/ con prefijo numérico.
La lógica real vive en app.py; esta página la importa y re-ejecuta.
"""
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; color: #555; }
    .block-container { padding-top: 1.5rem; }
    .stAlert { border-radius: 8px; }

    /* 🔴 OCULTAR BOTONES DE GITHUB / SHARE */
    header {visibility: hidden;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stStatusWidget"] {display: none;}

    /* Opcional: también quita el footer de Streamlit */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

from app import main

main()

# Esta página no es necesaria si app.py ya es la raíz.
# Streamlit usa app.py como página 1 automáticamente.
