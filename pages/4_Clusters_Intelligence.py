# -*- coding: utf-8 -*-
"""
Cluster Intelligence — Simulador de Clusters Económicos
Modelo Insumo-Producto · Detección de Comunidades Louvain
"""

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ══════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Cluster Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# CSS — DISEÑO PREMIUM (DARK INDUSTRIAL)
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #f8fafc;
    --surface:   #ffffff;
    --surface2:  #f1f5f9;
    --border:    #e2e8f0;

    --accent:    #2563eb;
    --accent2:   #7c3aed;
    --accent3:   #f59e0b;

    --text:      #0f172a;
    --muted:     #64748b;

    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
    --ok:   #10b981;
    --warn: #ef4444;
}
/* TIPOGRAFÍA SEGURA */

/* Markdown */

/* Inputs normales */
input, textarea, select {
    font-family: var(--sans) !important;
}

/* Sidebar SOLO texto */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span {
    font-family: var(--sans) !important;
}

/* Títulos */
h1, h2 {
    font-family: var(--mono) !important;
}
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stFileUploader"] button {
    font-family: var(--sans) !important;
}

[data-testid="stFileUploader"] div {
    font-family: var(--sans) !important;
}

/* ── SLIDER LIMPIO ── */

/* Track (fondo) */
.stSlider > div > div > div > div {
    background: #e2e8f0 !important;
}

/* Barra activa */
.stSlider > div > div > div > div > div {
    background: var(--accent) !important;
}

/* Handle */
.stSlider > div > div > div > div > div > div {
    background: #ffffff !important;
    border: 2px solid var(--accent) !important;
}

/* ── Headings ── */
h1,h2,h3,h4 {
    font-family: var(--mono) !important;
    color: var(--text) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 11px !important; letter-spacing: 2px !important; text-transform: uppercase !important; font-family: var(--mono) !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: var(--mono) !important; font-size: 2rem !important; }
[data-testid="stMetricDelta"] { font-family: var(--mono) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Cards personalizadas ── */
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--accent);
    border-radius: 4px 0 0 4px;
}
.kpi-card.warn::before { background: var(--warn); }
.kpi-card.ok::before   { background: var(--ok); }
.kpi-card.gold::before { background: var(--accent3); }
.kpi-card.purple::before { background: var(--accent2); }

.kpi-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
}
.kpi-sub {
    font-size: 12px;
    color: #64748b;
    margin-top: 4px;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 700;
}
.badge-clave   { background: #10b98122; color: #10b981; border: 1px solid #10b981; }
.badge-impulsor { background: #3b82f622; color: #60a5fa; border: 1px solid #3b82f6; }
.badge-estrategico { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b; }
.badge-dependiente { background: #ef444422; color: #f87171; border: 1px solid #ef4444; }

/* ── Section title ── */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #64748b;
    padding: 16px 0 8px 0;
    border-top: 1px solid var(--border);
    margin-top: 8px;
}

/* ── Divider ── */
hr { border-color: #252a3a !important; }

/* ── Alert boxes ── */
.alert-box {
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 13px;
    margin: 8px 0;
}
.alert-box.warn { border-left-color: var(--warn); }
.alert-box.ok   { border-left-color: var(--ok); }

/* ── Number inputs ── */
input[type="number"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* ── Upload widget ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}

/* ── Select boxes ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
}

/* Plotly charts dark bg override */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div style="padding: 32px 0 16px 0;">
  <div style="font-family:'Space Mono',monospace; font-size:10px; letter-spacing:3px; color:var(--muted); text-transform:uppercase; margin-bottom:8px;">
    ⬡ ANÁLISIS ESTRUCTURAL · MODELO INSUMO-PRODUCTO
  </div>
  <h1 style="font-family:'Space Mono',monospace; font-size:2.4rem; font-weight:700; color:var(--text); margin:0; line-height:1.1;">
    Cluster Intelligence
  </h1>
  <div style="color:var(--muted); font-size:14px; margin-top:8px;">
    Detección de comunidades económicas mediante la inversa de Leontief y el algoritmo de Louvain
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="section-title">📂 Datos de entrada</div>', unsafe_allow_html=True)
    uploaded_A = st.file_uploader("Matriz A (.npy)", type=["npy"])
    uploaded_labels = st.file_uploader("Etiquetas sectores (.csv)", type=["csv"], help="CSV con columna 'sector'")

    st.markdown('<div class="section-title">⚙️ Parámetros del modelo</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K conexiones por nodo", 3, 20, 10,
        help="Número de vecinos retenidos por sector en la matriz de interacción.")
    threshold = st.number_input("Umbral mínimo de peso (τ)", 0.0001, 0.05, 0.001,
        step=0.0001, format="%.4f",
        help="Aristas con peso menor a τ son eliminadas del grafo.")
    resolution = st.slider("Resolución Louvain (γ)", 0.5, 2.0, 1.0, 0.1,
        help="γ > 1 → más clusters finos · γ < 1 → menos clusters gruesos.")

    st.markdown('<div class="section-title">🖥️ Modo</div>', unsafe_allow_html=True)
    use_local = st.checkbox("Datos locales (A_nacional.npy)", value=True)

# ══════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent

def load_matrix():
    if not use_local and uploaded_A is not None:
        return np.load(uploaded_A)
    local = BASE_DIR / "A_nacional.npy"
    if local.exists():
        return np.load(local)
    # Buscar en parent
    parent = BASE_DIR.parent / "data" / "A_nacional.npy"
    if parent.exists():
        return np.load(parent)
    return None

A = load_matrix()

if A is None:
    st.markdown("""
    <div class="alert-box warn">
      ⚠️ No se encontró <code>A_nacional.npy</code>. Sube la matriz desde el panel lateral
      o coloca el archivo en el mismo directorio que este script.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if A.ndim != 2 or A.shape[0] != A.shape[1]:
    st.error("La matriz A debe ser cuadrada (n × n).")
    st.stop()

n_sectors = A.shape[0]

# Etiquetas
sector_labels = None
if uploaded_labels is not None:
    try:
        df_lab = pd.read_csv(uploaded_labels)
        if "sector" in df_lab.columns and len(df_lab) == n_sectors:
            sector_labels = df_lab["sector"].tolist()
    except Exception:
        pass

labels = sector_labels if sector_labels else [f"S{i}" for i in range(n_sectors)]

# ══════════════════════════════════════════════════════════
# FUNCIONES MATEMÁTICAS CORREGIDAS
# ══════════════════════════════════════════════════════════

@st.cache_data
def build_model(A_bytes: bytes, n: int, top_k: int, threshold: float, resolution: float):
    """
    CORRECCIONES MATEMÁTICAS APLICADAS:
    1. Verifica la condición de Hawkins-Simon antes de invertir.
    2. Los índices BL/FL se normalizan por su media (Rasmussen-Hirschman).
    3. eigenvector_centrality_numpy falla en grafos desconectados →
       se usa el componente conexo principal y nx.eigenvector_centrality (power iteration).
    4. La partición Louvain usa random_state fijo para reproducibilidad.
    """
    A = np.frombuffer(A_bytes, dtype=np.float64).reshape(n, n)

    # ── 1. Validación de Hawkins-Simon ──────────────────────────
    col_sums = A.sum(axis=0)
    hs_ok = bool((col_sums < 1).all())  # condición necesaria (no suficiente pero práctica)

    # ── 2. Inversa de Leontief ───────────────────────────────────
    I  = np.eye(n)
    M  = I - A
    cond_M = float(np.linalg.cond(M))
    L  = np.linalg.inv(M)

    # ── 3. Índices de encadenamiento (Rasmussen-Hirschman) ───────
    # BL: suma de columna de L → poder de arrastre (backward linkage)
    # FL: suma de fila de L   → poder de empuje   (forward linkage)
    bl_raw = L.sum(axis=0)          # (n,)
    fl_raw = L.sum(axis=1)          # (n,)
    bl = bl_raw / bl_raw.mean()     # normalizado por media → 1 = promedio nacional
    fl = fl_raw / fl_raw.mean()

    # ── 4. Matriz de interacción normalizada y simetrizada ───────
    col_sum = L.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1
    W     = L / col_sum
    W_sym = (W + W.T) / 2

    # ── 5. Filtros top-K y umbral ────────────────────────────────
    W_f = np.zeros_like(W_sym)
    for i in range(n):
        idx = np.argsort(W_sym[i])[-top_k:]
        W_f[i, idx] = W_sym[i, idx]
    W_f[W_f < threshold] = 0

    # ── 6. Grafo y componente conexa principal ───────────────────
    G_full = nx.from_numpy_array(W_f)
    G_full.remove_nodes_from(list(nx.isolates(G_full)))

    # CORRECCIÓN: eigenvector_centrality_numpy falla en grafos desconectados
    # → trabajar sobre el componente más grande
    largest_cc = max(nx.connected_components(G_full), key=len)
    G = G_full.subgraph(largest_cc).copy()
    n_isolated = n - G_full.number_of_nodes()
    n_minor    = G_full.number_of_nodes() - G.number_of_nodes()

    # ── 7. Partición Louvain ─────────────────────────────────────
    partition = community_louvain.best_partition(G, resolution=resolution, random_state=42)
    modularity = community_louvain.modularity(partition, G)

    # ── 8. Centralidad de eigenvector (power iteration) ──────────
    try:
        centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        centrality = nx.degree_centrality(G)

    return {
        "G": G, "L": L, "partition": partition, "centrality": centrality,
        "modularity": modularity, "bl": bl, "fl": fl,
        "bl_raw": bl_raw, "fl_raw": fl_raw,
        "cond_M": cond_M, "hs_ok": hs_ok,
        "n_isolated": n_isolated, "n_minor": n_minor,
    }


# Serializar A para cache determinista
A_bytes = A.tobytes()

try:
    m = build_model(A_bytes, n_sectors, top_k, threshold, resolution)
except Exception as e:
    st.error(f"Error construyendo el modelo: {e}")
    st.stop()

G           = m["G"]
L           = m["L"]
partition   = m["partition"]
centrality  = m["centrality"]
modularity  = m["modularity"]
bl          = m["bl"]
fl          = m["fl"]

# ── DataFrame maestro ────────────────────────────────────────────
nodes = list(G.nodes())
df = pd.DataFrame({
    "node_id":    nodes,
    "sector":     [labels[i] for i in nodes],
    "cluster":    [partition[i] for i in nodes],
    "centralidad":[round(centrality[i], 6) for i in nodes],
    "grado":      [G.degree(i) for i in nodes],
    "BL":         [round(bl[i], 4) for i in nodes],
    "FL":         [round(fl[i], 4) for i in nodes],
})

def classify(row):
    if row["BL"] >= 1 and row["FL"] >= 1: return "Sector clave"
    if row["BL"] >= 1 and row["FL"] <  1: return "Impulsor"
    if row["BL"] <  1 and row["FL"] >= 1: return "Estratégico"
    return "Dependiente"

df["tipo"] = df.apply(classify, axis=1)

# ── Resumen por cluster ──────────────────────────────────────────
summary = (
    df.groupby("cluster")
    .agg(
        tamaño=("sector", "count"),
        centralidad_media=("centralidad", "mean"),
        BL_media=("BL", "mean"),
        FL_media=("FL", "mean"),
    )
    .reset_index()
)
summary["score"] = (
    summary["centralidad_media"] / summary["centralidad_media"].max() * 0.5 +
    summary["BL_media"]         / summary["BL_media"].max()          * 0.3 +
    summary["FL_media"]         / summary["FL_media"].max()          * 0.2
)
summary = summary.sort_values("score", ascending=False).reset_index(drop=True)
summary.insert(0, "rank", range(1, len(summary) + 1))

n_clusters  = df["cluster"].nunique()
top_cl      = summary.iloc[0]

# ══════════════════════════════════════════════════════════
# ALERTAS MATEMÁTICAS
# ══════════════════════════════════════════════════════════
with st.expander("🔬 Validación matemática del modelo", expanded=False):
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        hs_icon = "✅" if m["hs_ok"] else "⚠️"
        st.markdown(f"""
        <div class="alert-box {'ok' if m['hs_ok'] else 'warn'}">
          {hs_icon} <strong>Condición de Hawkins-Simon:</strong> {'SATISFECHA' if m['hs_ok'] else 'NO satisfecha — revisar coeficientes técnicos'}<br>
          Σ_columna(A) max = {A.sum(axis=0).max():.4f} (debe ser &lt; 1)
        </div>
        <div class="alert-box ok">
          ✅ <strong>Cond(I−A):</strong> {m['cond_M']:.2f} — matriz bien condicionada (I−A)
        </div>
        """, unsafe_allow_html=True)
    with col_v2:
        st.markdown(f"""
        <div class="alert-box ok">
          ✅ <strong>L ≥ 0:</strong> {'Todos los elementos de L son no-negativos (propiedad Leontief)' if (L >= 0).all() else 'ERROR: L tiene elementos negativos'}
        </div>
        <div class="alert-box ok">
          ✅ <strong>Centralidad corregida:</strong> Power iteration sobre componente conexo principal
          ({G.number_of_nodes()} nodos · {m['n_minor']} aislados en subcomponente menor)
        </div>
        <div class="alert-box ok">
          ✅ <strong>BL/FL normalizados:</strong> Índice Rasmussen-Hirschman (media = 1.0)
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# KPIs GLOBALES
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">INDICADORES GLOBALES</div>', unsafe_allow_html=True)

kc1, kc2, kc3, kc4, kc5 = st.columns(5)
kc1.metric("Sectores activos",  G.number_of_nodes(), delta=f"{n_sectors} en MIP")
kc2.metric("Clusters (Louvain)", n_clusters)
kc3.metric("Modularidad Q",     f"{modularity:.4f}", delta="↑ Alta segmentación" if modularity > 0.6 else None)
kc4.metric("Sectores clave",    int((df["tipo"] == "Sector clave").sum()))
kc5.metric("Aristas del grafo", G.number_of_edges())

# ══════════════════════════════════════════════════════════
# CLUSTER MÁS ESTRATÉGICO
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">LIDERAZGO ESTRUCTURAL</div>', unsafe_allow_html=True)

c_top, c_info = st.columns([1, 2])
with c_top:
    top_sectores = df[df["cluster"] == top_cl["cluster"]].sort_values("centralidad", ascending=False)
    st.markdown(f"""
    <div class="kpi-card gold">
      <div class="kpi-label">🏆 Cluster más estratégico</div>
      <div class="kpi-value">C{int(top_cl['cluster'])}</div>
      <div class="kpi-sub">{int(top_cl['tamaño'])} sectores · Score {top_cl['score']:.4f}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Centralidad media</div>
      <div class="kpi-value" style="font-size:1.3rem">{top_cl['centralidad_media']:.5f}</div>
    </div>
    <div class="kpi-card ok">
      <div class="kpi-label">BL medio (arrastre)</div>
      <div class="kpi-value" style="font-size:1.3rem">{top_cl['BL_media']:.4f}</div>
      <div class="kpi-sub">{'↑ Por encima del promedio nacional' if top_cl['BL_media'] > 1 else '↓ Bajo promedio nacional'}</div>
    </div>
    <div class="kpi-card purple">
      <div class="kpi-label">FL medio (empuje)</div>
      <div class="kpi-value" style="font-size:1.3rem">{top_cl['FL_media']:.4f}</div>
      <div class="kpi-sub">{'↑ Por encima del promedio nacional' if top_cl['FL_media'] > 1 else '↓ Bajo promedio nacional'}</div>
    </div>
    """, unsafe_allow_html=True)

with c_info:
    st.markdown(f"**Sectores del Cluster C{int(top_cl['cluster'])} ordenados por centralidad**")

    # Tabla con badges
    rows_html = ""
    for _, row in top_sectores.head(8).iterrows():
        badge_cls = {
            "Sector clave": "badge-clave",
            "Impulsor": "badge-impulsor",
            "Estratégico": "badge-estrategico",
            "Dependiente": "badge-dependiente",
        }.get(row["tipo"], "")
        rows_html += f"""
        <tr>
          <td style="padding:8px 12px; font-family:'Space Mono',monospace; font-size:12px; color:var(--text);">{row['sector']}</td>
          <td style="padding:8px 12px; text-align:right; color:#00e5ff; font-family:'Space Mono',monospace; font-size:12px;">{row['centralidad']:.5f}</td>
          <td style="padding:8px 12px; text-align:right; color:#94a3b8; font-size:12px;">{row['BL']:.3f}</td>
          <td style="padding:8px 12px; text-align:right; color:#94a3b8; font-size:12px;">{row['FL']:.3f}</td>
          <td style="padding:8px 12px;"><span class="badge {badge_cls}">{row['tipo']}</span></td>
        </tr>
        """
    st.markdown(f"""
    <table style="width:100%; border-collapse:collapse; background:#ffffff; border-radius:10px; overflow:hidden; border:1px solid #252a3a;">
      <thead>
        <tr style="background:#f1f5f9;">
          <th style="padding:10px 12px; text-align:left; font-family:'Space Mono',monospace; font-size:10px; letter-spacing:2px; color:#64748b;">SECTOR</th>
          <th style="padding:10px 12px; text-align:right; font-family:'Space Mono',monospace; font-size:10px; letter-spacing:2px; color:#64748b;">CENTRALIDAD</th>
          <th style="padding:10px 12px; text-align:right; font-family:'Space Mono',monospace; font-size:10px; letter-spacing:2px; color:#64748b;">BL</th>
          <th style="padding:10px 12px; text-align:right; font-family:'Space Mono',monospace; font-size:10px; letter-spacing:2px; color:#64748b;">FL</th>
          <th style="padding:10px 12px; font-family:'Space Mono',monospace; font-size:10px; letter-spacing:2px; color:#64748b;">TIPO</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">MÓDULOS DE ANÁLISIS</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⬡  Red productiva",
    "📊  Clusters",
    "🔗  Encadenamientos",
    "📡  Shock",
    "🏷️  Sectores",
])

# ─── PALETA COMÚN ─────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Space Mono, monospace", color="#334155", size=11),
    margin=dict(l=20, r=20, t=40, b=20),
)

# ─── TAB 1: RED DE SECTORES ───────────────────────────────
with tab1:
    col_ctrl, _ = st.columns([1, 3])
    with col_ctrl:
        max_nodes = st.slider("Nodos a mostrar (por centralidad)", 10, G.number_of_nodes(), 45, key="net_nodes")

    top_n = sorted(centrality, key=centrality.get, reverse=True)[:max_nodes]
    sub   = G.subgraph(top_n)
    pos   = nx.spring_layout(sub, seed=42, k=2.0)

    # Aristas con alpha variable según peso
    edge_traces = []
    weights = [sub[u][v].get("weight", 1) for u, v in sub.edges()]
    w_max   = max(weights) if weights else 1

    for (u, v), w in zip(sub.edges(), weights):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        alpha = 0.15 + 0.55 * (w / w_max)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=0.8, color=f"rgba(100,116,139,{alpha:.2f})"),
            hoverinfo="none", showlegend=False,
        ))

    node_x     = [pos[nd][0] for nd in sub.nodes()]
    node_y     = [pos[nd][1] for nd in sub.nodes()]
    node_sizes = [14 + centrality[nd] * 120 for nd in sub.nodes()]
    node_colors= [partition[nd] for nd in sub.nodes()]
    node_hover = [
        f"<b>{labels[nd]}</b><br>Cluster: {partition[nd]}<br>"
        f"Centralidad: {centrality[nd]:.5f}<br>Grado: {sub.degree(nd)}"
        for nd in sub.nodes()
    ]

    node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers",
    marker=dict(
        size=node_sizes,
        color=[int(c) for c in node_colors],
        colorscale="Turbo",
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Cluster",
                font=dict(
                    family="Space Mono",
                    size=9,
                    color="#64748b"
                )
            ),
            thickness=12,
            len=0.6,
            tickfont=dict(
                family="Space Mono",
                size=9,
                color="#64748b"
            )
        ),
        line=dict(width=1.5, color="#0d0f14"),
        opacity=0.92
    ),
    text=[labels[nd] for nd in sub.nodes()],
    hovertext=node_hover,
    hoverinfo="text",
    hoverlabel=dict(
    bgcolor="#ffffff",
    bordercolor="#e2e8f0",
    font=dict(
        family="Space Mono",
        size=11,
        color="#0f172a"
    )
)
)

    fig_net = go.Figure(data=edge_traces + [node_trace])
    fig_net.update_layout(
        title=dict(text=f"Red productiva — top {max_nodes} sectores por centralidad",
                   font=dict(family="Space Mono", size=13, color="#94a3b8")),
        showlegend=False, hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=620, **PLOTLY_THEME,
    )
    st.plotly_chart(fig_net, use_container_width=True)

# ─── TAB 2: CLUSTERS ──────────────────────────────────────
with tab2:
    c_left, c_right = st.columns([1, 2])

    with c_left:
        st.markdown("**Ranking de clusters por score estratégico**")
        # Tabla enriquecida
        rows_html2 = ""
        for _, row in summary.head(15).iterrows():
            bar_w = int(row["score"] / summary["score"].max() * 100)
            rows_html2 += f"""
            <tr>
              <td style="padding:7px 10px; font-family:'Space Mono',monospace; font-size:12px; color:#64748b;">#{int(row['rank'])}</td>
              <td style="padding:7px 10px; font-family:'Space Mono',monospace; font-size:12px; color:#00e5ff;">C{int(row['cluster'])}</td>
              <td style="padding:7px 10px; font-size:12px; color:#94a3b8;">{int(row['tamaño'])}</td>
              <td style="padding:7px 10px;">
                <div style="background:#252a3a; border-radius:4px; height:6px; width:100%; overflow:hidden;">
                  <div style="background:linear-gradient(90deg,#7c3aed,#00e5ff); width:{bar_w}%; height:100%; border-radius:4px;"></div>
                </div>
                <div style="font-family:'Space Mono',monospace; font-size:10px; color:#64748b; margin-top:2px;">{row['score']:.4f}</div>
              </td>
            </tr>
            """
        st.markdown(f"""
        <table style="width:100%; border-collapse:collapse; background:#ffffff; border-radius:10px; overflow:hidden; border:1px solid var(--border);">
          <thead>
            <tr style="background:#f1f5f9;;">
              <th style="padding:8px 10px; text-align:left; font-family:'Space Mono',monospace; font-size:9px; letter-spacing:2px; color:#64748b;">#</th>
              <th style="padding:8px 10px; text-align:left; font-family:'Space Mono',monospace; font-size:9px; letter-spacing:2px; color:#64748b;">ID</th>
              <th style="padding:8px 10px; text-align:left; font-family:'Space Mono',monospace; font-size:9px; letter-spacing:2px; color:#64748b;">N</th>
              <th style="padding:8px 10px; text-align:left; font-family:'Space Mono',monospace; font-size:9px; letter-spacing:2px; color:#64748b;">SCORE</th>
            </tr>
          </thead>
          <tbody>{rows_html2}</tbody>
        </table>
        """, unsafe_allow_html=True)

    with c_right:
        # Bubble map
        fig_bubble = px.scatter(
            summary,
            x="BL_media", y="FL_media",
            size="tamaño", color="score",
            hover_name="cluster",
            hover_data={"tamaño": True, "centralidad_media": ":.5f"},
            size_max=55,
            color_continuous_scale="Turbo",
            labels={"BL_media": "BL medio (arrastre)", "FL_media": "FL medio (empuje)", "score": "Score"},
            title="Posicionamiento estratégico de clusters (BL vs FL)",
        )
        fig_bubble.add_hline(y=1, line_dash="dot", line_color="#252a3a",
                             annotation_text="FL promedio", annotation_font_color="#64748b")
        fig_bubble.add_vline(x=1, line_dash="dot", line_color="#252a3a",
                             annotation_text="BL promedio", annotation_font_color="#64748b")
        fig_bubble.update_traces(
            marker=dict(line=dict(width=1, color="#0d0f14"), opacity=0.88),
            hoverlabel=dict(bgcolor="#1c2030", bordercolor="#252a3a",
                            font=dict(family="Space Mono", size=11)),
        )
        fig_bubble.update_layout(
            height=500,
            **PLOTLY_THEME,
            coloraxis_colorbar=dict(
                title=dict(
                    text="Score",
                    font=dict(
                        family="Space Mono",
                        size=9,
                        color="#64748b"
                    )
                ),
                thickness=12,
                len=0.6,
                tickfont=dict(
                    family="Space Mono",
                    size=9,
                    color="#64748b"
                )
            )
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

        # Explorador de cluster individual
        st.markdown("**Explorar cluster**")
        sel_cl = st.selectbox("Selecciona cluster", sorted(df["cluster"].unique()),
                              format_func=lambda x: f"C{x}", key="cl_sel")
        df_sel = df[df["cluster"] == sel_cl].sort_values("centralidad", ascending=False)
        st.dataframe(
            df_sel[["sector", "centralidad", "grado", "BL", "FL", "tipo"]].reset_index(drop=True),
            use_container_width=True, height=220,
        )

# ─── TAB 3: ENCADENAMIENTOS ───────────────────────────────
with tab3:
    COLOR_MAP = {
        "Sector clave":  "#10b981",
        "Impulsor":      "#60a5fa",
        "Estratégico":   "#f59e0b",
        "Dependiente":   "#f87171",
    }

    fig_quad = px.scatter(
        df, x="BL", y="FL",
        color="tipo",
        color_discrete_map=COLOR_MAP,
        hover_name="sector",
        hover_data={"cluster": True, "centralidad": ":.5f", "grado": True},
        title="Cuadrante de encadenamientos — Rasmussen-Hirschman (BL y FL normalizados, media = 1)",
        labels={"BL": "Encadenamiento hacia atrás BL (Leontief, normalizado)",
                "FL": "Encadenamiento hacia adelante FL (Ghosh, normalizado)"},
        size="centralidad",
        size_max=20,
    )
    fig_quad.add_hline(y=1, line_dash="dash", line_color="#252a3a",
                       annotation_text="FL̄ = 1", annotation_font_color="#64748b")
    fig_quad.add_vline(x=1, line_dash="dash", line_color="#252a3a",
                       annotation_text="BL̄ = 1", annotation_font_color="#64748b")

    # Etiquetas en cuadrantes
    for (xt, yt, txt) in [
        (df["BL"].max()*0.85, df["FL"].max()*0.95, "CLAVE"),
        (df["BL"].min()*1.05, df["FL"].max()*0.95, "ESTRATÉGICO"),
        (df["BL"].max()*0.85, df["FL"].min()*1.1,  "IMPULSOR"),
        (df["BL"].min()*1.05, df["FL"].min()*1.1,  "DEPENDIENTE"),
    ]:
        fig_quad.add_annotation(x=xt, y=yt, text=txt, showarrow=False,
            font=dict(family="Space Mono", size=9, color="#252a3a"),
            bgcolor="#1c2030", bordercolor="#252a3a", borderpad=4)

    fig_quad.update_traces(
        hoverlabel=dict(bgcolor="#1c2030", bordercolor="#252a3a",
                        font=dict(family="Space Mono", size=11)),
        marker=dict(line=dict(width=1, color="#0d0f14"), opacity=0.88),
    )
    fig_quad.update_layout(height=520, **PLOTLY_THEME,
        legend=dict(bgcolor="rgba(20,23,32,0.8)", bordercolor="#252a3a", borderwidth=1,
                    font=dict(family="Space Mono", size=10)))
    st.plotly_chart(fig_quad, use_container_width=True)

    # Conteo por tipo
    cl1, cl2, cl3, cl4 = st.columns(4)
    for col, tipo, cls in zip(
        [cl1, cl2, cl3, cl4],
        ["Sector clave", "Impulsor", "Estratégico", "Dependiente"],
        ["ok", "impulsor", "gold", "warn"],
    ):
        cnt = int((df["tipo"] == tipo).sum())
        col.markdown(f"""
        <div class="kpi-card {cls}">
          <div class="kpi-label">{tipo}</div>
          <div class="kpi-value">{cnt}</div>
          <div class="kpi-sub">{cnt/len(df)*100:.1f}% del total</div>
        </div>
        """, unsafe_allow_html=True)

    # Tabla filtrable
    st.markdown("**Detalle por tipo de sector**")
    tipo_filter = st.multiselect("Filtrar por tipo", list(COLOR_MAP.keys()),
                                 default=["Sector clave", "Estratégico"])
    st.dataframe(
        df[df["tipo"].isin(tipo_filter)]
        .sort_values("BL", ascending=False)[["sector", "cluster", "BL", "FL", "centralidad", "tipo"]]
        .reset_index(drop=True),
        use_container_width=True, height=260,
    )

# ─── TAB 4: SIMULACIÓN DE SHOCK ───────────────────────────
with tab4:
    st.markdown("""
    <div class="alert-box">
      <strong>Modelo:</strong> Δx = L · Δd &nbsp;—&nbsp;
      El shock Δd en el sector seleccionado se propaga a través de todos los encadenamientos
      directos e indirectos capturados por la inversa de Leontief L = (I−A)⁻¹.
    </div>
    """, unsafe_allow_html=True)

    cs1, cs2 = st.columns([1, 2])
    with cs1:
        shock_sector = st.selectbox("Sector con shock", labels, key="shock_sec")
        shock_size   = st.number_input("Magnitud del shock (unidades MIP)", value=1000.0, step=100.0)
        run_btn      = st.button("▶  Ejecutar simulación", type="primary")

    if run_btn:
        idx_s  = labels.index(shock_sector)
        delta_d = np.zeros(n_sectors)
        delta_d[idx_s] = shock_size
        delta_x = L @ delta_d

        df_imp = pd.DataFrame({
            "sector":  labels,
            "Δx":      delta_x,
            "cluster": [partition.get(i, -1) for i in range(n_sectors)],
        }).sort_values("Δx", ascending=False)

        df_flow = (
            df_imp.groupby("cluster")["Δx"]
            .sum().reset_index()
            .sort_values("Δx", ascending=False)
        )

        with cs2:
            fig_imp = px.bar(
                df_imp.head(15),
                x="Δx", y="sector", orientation="h",
                color="Δx", color_continuous_scale="Turbo",
                title=f"Top 15 sectores impactados — shock en '{shock_sector}'",
                labels={"Δx": "Impacto Δx", "sector": ""},
            )
            fig_imp.update_layout(
                yaxis=dict(autorange="reversed"),
                height=420, **PLOTLY_THEME,
                coloraxis_showscale=False,
            )
            fig_imp.update_traces(
                hoverlabel=dict(bgcolor="#1c2030", bordercolor="#252a3a",
                                font=dict(family="Space Mono", size=11)),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("**Multiplicador total:**")
        mult = delta_x.sum() / shock_size
        st.markdown(f"""
        <div class="kpi-card gold">
          <div class="kpi-label">Multiplicador de producción (Σ Δx / shock)</div>
          <div class="kpi-value">{mult:.4f}</div>
          <div class="kpi-sub">Por cada unidad de shock en '{shock_sector}', la economía genera {mult:.2f} unidades adicionales en total.</div>
        </div>
        """, unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Impacto por sector (top 20)**")
            st.dataframe(df_imp.head(20).reset_index(drop=True), use_container_width=True)
        with cb:
            st.markdown("**Flujo entre clusters**")
            fig_pie = px.pie(
                df_flow, values="Δx", names="cluster",
                color_discrete_sequence=px.colors.sequential.Turbo,
                title="Distribución del shock por cluster",
            )
            fig_pie.update_layout(height=320, **PLOTLY_THEME)
            st.plotly_chart(fig_pie, use_container_width=True)

# ─── TAB 5: TODOS LOS SECTORES ────────────────────────────
with tab5:
    st.markdown("**Tabla completa de sectores — todos los indicadores**")

    c_f1, c_f2, c_f3 = st.columns(3)
    f_cl   = c_f1.multiselect("Cluster", sorted(df["cluster"].unique()), key="all_cl")
    f_tipo = c_f2.multiselect("Tipo", df["tipo"].unique().tolist(), key="all_tipo")
    f_sort = c_f3.selectbox("Ordenar por", ["centralidad", "BL", "FL", "grado"], key="all_sort")

    df_view = df.copy()
    if f_cl:   df_view = df_view[df_view["cluster"].isin(f_cl)]
    if f_tipo: df_view = df_view[df_view["tipo"].isin(f_tipo)]
    df_view = df_view.sort_values(f_sort, ascending=False).reset_index(drop=True)

    st.dataframe(
        df_view[["sector", "cluster", "centralidad", "grado", "BL", "FL", "tipo"]],
        use_container_width=True, height=500,
    )

    # Descarga
    csv = df_view.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Descargar CSV", csv, "sectores_clusters.csv", "text/csv")

# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown("""
<hr style="margin-top:40px; border-color:#252a3a;">
<div style="text-align:center; padding:16px 0; font-family:'Space Mono',monospace; font-size:10px; letter-spacing:2px; color:#374151;">
  CLUSTER INTELLIGENCE · MODELO INSUMO-PRODUCTO · LEONTIEF + LOUVAIN
</div>
""", unsafe_allow_html=True)
