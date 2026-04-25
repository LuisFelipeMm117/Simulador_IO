# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Cluster Intelligence", layout="wide")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")

theme_mode = st.sidebar.selectbox("🎨 Tema", ["Claro", "Oscuro"], index=0)

uploaded_A = st.sidebar.file_uploader("Matriz A (.npy)", type=["npy"])

top_k = st.sidebar.slider("Top-K conexiones", 3, 20, 10)
threshold = st.sidebar.number_input("Umbral mínimo", 0.0001, 0.05, 0.001, step=0.0001)
resolution = st.sidebar.slider("Resolución Louvain", 0.5, 2.0, 1.0, 0.1)

use_default = st.sidebar.checkbox("Usar datos precargados", value=True)

# ─────────────────────────────────────────────
# THEME (DINÁMICO)
# ─────────────────────────────────────────────
if theme_mode == "Oscuro":
    bg, surface, surface2 = "#0d0f14", "#141720", "#1c2030"
    text, border, accent = "#e2e8f0", "#252a3a", "#00e5ff"
else:
    bg, surface, surface2 = "#f8fafc", "#ffffff", "#f1f5f9"
    text, border, accent = "#0f172a", "#e2e8f0", "#2563eb"

st.markdown(f"""
<style>
html, body, [data-testid="stApp"] {{
    background-color: {bg};
    color: {text};
}}
[data-testid="stSidebar"] {{
    background: {surface};
}}
[data-testid="stMetric"] {{
    background: {surface2};
    border: 1px solid {border};
    border-radius: 12px;
    padding: 12px;
}}
[data-testid="stMetricValue"] {{
    color: {accent};
}}
</style>
""", unsafe_allow_html=True)

st.title("🏭 Cluster Intelligence")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_matrix(uploaded_file, default_path):
    try:
        if not use_default and uploaded_file:
            return np.load(uploaded_file)
        elif Path(default_path).exists():
            return np.load(default_path)
        else:
            return None
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None

BASE_DIR = Path(__file__).resolve().parent.parent
A = load_matrix(uploaded_A, BASE_DIR / "data/A_nacional.npy")

if A is None:
    st.error("No se pudo cargar la matriz A")
    st.stop()

if A.shape[0] != A.shape[1]:
    st.error("La matriz A debe ser cuadrada")
    st.stop()

if not np.isfinite(A).all():
    st.error("La matriz contiene valores inválidos")
    st.stop()

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
@st.cache_data
def build_model(A, top_k, threshold, resolution):

    A = A.copy()
    n = A.shape[0]

    I = np.eye(n)
    M = I - A

    if np.linalg.cond(M) > 1e10:
        raise ValueError("Matriz mal condicionada")

    L = np.linalg.inv(M)

    col_sum = L.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1
    W = L / col_sum
    W_sym = (W + W.T) / 2

    W_f = np.zeros_like(W_sym)
    for i in range(n):
        idx = np.argsort(W_sym[i])[-top_k:]
        W_f[i, idx] = W_sym[i, idx]

    W_f[W_f < threshold] = 0

    G_full = nx.from_numpy_array(W_f)
    G_full.remove_nodes_from(list(nx.isolates(G_full)))

    largest_cc = max(nx.connected_components(G_full), key=len)
    G = G_full.subgraph(largest_cc).copy()

    partition = community_louvain.best_partition(G, resolution=resolution, random_state=42)
    modularity = community_louvain.modularity(partition, G)

    try:
        centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        centrality = nx.degree_centrality(G)

    return G, partition, centrality, L, modularity

G, partition, centrality, L, modularity = build_model(
    A, top_k, threshold, resolution
)

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
df = pd.DataFrame({
    "node": list(G.nodes()),
    "cluster": [partition.get(i, -1) for i in G.nodes()],
    "centrality": [centrality.get(i, 0) for i in G.nodes()],
})

summary = (
    df.groupby("cluster")
    .agg(tamaño=("node", "count"), centralidad_media=("centrality", "mean"))
    .reset_index()
)

summary["score"] = (
    summary["centralidad_media"] * 0.6 +
    (summary["tamaño"] / summary["tamaño"].max()) * 0.4
)

summary = summary.sort_values("score", ascending=False)
summary["rank"] = range(1, len(summary) + 1)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Sectores activos", G.number_of_nodes())
c2.metric("Clusters", df["cluster"].nunique())
c3.metric("Modularidad", f"{modularity:.3f}")

st.markdown("💡 **Insight:** La economía se organiza en clusters con distinta influencia.")

# ─────────────────────────────────────────────
# TOP CLUSTER
# ─────────────────────────────────────────────
top_cluster = summary.iloc[0]

st.subheader("🏆 Cluster más estratégico")
st.write(top_cluster)

# ─────────────────────────────────────────────
# BUBBLE
# ─────────────────────────────────────────────
st.subheader("🌐 Mapa de clusters")

fig = px.scatter(
    summary,
    x="tamaño",
    y="centralidad_media",
    size="score",
    color="cluster",
    size_max=60
)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# NETWORK (ARREGLADO)
# ─────────────────────────────────────────────
st.subheader("🕸️ Red de influencia")

top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:20]
sub_G = G.subgraph(top_nodes)

pos = nx.spring_layout(sub_G, seed=42, k=2.5, iterations=100)

edge_threshold = 0.02

edge_x, edge_y = [], []
for u, v in sub_G.edges():
    w = G[u][v].get("weight", 0)
    if w > edge_threshold:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    mode="lines",
    line=dict(width=1, color="rgba(150,150,150,0.4)"),
    hoverinfo="none"
)

node_x, node_y, node_size, node_color = [], [], [], []

for node in sub_G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_size.append(25 + centrality[node] * 100)
    node_color.append(partition[node])

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers",
    marker=dict(
        size=node_size,
        color=node_color,
        colorscale="Turbo",
        showscale=True,
        line=dict(width=1, color="white")
    ),
    text=[str(n) for n in sub_G.nodes()],
    hovertemplate="Sector %{text}"
)

fig_net = go.Figure(data=[edge_trace, node_trace])

fig_net.update_layout(
    showlegend=False,
    height=650,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False)
)

st.plotly_chart(fig_net, use_container_width=True)

# ─────────────────────────────────────────────
# RANKING
# ─────────────────────────────────────────────
st.subheader("📊 Ranking de clusters")
st.dataframe(summary, use_container_width=True)

# ─────────────────────────────────────────────
# TOP SECTORS
# ─────────────────────────────────────────────
st.subheader("🔥 Sectores clave")
st.dataframe(
    df.sort_values("centrality", ascending=False).head(10),
    use_container_width=True
)
