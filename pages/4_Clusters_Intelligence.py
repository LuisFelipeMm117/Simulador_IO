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
st.title("🏭 Cluster Intelligence")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")

uploaded_A = st.sidebar.file_uploader("Matriz A (.npy)", type=["npy"])

top_k = st.sidebar.slider("Top-K conexiones", 3, 20, 10)
threshold = st.sidebar.number_input("Umbral mínimo", 0.0001, 0.05, 0.001, step=0.0001)
resolution = st.sidebar.slider("Resolución Louvain", 0.5, 2.0, 1.0, 0.1)

use_default = st.sidebar.checkbox("Usar datos precargados", value=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_matrix(uploaded_file, default_path):
    try:
        if not use_default and uploaded_file is not None:
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

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
@st.cache_data
def build_model(A, top_k, threshold, resolution):
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

    G = nx.from_numpy_array(W_f)
    G.remove_nodes_from(list(nx.isolates(G)))

    partition = community_louvain.best_partition(G, resolution=resolution, random_state=42)

    try:
        centrality = nx.eigenvector_centrality_numpy(G)
    except:
        centrality = nx.degree_centrality(G)

    modularity = community_louvain.modularity(partition, G)

    return G, partition, centrality, L, modularity

# RUN
try:
    G, partition, centrality, L, modularity = build_model(
        A, top_k, threshold, resolution
    )
except Exception as e:
    st.error(f"Error en modelo: {e}")
    st.stop()

# ─────────────────────────────────────────────
# DATAFRAME
# ─────────────────────────────────────────────
df = pd.DataFrame({
    "node": list(G.nodes()),
    "cluster": [partition.get(i, -1) for i in G.nodes()],
    "centrality": [centrality.get(i, 0) for i in G.nodes()],
})

summary = (
    df.groupby("cluster")
    .agg(
        tamaño=("node", "count"),
        centralidad_media=("centrality", "mean")
    )
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
c2.metric("Clusters detectados", df["cluster"].nunique())
c3.metric("Modularidad", f"{modularity:.3f}")

st.markdown("💡 **Insight:** La economía se organiza en clusters con distinta influencia sistémica.")

# ─────────────────────────────────────────────
# TOP CLUSTER
# ─────────────────────────────────────────────
top_cluster = summary.iloc[0]

st.subheader("🏆 Cluster más estratégico")
st.markdown(f"""
- **Cluster:** {top_cluster['cluster']}
- **Tamaño:** {top_cluster['tamaño']} sectores  
- **Score:** {top_cluster['score']:.4f}  

👉 Este cluster concentra la mayor influencia dentro del sistema.
""")

# ─────────────────────────────────────────────
# BUBBLE GRAPH
# ─────────────────────────────────────────────
st.subheader("🌐 Mapa de influencia (clusters)")

fig_bubble = px.scatter(
    summary,
    x="tamaño",
    y="centralidad_media",
    size="score",
    color="cluster",
    hover_name="cluster",
    size_max=60
)

fig_bubble.update_layout(height=500)
st.plotly_chart(fig_bubble, use_container_width=True)

# ─────────────────────────────────────────────
# 🔥 NETWORK GRAPH
# ─────────────────────────────────────────────
st.subheader("🕸️ Red de influencia económica")

top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:40]
sub_G = G.subgraph(top_nodes)

pos = nx.spring_layout(sub_G, seed=42, k=1.5)

# EDGES
edge_x, edge_y = [], []
for u, v in sub_G.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    mode="lines",
    line=dict(width=0.7, color="#888"),
    hoverinfo="none"
)

# NODES
node_x, node_y, node_size, node_color, node_text = [], [], [], [], []

for node in sub_G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_size.append(20 + centrality[node] * 80)
    node_color.append(partition[node])
    node_text.append(f"Sector {node}<br>Cluster {partition[node]}")

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    text=[str(n) for n in sub_G.nodes()],
    textposition="top center",
    hovertext=node_text,
    hoverinfo="text",
    marker=dict(
        showscale=True,
        colorscale="Turbo",
        color=node_color,
        size=node_size,
        line_width=1
    )
)

fig_net = go.Figure(data=[edge_trace, node_trace])

fig_net.update_layout(
    title="Red económica (sectores más influyentes)",
    showlegend=False,
    hovermode="closest",
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=650
)

st.plotly_chart(fig_net, use_container_width=True)

# ─────────────────────────────────────────────
# RANKING
# ─────────────────────────────────────────────
st.subheader("📊 Ranking de Clusters")

st.dataframe(
    summary[["rank", "cluster", "tamaño", "centralidad_media", "score"]],
    use_container_width=True
)

# ─────────────────────────────────────────────
# TOP SECTORS
# ─────────────────────────────────────────────
st.subheader("🔥 Sectores más importantes")

top_nodes_df = df.sort_values("centrality", ascending=False).head(10)
st.dataframe(top_nodes_df, use_container_width=True)
