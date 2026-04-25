# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import plotly.express as px
import streamlit as st
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Clusters Intelligence", layout="wide")
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
# LOAD DATA (CORRECTO)
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

n = A.shape[0]

# ─────────────────────────────────────────────
# CORE MODEL (ARREGLADO)
# ─────────────────────────────────────────────
@st.cache_data
def build_model(A, top_k, threshold, resolution):
    
    n = A.shape[0]
    I = np.eye(n)
    M = I - A

    # Validación numérica
    if np.linalg.cond(M) > 1e10:
        raise ValueError("Matriz mal condicionada")

    L = np.linalg.inv(M)

    # Normalización
    col_sum = L.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1
    W = L / col_sum

    # Simetrización
    W_sym = (W + W.T) / 2

    # Filtro
    W_f = np.zeros_like(W_sym)
    for i in range(n):
        idx = np.argsort(W_sym[i])[-top_k:]
        W_f[i, idx] = W_sym[i, idx]

    W_f[W_f < threshold] = 0

    # Grafo
    G = nx.from_numpy_array(W_f)
    G.remove_nodes_from(list(nx.isolates(G)))

    # Louvain
    partition = community_louvain.best_partition(G, resolution=resolution, random_state=42)

    # Centralidad
    try:
        centrality = nx.eigenvector_centrality_numpy(G)
    except:
        centrality = nx.degree_centrality(G)

    modularity = community_louvain.modularity(partition, G)

    return G, partition, centrality, L, modularity

# ─────────────────────────────────────────────
# RUN MODEL
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Nodos", G.number_of_nodes())
c2.metric("Clusters", df["cluster"].nunique())
c3.metric("Modularidad", f"{modularity:.4f}")

st.divider()

# ─────────────────────────────────────────────
# CLUSTER SUMMARY
# ─────────────────────────────────────────────
summary = (
    df.groupby("cluster")
    .agg(
        tamaño=("node", "count"),
        centralidad_media=("centrality", "mean")
    )
    .reset_index()
    .sort_values("centralidad_media", ascending=False)
)

st.subheader("📊 Ranking de Clusters")
st.dataframe(summary, use_container_width=True)

fig = px.bar(
    summary,
    x="cluster",
    y="tamaño",
    color="centralidad_media",
    title="Clusters por tamaño e importancia"
)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# TOP NODES
# ─────────────────────────────────────────────
st.subheader("🔥 Sectores más importantes")

top_nodes = df.sort_values("centrality", ascending=False).head(15)
st.dataframe(top_nodes, use_container_width=True)
