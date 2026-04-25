# -*- coding: utf-8 -*-
"""
Simulador de Clusters Económicos – Modelo Insumo-Producto
Basado en la regionalización nacional (MIP)
Compatible con VS Code / Streamlit
"""

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador de Clusters Económicos",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Simulador de Clusters Económicos")
st.markdown(
    "Análisis de encadenamientos productivos mediante el modelo Insumo-Producto "
    "y detección de comunidades con el algoritmo de Louvain."
)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR – PARÁMETROS Y CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")

# Carga de archivos
st.sidebar.subheader("1. Datos de entrada")
uploaded_A = st.sidebar.file_uploader("Matriz A (coeficientes técnicos)", type=["npy"])
uploaded_X = st.sidebar.file_uploader("Vector X (producción total)", type=["npy"])
uploaded_Y = st.sidebar.file_uploader("Vector Y (demanda final)", type=["npy"])
uploaded_Z = st.sidebar.file_uploader("Matriz Z (flujos intersectoriales)", type=["npy"])

# Etiquetas de sectores
st.sidebar.subheader("2. Etiquetas de sectores")
label_mode = st.sidebar.radio(
    "Modo de etiquetas",
    ["Automático (S0, S1, …)", "Cargar CSV con nombres"],
)
sector_labels = None
if label_mode == "Cargar CSV con nombres":
    csv_file = st.sidebar.file_uploader("CSV con columna 'sector'", type=["csv"])
    if csv_file:
        df_labels = pd.read_csv(csv_file)
        sector_labels = df_labels["sector"].tolist()

# Parámetros del modelo
st.sidebar.subheader("3. Parámetros del modelo")
top_k = st.sidebar.slider("Top-K conexiones por nodo", min_value=3, max_value=20, value=10)
threshold = st.sidebar.number_input(
    "Umbral mínimo de peso", min_value=0.0001, max_value=0.05,
    value=0.001, step=0.0001, format="%.4f"
)
resolution = st.sidebar.slider(
    "Resolución Louvain (mayor = más clusters)", 0.5, 2.0, 1.0, 0.1
)

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES CORE
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Construyendo modelo…")
def build_model(A_bytes, top_k, threshold, resolution):
    """
    Construye el modelo completo:
      1. Matriz de Leontief  L = (I - A)^{-1}
      2. Matriz de interacción normalizada W (simétrica)
      3. Filtrado top-K + umbral
      4. Grafo NetworkX
      5. Partición Louvain
      6. Centralidad de eigenvector
    """
    A = np.frombuffer(A_bytes, dtype=np.float64)
    n = int(np.sqrt(len(A)))
    A = A.reshape(n, n)

    I = np.eye(n)
    L = np.linalg.inv(I - A)  # Inversa de Leontief

    # Normalización columna → proporciones de interacción
    col_sum = L.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1
    W = L / col_sum

    # Simetrización
    W_sym = (W + W.T) / 2

    # Filtro top-K por fila
    W_f = np.zeros_like(W_sym)
    for i in range(n):
        idx = np.argsort(W_sym[i])[-top_k:]
        W_f[i, idx] = W_sym[i, idx]

    # Umbral de peso mínimo
    W_f[W_f < threshold] = 0

    # Construcción del grafo
    G = nx.from_numpy_array(W_f)
    G.remove_nodes_from(list(nx.isolates(G)))

    # Partición Louvain
    partition = community_louvain.best_partition(G, resolution=resolution, random_state=42)

    # Centralidad de eigenvector
    try:
        centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        centrality = nx.degree_centrality(G)

    # Métricas de red
    modularity = community_louvain.modularity(partition, G)
    density = nx.density(G)

    return G, partition, centrality, L, W_f, modularity, density, n


def make_labels(n, custom=None):
    if custom and len(custom) == n:
        return custom
    return [f"S{i}" for i in range(n)]


def build_df(G, partition, centrality, labels_map):
    """Construye DataFrame maestro de sectores."""
    rows = []
    for node in G.nodes():
        rows.append({
            "sector": labels_map.get(node, str(node)),
            "node_id": node,
            "cluster": partition.get(node, -1),
            "centrality": centrality.get(node, 0.0),
            "degree": G.degree(node),
        })
    return pd.DataFrame(rows)


def simulate_shock(L, n, labels, shock_sector_idx, shock_size=1.0):
    """
    Simula un shock de demanda final en un sector y propaga
    mediante la inversa de Leontief.
    Retorna vector de impacto total por sector.
    """
    d = np.zeros(n)
    d[shock_sector_idx] = shock_size
    impact = L @ d  # Δx = L · Δd
    return impact


def forward_linkage(L, n):
    """Índice de encadenamiento hacia adelante (Ghosh): suma de fila / n"""
    return L.sum(axis=1) / n


def backward_linkage(L, n):
    """Índice de encadenamiento hacia atrás (Leontief): suma de columna / n"""
    return L.sum(axis=0) / n


def cluster_summary(df, L, n):
    """Estadísticas agregadas por cluster."""
    bl = backward_linkage(L, n)
    fl = forward_linkage(L, n)
    df2 = df.copy()
    df2["backward_linkage"] = [bl[r] for r in df2["node_id"]]
    df2["forward_linkage"]  = [fl[r] for r in df2["node_id"]]
    return (
        df2.groupby("cluster")
        .agg(
            num_sectores=("sector", "count"),
            centrality_media=("centrality", "mean"),
            encad_atras=("backward_linkage", "mean"),
            encad_adelante=("forward_linkage", "mean"),
        )
        .reset_index()
    )


# ──────────────────────────────────────────────────────────────────────────────
# LÓGICA PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

# Determinar fuente de A
if uploaded_A:
    A_bytes = uploaded_A.read()
    n_inferred = int(np.sqrt(len(np.frombuffer(A_bytes, dtype=np.float64))))
else:
    # Intentar ruta local (modo VS Code)
    local_path = Path("A_nacional.npy")
    if local_path.exists():
        A_arr = np.load(local_path)
        A_bytes = A_arr.tobytes()
        n_inferred = A_arr.shape[0]
        st.sidebar.success(f"Cargado A_nacional.npy local ({n_inferred}×{n_inferred})")
    else:
        st.info("👈 Sube la **Matriz A** en el panel lateral para comenzar.")
        st.stop()

# Construir modelo
G, partition, centrality, L, W_f, modularity, density, n = build_model(
    A_bytes, top_k, threshold, resolution
)

# Etiquetas
labels = make_labels(n, sector_labels)
labels_map = {i: labels[i] for i in range(n)}
labels_map_inv = {v: k for k, v in labels_map.items()}

# DataFrame maestro
df = build_df(G, partition, centrality, labels_map)
df_summary = cluster_summary(df, L, n)

num_clusters = df["cluster"].nunique()

# ──────────────────────────────────────────────────────────────────────────────
# MÉTRICAS GLOBALES
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Indicadores globales del grafo")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Nodos activos", G.number_of_nodes())
c2.metric("Aristas", G.number_of_edges())
c3.metric("Clusters (Louvain)", num_clusters)
c4.metric("Modularidad", f"{modularity:.4f}")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Explorador de Clusters",
    "📡 Simulación de Shock",
    "🔗 Encadenamientos",
    "🗺️ Red de Sectores",
])

# ─── TAB 1: EXPLORADOR DE CLUSTERS ───────────────────────────────────────────
with tab1:
    st.subheader("Explorador de Clusters")

    col_sel, col_info = st.columns([1, 2])

    with col_sel:
        cluster_ids = sorted(df["cluster"].unique())
        selected_cluster = st.selectbox("Selecciona un cluster", cluster_ids)

        df_c = df[df["cluster"] == selected_cluster].sort_values("centrality", ascending=False)

        st.markdown(f"**Sectores en Cluster {selected_cluster}:** {len(df_c)}")
        st.dataframe(
            df_c[["sector", "centrality", "degree"]].reset_index(drop=True),
            use_container_width=True,
        )

    with col_info:
        # Resumen de clusters
        st.markdown("**Resumen de todos los clusters**")
        st.dataframe(df_summary.set_index("cluster"), use_container_width=True)

        # Distribución de tamaños
        fig_size = px.bar(
            df_summary,
            x="cluster", y="num_sectores",
            color="centrality_media",
            color_continuous_scale="Blues",
            title="Tamaño y centralidad media por cluster",
            labels={"num_sectores": "# Sectores", "cluster": "Cluster"},
        )
        fig_size.update_layout(height=300)
        st.plotly_chart(fig_size, use_container_width=True)

# ─── TAB 2: SIMULACIÓN DE SHOCK ──────────────────────────────────────────────
with tab2:
    st.subheader("Simulación de Shock de Demanda Final")
    st.markdown(
        "Selecciona un sector y una magnitud de shock. "
        "El modelo propaga el impacto mediante la inversa de Leontief: **Δx = L · Δd**"
    )

    col_shock1, col_shock2 = st.columns([1, 2])

    with col_shock1:
        shock_sector = st.selectbox("Sector con el shock", labels)
        shock_size   = st.number_input("Magnitud del shock (millones)", value=1000.0, step=100.0)
        run_shock    = st.button("▶ Ejecutar simulación", type="primary")

    if run_shock:
        idx = labels_map_inv.get(shock_sector, int(shock_sector.replace("S", "")))
        impact = simulate_shock(L, n, labels, idx, shock_size)

        df_impact = pd.DataFrame({
            "sector":  labels,
            "impacto": impact,
            "cluster": [partition.get(i, -1) for i in range(n)],
        }).sort_values("impacto", ascending=False)

        # Flujo por cluster
        df_flow = (
            df_impact.groupby("cluster")["impacto"]
            .sum()
            .reset_index()
            .sort_values("impacto", ascending=False)
        )

        with col_shock2:
            st.markdown(f"**Top 15 sectores impactados por shock en '{shock_sector}'**")
            fig_imp = px.bar(
                df_impact.head(15),
                x="impacto", y="sector",
                orientation="h",
                color="cluster",
                color_continuous_scale="RdYlGn",
                title="Impacto directo e indirecto por sector",
            )
            fig_imp.update_layout(height=450, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("**Impacto por sector (top 20)**")
            st.dataframe(df_impact.head(20).reset_index(drop=True), use_container_width=True)
        with col_t2:
            st.markdown("**Flujo de impacto entre clusters**")
            st.dataframe(df_flow.reset_index(drop=True), use_container_width=True)

            fig_flow = px.pie(
                df_flow, values="impacto", names="cluster",
                title="Distribución del shock entre clusters",
            )
            st.plotly_chart(fig_flow, use_container_width=True)

# ─── TAB 3: ENCADENAMIENTOS ───────────────────────────────────────────────────
with tab3:
    st.subheader("Índices de Encadenamiento Productivo")
    st.markdown(
        "Los índices de Leontief (hacia atrás) y Rasmussen-Hirschman (hacia adelante) "
        "identifican sectores *clave* de la economía."
    )

    bl = backward_linkage(L, n)
    fl = forward_linkage(L, n)
    mean_bl = bl.mean()
    mean_fl = fl.mean()

    df_link = pd.DataFrame({
        "sector":            labels,
        "encad_atras":       bl,
        "encad_adelante":    fl,
        "cluster":           [partition.get(i, -1) for i in range(n)],
    })

    # Clasificación cuadrante
    def classify(row):
        high_bl = row["encad_atras"]  >= mean_bl
        high_fl = row["encad_adelante"] >= mean_fl
        if high_bl and high_fl:
            return "Sector clave"
        elif high_bl:
            return "Impulsor"
        elif high_fl:
            return "Estratégico"
        else:
            return "Dependiente"

    df_link["tipo"] = df_link.apply(classify, axis=1)

    color_map = {
        "Sector clave":  "#2ca02c",
        "Impulsor":      "#1f77b4",
        "Estratégico":   "#ff7f0e",
        "Dependiente":   "#d62728",
    }

    fig_quad = px.scatter(
        df_link,
        x="encad_atras",
        y="encad_adelante",
        color="tipo",
        color_discrete_map=color_map,
        hover_name="sector",
        hover_data={"cluster": True},
        title="Clasificación de sectores por encadenamientos",
        labels={
            "encad_atras":    "Encadenamiento hacia atrás (Leontief)",
            "encad_adelante": "Encadenamiento hacia adelante (Ghosh)",
        },
    )
    # Líneas de referencia
    fig_quad.add_hline(y=mean_fl, line_dash="dash", line_color="gray", opacity=0.6)
    fig_quad.add_vline(x=mean_bl, line_dash="dash", line_color="gray", opacity=0.6)
    fig_quad.update_layout(height=500)
    st.plotly_chart(fig_quad, use_container_width=True)

    col_link1, col_link2 = st.columns(2)
    with col_link1:
        tipo_filter = st.multiselect(
            "Filtrar por tipo",
            df_link["tipo"].unique().tolist(),
            default=["Sector clave"],
        )
        st.dataframe(
            df_link[df_link["tipo"].isin(tipo_filter)]
            .sort_values("encad_atras", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
        )
    with col_link2:
        conteo = df_link["tipo"].value_counts().reset_index()
        conteo.columns = ["tipo", "count"]
        fig_pie_link = px.pie(
            conteo, values="count", names="tipo",
            color="tipo", color_discrete_map=color_map,
            title="Composición por tipo de sector",
        )
        st.plotly_chart(fig_pie_link, use_container_width=True)

# ─── TAB 4: RED DE SECTORES ───────────────────────────────────────────────────
with tab4:
    st.subheader("Visualización de la Red Productiva")
    st.markdown(
        "Cada nodo es un sector; el color indica su cluster (Louvain). "
        "El tamaño refleja la centralidad de eigenvector."
    )

    max_nodes = st.slider("Máx. nodos a mostrar (por centralidad)", 10, min(78, G.number_of_nodes()), 40)

    # Subgrafo con nodos más centrales
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:max_nodes]
    sub_G = G.subgraph(top_nodes)

    pos = nx.spring_layout(sub_G, seed=42, k=1.5)

    # Aristas
    edge_x, edge_y = [], []
    for u, v in sub_G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="#aaaaaa"),
        hoverinfo="none",
    )

    # Nodos
    node_x    = [pos[n][0] for n in sub_G.nodes()]
    node_y    = [pos[n][1] for n in sub_G.nodes()]
    node_text = [labels_map.get(n, str(n)) for n in sub_G.nodes()]
    node_clus = [partition.get(n, 0) for n in sub_G.nodes()]
    node_size = [20 + 60 * centrality.get(n, 0) for n in sub_G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(
            size=node_size,
            color=node_clus,
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(title="Cluster"),
            line=dict(width=1, color="white"),
        ),
        text=node_text,
        textposition="top center",
        hovertemplate="<b>%{text}</b><br>Cluster: %{marker.color}<extra></extra>",
    )

    fig_net = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Red productiva – top {max_nodes} sectores por centralidad",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
        ),
    )
    st.plotly_chart(fig_net, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# DESCARGA DE RESULTADOS
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("⬇️ Exportar resultados")

@st.cache_data
def to_csv(frame):
    return frame.to_csv(index=False).encode("utf-8")

col_d1, col_d2 = st.columns(2)
with col_d1:
    st.download_button(
        "Descargar tabla de sectores (CSV)",
        data=to_csv(df),
        file_name="sectores_clusters.csv",
        mime="text/csv",
    )
with col_d2:
    st.download_button(
        "Descargar resumen de clusters (CSV)",
        data=to_csv(df_summary),
        file_name="resumen_clusters.csv",
        mime="text/csv",
    )
