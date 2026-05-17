# -*- coding: utf-8 -*-
"""
Cluster Intelligence v2.0 — Simulador de Clusters Económicos
Modelo Insumo-Producto · Detección de Comunidades Louvain
Módulos:
  1. Análisis Nacional   — Matriz A (Leontief) + Louvain
  2. Análisis Nacional con Correlaciones WIOD
  3. Análisis Regional   — Matrices Zr / Lr por entidad federativa + Contagio

CAMBIOS v2.1:
  - Módulo 3 ahora carga automáticamente archivos .npy desde disco (carpeta de
    datos del estado), eliminando la necesidad de subir CSV manualmente.
  - Se mantiene compatibilidad con carga manual como fallback opcional.
  - Nueva función `_load_regional_npy()` centraliza la detección y lectura.
  - Los nombres de sector se leen de sectores.csv (si existe) o se generan
    automáticamente como "Sector 001 … Sector N".
  - El selector de estado en la barra lateral enumera automáticamente todas
    las carpetas que contengan el conjunto mínimo de archivos requeridos.
"""

import numpy as np
import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ══════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Cluster Intelligence v2",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# CSS GLOBAL
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
    --ok:        #10b981;
    --warn:      #ef4444;
}

header {visibility: hidden;}
[data-testid="stToolbar"]    {display: none;}
[data-testid="stDecoration"] {display: none;}
[data-testid="stStatusWidget"]{display: none;}
footer {visibility: hidden;}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { font-family: var(--sans) !important; }

h1, h2, h3, h4 { font-family: var(--mono) !important; color: var(--text) !important; }

[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important; font-size: 11px !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    font-family: var(--mono) !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important; font-family: var(--mono) !important; font-size: 2rem !important;
}

[data-testid="stTabs"] button {
    font-family: var(--mono) !important; font-size: 12px !important;
    letter-spacing: 1px !important; color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important;
}

.kpi-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px 24px; margin-bottom: 12px;
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 4px; height: 100%; background: var(--accent);
    border-radius: 4px 0 0 4px;
}
.kpi-card.warn::before   { background: var(--warn); }
.kpi-card.ok::before     { background: var(--ok); }
.kpi-card.gold::before   { background: var(--accent3); }
.kpi-card.purple::before { background: var(--accent2); }

.kpi-label {
    font-family: 'Space Mono', monospace; font-size: 10px;
    letter-spacing: 2px; text-transform: uppercase; color: #64748b; margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Space Mono', monospace; font-size: 1.8rem;
    font-weight: 700; color: var(--text); line-height: 1.1;
}
.kpi-sub { font-size: 12px; color: #64748b; margin-top: 4px; }

.badge { display: inline-block; padding: 2px 10px; border-radius: 20px;
         font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 700; }
.badge-clave       { background: #10b98122; color: #10b981; border: 1px solid #10b981; }
.badge-impulsor    { background: #3b82f622; color: #60a5fa; border: 1px solid #3b82f6; }
.badge-estrategico { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b; }
.badge-dependiente { background: #ef444422; color: #f87171; border: 1px solid #ef4444; }

.section-title {
    font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 3px;
    text-transform: uppercase; color: #64748b; padding: 16px 0 8px 0;
    border-top: 1px solid var(--border); margin-top: 8px;
}

.alert-box {
    background: var(--surface2); border-left: 3px solid var(--accent);
    padding: 12px 16px; border-radius: 0 8px 8px 0; font-size: 13px; margin: 8px 0;
}
.alert-box.warn { border-left-color: var(--warn); }
.alert-box.ok   { border-left-color: var(--ok); }

.module-header {
    font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 3px;
    color: var(--muted); text-transform: uppercase; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# CONSTANTES / PALETA
# ══════════════════════════════════════════════════════════
PLOTLY_THEME = dict(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Space Mono, monospace", color="#334155", size=11),
    margin=dict(l=20, r=20, t=40, b=20),
)
COLOR_MAP_TIPO = {
    "Sector clave": "#10b981", "Impulsor": "#60a5fa",
    "Estratégico":  "#f59e0b", "Dependiente": "#f87171",
}

BASE_DIR = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════
# HEADER GLOBAL
# ══════════════════════════════════════════════════════════
st.markdown("""
<div style="padding: 32px 0 16px 0;">
  <div class="module-header">⬡ ANÁLISIS ESTRUCTURAL · MODELO INSUMO-PRODUCTO v2.0</div>
  <h1 style="font-family:'Space Mono',monospace; font-size:2.4rem; font-weight:700;
              color:var(--text); margin:0; line-height:1.1;">Cluster Intelligence</h1>
  <div style="color:var(--muted); font-size:14px; margin-top:8px;">
    Nacional (Leontief + WIOD) · Regional (Entidades Federativas) · Contagio Financiero
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MÓDULOS DE NAVEGACIÓN (nivel superior)
# ══════════════════════════════════════════════════════════
MODULE_LABELS = [
    "🏛️  Nacional — Leontief",
    "🔗  Nacional — Correlaciones WIOD",
    "🗺️  Regional — Entidades",
]
module = st.radio(
    "Módulo activo", MODULE_LABELS,
    horizontal=True, label_visibility="collapsed",
)

st.markdown("---")

# ╔═══════════════════════════════════════════════════════╗
# ║  MÓDULO 1 — NACIONAL LEONTIEF (código original v1)   ║
# ╚═══════════════════════════════════════════════════════╝
if module == MODULE_LABELS[0]:

    # ── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="section-title">📂 Datos Nacionales</div>', unsafe_allow_html=True)
        uploaded_A = st.file_uploader("Matriz A (.npy)", type=["npy"], key="m1_A")
        st.markdown('<div class="section-title">⚙️ Parámetros</div>', unsafe_allow_html=True)
        top_k      = st.slider("Top-K conexiones por nodo", 3, 20, 10, key="m1_k")
        threshold  = st.number_input("Umbral mínimo de peso (τ)", 0.0001, 0.05, 0.001,
                                     step=0.0001, format="%.4f", key="m1_tau")
        resolution = st.slider("Resolución Louvain (γ)", 0.5, 2.0, 1.0, 0.1, key="m1_res")
        use_local  = st.checkbox("Datos locales (A_nacional.npy)", value=True, key="m1_local")

    # ── Carga de datos ────────────────────────────────────
    def load_A():
        if not use_local and uploaded_A is not None:
            return np.load(uploaded_A)
        for p in [BASE_DIR / "A_nacional.npy",
                  BASE_DIR.parent / "data" / "A_nacional.npy"]:
            if p.exists():
                return np.load(p)
        return None

    A = load_A()

    if A is None:
        st.markdown('<div class="alert-box warn">⚠️ No se encontró <code>A_nacional.npy</code>. '
                    'Sube la matriz desde el panel lateral.</div>', unsafe_allow_html=True)
        st.stop()

    n_sectors = A.shape[0]

    labels_path = BASE_DIR.parent / "data" / "sectores.csv"
    if not labels_path.exists():
        st.error("No se encontró sectores.csv"); st.stop()
    df_lab = pd.read_csv(labels_path)
    if "nombre" not in df_lab.columns or len(df_lab) != n_sectors:
        st.error("Formato incorrecto en sectores.csv"); st.stop()
    labels = df_lab["nombre"].astype(str).tolist()

    # ── Modelo matemático ─────────────────────────────────
    @st.cache_data
    def build_leontief_model(A_bytes, n, top_k, threshold, resolution):
        A  = np.frombuffer(A_bytes, dtype=np.float64).reshape(n, n)
        hs_ok  = bool((A.sum(axis=0) < 1).all())
        I      = np.eye(n)
        M      = I - A
        cond_M = float(np.linalg.cond(M))
        L      = np.linalg.inv(M)

        bl_raw = L.sum(axis=0)
        fl_raw = L.sum(axis=1)
        bl     = bl_raw / bl_raw.mean()
        fl     = fl_raw / fl_raw.mean()

        col_sum = L.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1
        W     = L / col_sum
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

        partition   = community_louvain.best_partition(G, resolution=resolution, random_state=42)
        modularity  = community_louvain.modularity(partition, G)

        try:
            centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
        except nx.PowerIterationFailedConvergence:
            centrality = nx.degree_centrality(G)

        return {
            "G": G, "L": L, "partition": partition, "centrality": centrality,
            "modularity": modularity, "bl": bl, "fl": fl,
            "cond_M": cond_M, "hs_ok": hs_ok,
            "n_isolated": n - G_full.number_of_nodes(),
            "n_minor":    G_full.number_of_nodes() - G.number_of_nodes(),
        }

    try:
        m = build_leontief_model(A.tobytes(), n_sectors, top_k, threshold, resolution)
    except Exception as e:
        st.error(f"Error construyendo el modelo: {e}"); st.stop()

    G         = m["G"]; L = m["L"]; partition = m["partition"]
    centrality= m["centrality"]; modularity = m["modularity"]
    bl        = m["bl"]; fl = m["fl"]

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
        if row["BL"] >= 1 and row["FL"] < 1:  return "Impulsor"
        if row["BL"] < 1  and row["FL"] >= 1: return "Estratégico"
        return "Dependiente"

    df["tipo"] = df.apply(classify, axis=1)

    summary = (df.groupby("cluster")
               .agg(tamaño=("sector","count"), centralidad_media=("centralidad","mean"),
                    BL_media=("BL","mean"), FL_media=("FL","mean"))
               .reset_index())
    summary["score"] = (
        summary["centralidad_media"] / summary["centralidad_media"].max() * 0.5 +
        summary["BL_media"]         / summary["BL_media"].max()          * 0.3 +
        summary["FL_media"]         / summary["FL_media"].max()          * 0.2
    )
    summary = summary.sort_values("score", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary)+1))

    n_clusters = df["cluster"].nunique()
    top_cl     = summary.iloc[0]

    # ── KPIs ─────────────────────────────────────────────
    st.markdown('<div class="section-title">INDICADORES GLOBALES</div>', unsafe_allow_html=True)
    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    kc1.metric("Sectores activos",   G.number_of_nodes(), delta=f"{n_sectors} en MIP")
    kc2.metric("Clusters (Louvain)", n_clusters)
    kc3.metric("Modularidad Q",      f"{modularity:.4f}")
    kc4.metric("Sectores clave",     int((df["tipo"]=="Sector clave").sum()))
    kc5.metric("Aristas del grafo",  G.number_of_edges())

    # ── Tabs ─────────────────────────────────────────────
    st.markdown('<div class="section-title">MÓDULOS DE ANÁLISIS</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⬡  Red productiva", "📊  Clusters", "🔗  Encadenamientos", "📡  Shock", "🏷️  Sectores"
    ])

    with tab1:
        col_ctrl, _ = st.columns([1,3])
        with col_ctrl:
            max_nodes = st.slider("Nodos a mostrar", 10, G.number_of_nodes(), 45, key="m1_net")
        top_n = sorted(centrality, key=centrality.get, reverse=True)[:max_nodes]
        sub   = G.subgraph(top_n)
        pos   = nx.spring_layout(sub, seed=42, k=2.0)
        edge_traces = []
        weights = [sub[u][v].get("weight",1) for u,v in sub.edges()]
        w_max   = max(weights) if weights else 1
        for (u,v),w in zip(sub.edges(), weights):
            x0,y0=pos[u]; x1,y1=pos[v]
            alpha = 0.15 + 0.55*(w/w_max)
            edge_traces.append(go.Scatter(
                x=[x0,x1,None], y=[y0,y1,None], mode="lines",
                line=dict(width=0.8, color=f"rgba(100,116,139,{alpha:.2f})"),
                hoverinfo="none", showlegend=False))
        node_trace = go.Scatter(
            x=[pos[nd][0] for nd in sub.nodes()],
            y=[pos[nd][1] for nd in sub.nodes()],
            mode="markers",
            marker=dict(size=[14+centrality[nd]*120 for nd in sub.nodes()],
                        color=[int(partition[nd]) for nd in sub.nodes()],
                        colorscale="Turbo", showscale=True,
                        line=dict(width=1.5, color="#0d0f14"), opacity=0.92),
            text=[labels[nd] for nd in sub.nodes()],
            hovertext=[f"<b>{labels[nd]}</b><br>Cluster: {partition[nd]}<br>"
                       f"Centralidad: {centrality[nd]:.5f}" for nd in sub.nodes()],
            hoverinfo="text")
        fig_net = go.Figure(data=edge_traces+[node_trace])
        fig_net.update_layout(
            title=dict(text=f"Red productiva — top {max_nodes} sectores",
                       font=dict(family="Space Mono",size=13,color="#94a3b8")),
            showlegend=False, hovermode="closest",
            xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            height=620, **PLOTLY_THEME)
        st.plotly_chart(fig_net, use_container_width=True)

    with tab2:
        c_left, c_right = st.columns([1,2])
        with c_left:
            st.dataframe(summary.head(15)[["rank","cluster","tamaño","score"]], use_container_width=True)
        with c_right:
            fig_bubble = px.scatter(summary, x="BL_media", y="FL_media", size="tamaño",
                color="score", hover_name="cluster", size_max=55,
                color_continuous_scale="Turbo",
                title="Posicionamiento estratégico (BL vs FL)")
            fig_bubble.add_hline(y=1, line_dash="dot", line_color="#252a3a")
            fig_bubble.add_vline(x=1, line_dash="dot", line_color="#252a3a")
            fig_bubble.update_layout(height=450, **PLOTLY_THEME)
            st.plotly_chart(fig_bubble, use_container_width=True)
            sel_cl = st.selectbox("Explorar cluster", sorted(df["cluster"].unique()),
                                  format_func=lambda x: f"C{x}", key="m1_cl")
            st.dataframe(df[df["cluster"]==sel_cl].sort_values("centralidad",ascending=False)
                         [["sector","centralidad","grado","BL","FL","tipo"]].reset_index(drop=True),
                         use_container_width=True, height=220)

    with tab3:
        fig_quad = px.scatter(df, x="BL", y="FL", color="tipo",
            color_discrete_map=COLOR_MAP_TIPO, hover_name="sector", size="centralidad",
            size_max=20, title="Cuadrante Rasmussen-Hirschman (BL y FL normalizados, media = 1)")
        fig_quad.add_hline(y=1, line_dash="dash", line_color="#252a3a")
        fig_quad.add_vline(x=1, line_dash="dash", line_color="#252a3a")
        fig_quad.update_layout(height=520, **PLOTLY_THEME)
        st.plotly_chart(fig_quad, use_container_width=True)

        cl1,cl2,cl3,cl4 = st.columns(4)
        for col,tipo,cls in zip([cl1,cl2,cl3,cl4],
            ["Sector clave","Impulsor","Estratégico","Dependiente"],
            ["ok","","gold","warn"]):
            cnt = int((df["tipo"]==tipo).sum())
            col.markdown(f'<div class="kpi-card {cls}"><div class="kpi-label">{tipo}</div>'
                         f'<div class="kpi-value">{cnt}</div>'
                         f'<div class="kpi-sub">{cnt/len(df)*100:.1f}% del total</div></div>',
                         unsafe_allow_html=True)

    with tab4:
        cs1, cs2 = st.columns([1,2])
        with cs1:
            shock_sector = st.selectbox("Sector con shock", labels, key="m1_shock_sec")
            shock_size   = st.number_input("Magnitud del shock", value=1000.0, step=100.0)
            run_btn      = st.button("▶ Ejecutar simulación", type="primary")
        if run_btn:
            idx_s   = labels.index(shock_sector)
            delta_d = np.zeros(n_sectors); delta_d[idx_s] = shock_size
            delta_x = L @ delta_d
            df_imp  = pd.DataFrame({"sector": labels, "Δx": delta_x,
                                    "cluster": [partition.get(i,-1) for i in range(n_sectors)]
                                    }).sort_values("Δx", ascending=False)
            with cs2:
                fig_imp = px.bar(df_imp.head(15), x="Δx", y="sector", orientation="h",
                    color="Δx", color_continuous_scale="Turbo",
                    title=f"Top 15 impactados — shock en '{shock_sector}'")
                fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=420, **PLOTLY_THEME)
                st.plotly_chart(fig_imp, use_container_width=True)
            mult = delta_x.sum() / shock_size
            st.markdown(f'<div class="kpi-card gold"><div class="kpi-label">Multiplicador de producción</div>'
                        f'<div class="kpi-value">{mult:.4f}</div>'
                        f'<div class="kpi-sub">Por cada 1 unidad de shock → {mult:.2f} unidades totales</div></div>',
                        unsafe_allow_html=True)

    with tab5:
        c_f1,c_f2,c_f3 = st.columns(3)
        f_cl   = c_f1.multiselect("Cluster", sorted(df["cluster"].unique()), key="m1_fcl")
        f_tipo = c_f2.multiselect("Tipo", df["tipo"].unique().tolist(), key="m1_ftp")
        f_sort = c_f3.selectbox("Ordenar por", ["centralidad","BL","FL","grado"], key="m1_fs")
        df_view = df.copy()
        if f_cl:   df_view = df_view[df_view["cluster"].isin(f_cl)]
        if f_tipo: df_view = df_view[df_view["tipo"].isin(f_tipo)]
        df_view = df_view.sort_values(f_sort, ascending=False).reset_index(drop=True)
        st.dataframe(df_view[["sector","cluster","centralidad","grado","BL","FL","tipo"]],
                     use_container_width=True, height=500)
        st.download_button("⬇ Descargar CSV", df_view.to_csv(index=False).encode(),
                           "sectores_clusters.csv", "text/csv")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  MÓDULO 2 — NACIONAL CORRELACIONES WIOD                          ║
# ╚═══════════════════════════════════════════════════════════════════╝
elif module == MODULE_LABELS[1]:

    with st.sidebar:
        st.markdown('<div class="section-title">📂 Datos — MIP Nacional</div>', unsafe_allow_html=True)
        uploaded_mip = st.file_uploader("MIP Nacional (.csv)", type=["csv"], key="m2_mip")
        st.markdown('<div class="section-title">⚙️ Parámetros</div>', unsafe_allow_html=True)
        num_sectors_m2  = st.number_input("Número de sectores", 5, 200, 20, key="m2_ns")
        threshold_pct_m2 = st.slider("Percentil de umbral (%)", 50, 95, 75, key="m2_thresh",
                                     help="Se retienen únicamente las correlaciones por encima de este percentil.")
        resolution_m2   = st.slider("Resolución Louvain (γ)", 0.5, 2.0, 1.0, 0.1, key="m2_res")

    st.markdown("""
    <div class="alert-box">
      <strong>Metodología WIOD:</strong> Este módulo calcula la matriz de máxima correlación de
      <em>cero-orden</em> R entre sectores, comparando cuatro combinaciones de perfiles de compra (x)
      y venta (y) relativas. La red se construye reteniendo las correlaciones por encima del percentil
      seleccionado y detectando comunidades con Louvain.
    </div>
    """, unsafe_allow_html=True)

    if uploaded_mip is None:
        st.info("📂 Carga el archivo MIP nacional (.csv) en el panel lateral para continuar.")
        st.markdown("""
        **Formato esperado:**
        - CSV con columna índice `Concepto`
        - La región de transacciones intermedias comienza en la columna 3 (índice 2)
        - Filas especiales: `IET - Importaciones de la economía total`, `B.1bV - Valor agregado bruto`
        """)
        st.stop()

    @st.cache_data
    def build_wiod_model(file_bytes: bytes, num_sectors: int,
                         threshold_pct: float, resolution: float):
        import io
        df_national = pd.read_csv(io.BytesIO(file_bytes), sep=',', index_col='Concepto')
        sector_names_raw = df_national.index[:num_sectors]
        sector_names     = [str(s)[:30] + ("..." if len(str(s)) > 30 else "") for s in sector_names_raw]

        Z_raw = df_national.iloc[:num_sectors, 2: 2 + num_sectors]
        Z     = Z_raw.apply(pd.to_numeric, errors='coerce').fillna(0).values

        imports_row = df_national.loc['IET - Importaciones de la economía total'].iloc[2: 2 + num_sectors]
        vab_row     = df_national.loc['B.1bV - Valor agregado bruto'].iloc[2: 2 + num_sectors]
        imports_vec = pd.to_numeric(imports_row, errors='coerce').fillna(0).values
        vab_vec     = pd.to_numeric(vab_row,     errors='coerce').fillna(0).values
        X_values    = Z.sum(axis=0) + imports_vec + vab_vec
        X_dict      = dict(zip(sector_names, X_values))

        epsilon = 1e-9
        p = Z.sum(axis=0)
        s = Z.sum(axis=1)
        X_mat     = Z / (p + epsilon)
        Y_mat_col = (Z.T / (s + epsilon)).T

        n = num_sectors
        R = np.zeros((n, n))
        np.seterr(invalid='ignore')

        def safe_corr(v1, v2):
            if np.std(v1) == 0 or np.std(v2) == 0:
                return 0.0
            return float(np.corrcoef(v1, v2)[0, 1])

        for k in range(n):
            for l in range(n):
                if k == l:
                    continue
                R[k, l] = max(
                    safe_corr(X_mat[:, k], X_mat[:, l]),
                    safe_corr(Y_mat_col[:, k], Y_mat_col[:, l]),
                    safe_corr(X_mat[:, k], Y_mat_col[:, l]),
                    safe_corr(Y_mat_col[:, k], X_mat[:, l]),
                )
        R = np.nan_to_num(R)

        pos_corrs = R[R > 0]
        thresh    = np.percentile(pos_corrs, threshold_pct) if len(pos_corrs) > 0 else 0

        G = nx.Graph()
        for i, u in enumerate(sector_names):
            G.add_node(u)
            for j, v in enumerate(sector_names):
                if i < j and R[i, j] >= thresh:
                    G.add_edge(u, v, weight=R[i, j])
        G.remove_nodes_from(list(nx.isolates(G)))

        partition  = community_louvain.best_partition(G, weight='weight',
                                                      resolution=resolution, random_state=42)
        modularity = community_louvain.modularity(partition, G)
        num_cl     = len(set(partition.values()))

        return {
            "G": G, "R": R, "partition": partition, "modularity": modularity,
            "num_clusters": num_cl, "sector_names": sector_names,
            "X_dict": X_dict, "thresh": thresh,
        }

    file_bytes = uploaded_mip.read()
    try:
        m2 = build_wiod_model(file_bytes, num_sectors_m2, threshold_pct_m2, resolution_m2)
    except KeyError as ke:
        st.error(f"No se encontró la fila esperada en el CSV: {ke}")
        st.stop()
    except Exception as e:
        st.error(f"Error procesando MIP: {e}")
        st.stop()

    G2        = m2["G"]
    partition2= m2["partition"]
    X_dict2   = m2["X_dict"]
    snames    = m2["sector_names"]

    # KPIs
    st.markdown('<div class="section-title">INDICADORES — CORRELACIONES WIOD</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Sectores en red", G2.number_of_nodes())
    k2.metric("Clusters (Louvain)", m2["num_clusters"])
    k3.metric("Modularidad Q", f"{m2['modularity']:.4f}")
    k4.metric("Umbral de correlación", f"{m2['thresh']:.3f}")

    wtab1, wtab2 = st.tabs(["⬡  Red de correlaciones", "📊  Tabla de comunidades"])

    with wtab1:
        col_ctrl2, _ = st.columns([1,3])
        with col_ctrl2:
            max_n2 = st.slider("Nodos a mostrar", 5, G2.number_of_nodes(), min(35, G2.number_of_nodes()), key="m2_net")

        deg_cent2 = nx.degree_centrality(G2)
        top_n2    = sorted(deg_cent2, key=deg_cent2.get, reverse=True)[:max_n2]
        sub2      = G2.subgraph(top_n2)
        pos2      = nx.spring_layout(sub2, k=0.45, seed=42)

        edge_traces2 = []
        ew2 = [sub2[u][v]['weight'] for u,v in sub2.edges()]
        wmax2 = max(ew2) if ew2 else 1
        thresh2 = m2["thresh"]
        for (u,v),w in zip(sub2.edges(), ew2):
            x0,y0=pos2[u]; x1,y1=pos2[v]
            norm_w = (w - thresh2) / (1 - thresh2 + 1e-9) * 3 + 0.5
            edge_traces2.append(go.Scatter(
                x=[x0,x1,None], y=[y0,y1,None], mode="lines",
                line=dict(width=float(norm_w), color="rgba(100,116,139,0.4)"),
                hoverinfo="none", showlegend=False))

        active_X = [X_dict2.get(nd, 1) for nd in sub2.nodes()]
        min_X2, max_X2 = min(active_X), max(active_X)
        if max_X2 > min_X2:
            node_sizes2 = [300 + 3000*(x - min_X2)/(max_X2 - min_X2) for x in active_X]
        else:
            node_sizes2 = [800 for _ in active_X]

        node_trace2 = go.Scatter(
            x=[pos2[nd][0] for nd in sub2.nodes()],
            y=[pos2[nd][1] for nd in sub2.nodes()],
            mode="markers+text",
            marker=dict(size=[s**0.5 * 2.5 for s in node_sizes2],
                        color=[int(partition2[nd]) for nd in sub2.nodes()],
                        colorscale="Set2_r", showscale=False,
                        line=dict(width=2, color="white"), opacity=0.9),
            text=[nd[:18] for nd in sub2.nodes()],
            textfont=dict(size=7, family="Space Mono", color="#2c3e50"),
            textposition="top center",
            hovertext=[f"<b>{nd}</b><br>Cluster: {partition2[nd]}<br>"
                       f"X: {X_dict2.get(nd,0):,.0f}" for nd in sub2.nodes()],
            hoverinfo="text")

        fig_net2 = go.Figure(data=edge_traces2 + [node_trace2])
        fig_net2.update_layout(
            title=dict(text="Red de correlaciones WIOD — tamaño de nodo proporcional a Producción Bruta X",
                       font=dict(family="Space Mono", size=12, color="#94a3b8")),
            showlegend=False, hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=640, **PLOTLY_THEME)
        st.plotly_chart(fig_net2, use_container_width=True)

    with wtab2:
        df_wiod = pd.DataFrame({
            "sector":   list(partition2.keys()),
            "cluster":  list(partition2.values()),
            "X":        [X_dict2.get(s, 0) for s in partition2.keys()],
        })
        summary_wiod = (df_wiod.groupby("cluster")
                        .agg(tamaño=("sector","count"), X_total=("X","sum"))
                        .reset_index().sort_values("X_total", ascending=False)
                        .reset_index(drop=True))
        st.dataframe(summary_wiod, use_container_width=True)
        st.markdown("**Sectores por cluster**")
        sel_cl2 = st.selectbox("Cluster", sorted(df_wiod["cluster"].unique()),
                                format_func=lambda x: f"C{x}", key="m2_cl")
        st.dataframe(df_wiod[df_wiod["cluster"]==sel_cl2].sort_values("X", ascending=False)
                     .reset_index(drop=True), use_container_width=True)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  MÓDULO 3 — REGIONAL: ENTIDADES FEDERATIVAS + CONTAGIO           ║
# ║  ACTUALIZADO v2.1: carga automática de archivos .npy             ║
# ╚═══════════════════════════════════════════════════════════════════╝
elif module == MODULE_LABELS[2]:

    # ──────────────────────────────────────────────────────────────────
    # NUEVO ▶ Helper: detección y carga de estados desde disco
    # ──────────────────────────────────────────────────────────────────
    # Archivos mínimos requeridos por carpeta de estado
    _REQUIRED_NPY = {"Z.npy", "L.npy"}
    # Archivos opcionales enriquecen la visualización pero no son
    # indispensables para el análisis de clusters y contagio
    _OPTIONAL_NPY = {"A.npy", "X.npy", "Y.npy", "v.npy", "e.npy",
                     "VA_r.npy", "PO_r.npy", "FLQ.npy"}

    def _discover_states(base: Path) -> dict[str, Path]:
        """
        Recorre subdirectorios de `base` y devuelve un mapa
        {nombre_legible: Path} de todos los estados que contienen
        al menos los archivos .npy mínimos requeridos.
        """
        states: dict[str, Path] = {}
        for folder in sorted(base.iterdir()):
            if not folder.is_dir():
                continue
            files = {f.name for f in folder.iterdir() if f.suffix == ".npy"}
            if _REQUIRED_NPY.issubset(files):
                # Convierte el nombre de carpeta a título legible
                readable = folder.name.replace("_", " ").title()
                states[readable] = folder
        return states

    def _load_regional_npy(state_path: Path) -> dict:
        """
        Carga todas las matrices .npy disponibles en `state_path`.
        Devuelve un dict con las matrices y los nombres de sector
        (leídos de sectores.csv si existe en la carpeta o en el padre).
        """
        Z_r = np.load(state_path / "Z.npy")
        L_r = np.load(state_path / "L.npy")
        n   = Z_r.shape[0]

        # ── Nombres de sector ──────────────────────────────────────
        # Prioridad: sectores.csv en la carpeta del estado →
        #            sectores.csv en el directorio padre →
        #            etiquetas genéricas "Sector 001 … N"
        label_candidates = [
            state_path / "sectores.csv",
            state_path.parent / "sectores.csv",
        ]
        sector_names: list[str] = []
        for cand in label_candidates:
            if cand.exists():
                df_s = pd.read_csv(cand)
                col  = next((c for c in ["nombre", "name", "sector"] if c in df_s.columns), None)
                if col and len(df_s) >= n:
                    sector_names = df_s[col].astype(str).tolist()[:n]
                    break
        if not sector_names:
            sector_names = [f"Sector {i+1:03d}" for i in range(n)]

        # ── Matrices opcionales ────────────────────────────────────
        def _try_load(fname: str) -> np.ndarray | None:
            p = state_path / fname
            return np.load(p) if p.exists() else None

        return {
            "Z_r":   Z_r,
            "L_r":   L_r,
            "A":     _try_load("A.npy"),
            "X":     _try_load("X.npy"),
            "Y":     _try_load("Y.npy"),
            "v":     _try_load("v.npy"),
            "e":     _try_load("e.npy"),
            "VA_r":  _try_load("VA_r.npy"),
            "PO_r":  _try_load("PO_r.npy"),
            "FLQ":   _try_load("FLQ.npy"),
            "sector_names": sector_names,
            "n": n,
        }

    # ── Sidebar Regional ─────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="section-title">📂 Datos Regionales</div>', unsafe_allow_html=True)

        # ── NUEVO: selector automático de estado ──────────────────
        discovered = _discover_states(BASE_DIR.parent / "data2")

        if discovered:
            state_options = list(discovered.keys())
            selected_state_name = st.selectbox(
                "Estado (carga automática)",
                state_options,
                key="m3_state_sel",
                help="Carpetas con Z.npy y L.npy detectadas automáticamente.",
            )
            selected_state_path = discovered[selected_state_name]
            use_manual_upload   = st.checkbox(
                "Usar carga manual (CSV)", value=False, key="m3_manual",
                help="Activa esta opción si prefieres subir archivos CSV en vez de usar los .npy del disco.",
            )
        else:
            # Si no hay carpetas con .npy, caemos en modo manual
            st.info("No se detectaron carpetas con matrices .npy. Usa la carga manual.")
            use_manual_upload   = True
            selected_state_name = None
            selected_state_path = None

        # ── Carga manual (fallback) ───────────────────────────────
        if use_manual_upload or not discovered:
            uploaded_Zr = st.file_uploader("Matriz Zr (.csv)", type=["csv"], key="m3_Zr")
            uploaded_Lr = st.file_uploader("Matriz Lr (.csv)", type=["csv"], key="m3_Lr")
        else:
            uploaded_Zr = None
            uploaded_Lr = None

        st.markdown('<div class="section-title">⚙️ Parámetros Regionales</div>', unsafe_allow_html=True)
        threshold_pct_r = st.slider("Percentil de umbral correlaciones (%)", 50, 95, 75, key="m3_thresh")
        resolution_r    = st.slider("Resolución Louvain (γ)", 0.5, 2.0, 1.0, 0.1, key="m3_res")

        st.markdown('<div class="section-title">💥 Parámetros de Shock</div>', unsafe_allow_html=True)
        shock_value_r = st.number_input(
            "Magnitud del shock (MXN M)", value=-100.0, step=50.0, key="m3_shock",
            help="Valor negativo = contracción de demanda; positivo = expansión",
        )

    st.markdown("""
    <div class="alert-box">
      <strong>Módulo Regional:</strong> Implementa el análisis de vulnerabilidad estructural y
      mapeo de contagio financiero. Las matrices se cargan automáticamente desde archivos
      <code>.npy</code> por estado; la carga manual en CSV sigue disponible como alternativa.
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────
    # NUEVO ▶ Backend cacheado unificado: acepta npy (disco) o csv (manual)
    # ──────────────────────────────────────────────────────────────────
    @st.cache_data
    def build_regional_model_npy(
        Z_bytes: bytes, L_bytes: bytes,
        sector_names: list[str],
        threshold_pct: float, resolution: float,
    ):
        """
        Construye el modelo regional a partir de arrays serializados.
        Acepta los arrays como bytes (Z_bytes, L_bytes) para que
        @st.cache_data pueda hashearlos correctamente.
        """
        Z_r = np.frombuffer(Z_bytes, dtype=np.float64).reshape(len(sector_names), -1)
        L_r = np.frombuffer(L_bytes, dtype=np.float64).reshape(len(sector_names), -1)
        n   = len(sector_names)

        epsilon = 1e-9
        p = Z_r.sum(axis=0)
        s = Z_r.sum(axis=1)
        active_mask = (p > epsilon) & (s > epsilon)

        X_mat     = np.zeros_like(Z_r)
        Y_mat_col = np.zeros_like(Z_r)
        for j in range(n):
            if active_mask[j]:
                X_mat[:, j] = Z_r[:, j] / p[j]
        for i in range(n):
            if active_mask[i]:
                Y_mat_col[i, :] = Z_r[i, :] / s[i]

        def safe_corr(v1, v2):
            if np.std(v1) == 0 or np.std(v2) == 0:
                return 0.0
            return float(np.corrcoef(v1, v2)[0, 1])

        R_r = np.zeros((n, n))
        np.seterr(invalid="ignore")
        for k in range(n):
            for l in range(n):
                if k == l or not (active_mask[k] and active_mask[l]):
                    continue
                R_r[k, l] = max(
                    safe_corr(X_mat[:, k], X_mat[:, l]),
                    safe_corr(Y_mat_col[:, k], Y_mat_col[:, l]),
                    safe_corr(X_mat[:, k], Y_mat_col[:, l]),
                    safe_corr(Y_mat_col[:, k], X_mat[:, l]),
                )
        R_r = np.nan_to_num(R_r)

        pos_corrs = R_r[R_r > 0]
        thresh    = np.percentile(pos_corrs, threshold_pct) if len(pos_corrs) > 0 else 0

        G = nx.Graph()
        for i, u in enumerate(sector_names):
            if active_mask[i]:
                G.add_node(u, sector_idx=i)
                for j, v in enumerate(sector_names):
                    if i < j and active_mask[j] and R_r[i, j] >= thresh:
                        G.add_edge(u, v, weight=R_r[i, j])
        G.remove_nodes_from(list(nx.isolates(G)))

        partition  = community_louvain.best_partition(G, weight="weight",
                                                      resolution=resolution, random_state=42)
        modularity = community_louvain.modularity(partition, G)

        return {
            "G": G, "R_r": R_r, "L_r": L_r, "partition": partition,
            "modularity": modularity, "sector_names": sector_names,
            "active_mask": active_mask, "thresh": thresh,
            "n_active": int(active_mask.sum()), "n_total": n,
        }

    @st.cache_data
    def build_regional_model_csv(Zr_bytes: bytes, Lr_bytes: bytes,
                                  threshold_pct: float, resolution: float):
        """Ruta original: carga desde CSV. Mantenida para compatibilidad."""
        import io
        Z_r_df = pd.read_csv(io.BytesIO(Zr_bytes), index_col=0)
        L_r_df = pd.read_csv(io.BytesIO(Lr_bytes), index_col=0)
        raw_names    = [str(name) for name in Z_r_df.index]
        sector_names = [s[:28] + ("..." if len(s) > 28 else "") for s in raw_names]
        Z_r_df.index = sector_names; Z_r_df.columns = sector_names
        L_r_df.index = sector_names; L_r_df.columns = sector_names
        Z_r = Z_r_df.values
        L_r = L_r_df.values
        n   = len(sector_names)
        return build_regional_model_npy(
            Z_r.tobytes(), L_r.tobytes(), sector_names, threshold_pct, resolution
        )

    # ── Resolución de fuente de datos ─────────────────────
    mr = None
    _extra: dict = {}  # matrices opcionales (v, e, VA_r, etc.)

    if not use_manual_upload and selected_state_path is not None:
        # ── Ruta automática: .npy desde disco ────────────────
        try:
            raw = _load_regional_npy(selected_state_path)
            Z_r, L_r = raw["Z_r"], raw["L_r"]
            mr = build_regional_model_npy(
                Z_r.tobytes(), L_r.tobytes(),
                raw["sector_names"],
                threshold_pct_r, resolution_r,
            )
            _extra = raw  # guarda las matrices opcionales para KPIs adicionales
        except Exception as e:
            st.error(f"Error cargando matrices .npy de '{selected_state_name}': {e}")
            st.stop()

    elif use_manual_upload:
        # ── Ruta manual: CSV subido por el usuario ────────────
        if uploaded_Zr is None or uploaded_Lr is None:
            st.info("📂 Carga ambas matrices regionales (Zr y Lr) en el panel lateral.")
            st.markdown("""
            **Formato esperado:**
            - CSV con primera columna como índice (nombres de sectores)
            - Dimensión: n×n donde n es el número de sectores
            """)
            st.stop()
        try:
            Zr_bytes = uploaded_Zr.read()
            Lr_bytes = uploaded_Lr.read()
            mr = build_regional_model_csv(Zr_bytes, Lr_bytes, threshold_pct_r, resolution_r)
        except Exception as e:
            st.error(f"Error procesando matrices regionales CSV: {e}")
            st.stop()

    else:
        st.error("No hay fuente de datos configurada. Verifica la carpeta de datos o usa la carga manual.")
        st.stop()

    # A partir de aquí el código es idéntico al original, usando `mr`
    Gr          = mr["G"]
    L_r         = mr["L_r"]
    partition_r = mr["partition"]
    snames_r    = mr["sector_names"]
    n_r         = mr["n_total"]
    thresh_r    = mr["thresh"]

    # ── KPIs Regionales ──────────────────────────────────
    st.markdown(
        f'<div class="section-title">ESTRUCTURA REGIONAL — {(selected_state_name or "carga manual").upper()}</div>',
        unsafe_allow_html=True,
    )
    kr1, kr2, kr3, kr4, kr5 = st.columns(5)
    kr1.metric("Sectores totales",  n_r)
    kr2.metric("Sectores activos",  mr["n_active"])
    kr3.metric("Nodos en red",      Gr.number_of_nodes())
    kr4.metric("Comunidades",       len(set(partition_r.values())))
    kr5.metric("Modularidad Q",     f"{mr['modularity']:.4f}")

    # ── KPIs de matrices opcionales (solo en modo automático) ──
    if _extra.get("VA_r") is not None and _extra.get("PO_r") is not None:
        ek1, ek2, ek3 = st.columns(3)
        ek1.metric("VA total (Mmdp)",  f"{_extra['VA_r'].sum():,.1f}")
        ek2.metric("Personal ocupado", f"{int(_extra['PO_r'].sum()):,}")
        if _extra.get("X") is not None:
            ek3.metric("Producción total (Mmdp)", f"{_extra['X'].sum():,.1f}")

    # ── Selector de sector para shock ────────────────────
    active_sectors_r = list(Gr.nodes())
    shock_sector_r   = st.selectbox(
        "🎯 Sector de origen del shock financiero",
        active_sectors_r,
        help="Selecciona el sector que recibe el shock exógeno de demanda",
        key="m3_shock_sec",
    )

    # ── CALCULAR SHOCK ────────────────────────────────────
    @st.cache_data
    def compute_shock(Lr_bytes: bytes, n: int, snames: list, target_name: str, shock_val: float):
        L_r        = np.frombuffer(Lr_bytes, dtype=np.float64).reshape(n, n)
        idx        = snames.index(target_name)
        Delta_f    = np.zeros(n)
        Delta_f[idx] = shock_val
        Delta_x    = np.dot(L_r, Delta_f)
        Damage     = np.abs(Delta_x)
        total_loss = Damage.sum()
        multiplier = Delta_x.sum() / shock_val if shock_val != 0 else 0
        return (
            dict(zip(snames, Damage)),
            dict(zip(snames, Delta_x)),
            total_loss, multiplier,
        )

    Damage_dict_r, Delta_x_dict_r, total_loss_r, multiplier_r = compute_shock(
        mr["L_r"].tobytes(), n_r, snames_r, shock_sector_r, shock_value_r
    )

    # ── TABS REGIONALES ───────────────────────────────────
    rtab1, rtab2, rtab3, rtab4 = st.tabs([
        "⬡  Red Regional",
        "🗺️  Contagio (Antes/Después)",
        "📊  Ecosistema de Clusters",
        "📋  Datos completos",
    ])

    # Posiciones compartidas entre tabs
    deg_r   = nx.degree_centrality(Gr)
    _max_rn = min(40, Gr.number_of_nodes())

    with rtab1:
        st.markdown("**Red de comunidades estructurales — baseline pre-shock**")
        col_rn, _ = st.columns([1,3])
        with col_rn:
            max_rn = st.slider("Nodos a mostrar", 5, Gr.number_of_nodes(), _max_rn, key="m3_net")

        top_r   = sorted(deg_r, key=deg_r.get, reverse=True)[:max_rn]
        sub_r   = Gr.subgraph(top_r)
        pos_r   = nx.spring_layout(sub_r, k=0.45, seed=42)

        edge_r = []
        ew_r   = [sub_r[u][v]["weight"] for u,v in sub_r.edges()]
        for (u,v),w in zip(sub_r.edges(), ew_r):
            x0,y0=pos_r[u]; x1,y1=pos_r[v]
            norm = (w - thresh_r)/(1 - thresh_r + 1e-9)*4 + 0.5
            edge_r.append(go.Scatter(x=[x0,x1,None], y=[y0,y1,None], mode="lines",
                line=dict(width=float(norm), color="rgba(100,116,139,0.4)"),
                hoverinfo="none", showlegend=False))

        node_r = go.Scatter(
            x=[pos_r[nd][0] for nd in sub_r.nodes()],
            y=[pos_r[nd][1] for nd in sub_r.nodes()],
            mode="markers+text",
            marker=dict(
                size=16,
                color=[int(partition_r[nd]) for nd in sub_r.nodes()],
                colorscale="Turbo", showscale=True,
                colorbar=dict(title=dict(text="Cluster", font=dict(family="Space Mono", size=9))),
                line=dict(width=2, color="white"), opacity=0.9,
            ),
            text=[nd[:18] for nd in sub_r.nodes()],
            textfont=dict(size=7, family="Space Mono"),
            textposition="top center",
            hovertext=[f"<b>{nd}</b><br>Cluster: {partition_r[nd]}<br>"
                       f"Grado: {sub_r.degree(nd)}" for nd in sub_r.nodes()],
            hoverinfo="text")

        fig_rnet = go.Figure(data=edge_r + [node_r])
        fig_rnet.update_layout(
            title=dict(text="Red regional — comunidades estructurales (baseline)",
                       font=dict(family="Space Mono", size=12, color="#94a3b8")),
            showlegend=False, hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=620, **PLOTLY_THEME)
        st.plotly_chart(fig_rnet, use_container_width=True)

    with rtab2:
        st.markdown(f"""
        <div class="alert-box">
          <strong>Shock aplicado:</strong> {shock_value_r:,.0f} MXN M en
          <em>'{shock_sector_r}'</em> &nbsp;→&nbsp;
          Pérdida total regional estimada: <strong>{total_loss_r:,.2f} MXN M</strong>
          &nbsp;·&nbsp; Multiplicador: <strong>{multiplier_r:.4f}</strong>
        </div>
        """, unsafe_allow_html=True)

        col_pre, col_post = st.columns(2)

        with col_pre:
            st.markdown("**Pre-Shock: Estructura Baseline**")
            node_pre = go.Scatter(
                x=[pos_r.get(nd, [0])[0] for nd in sub_r.nodes() if nd in pos_r],
                y=[pos_r.get(nd, [0,0])[1] for nd in sub_r.nodes() if nd in pos_r],
                mode="markers+text",
                marker=dict(size=16,
                    color=[int(partition_r[nd]) for nd in sub_r.nodes() if nd in pos_r],
                    colorscale="Set2", showscale=False,
                    line=dict(width=2, color="white"), opacity=0.85),
                text=[nd[:14] for nd in sub_r.nodes() if nd in pos_r],
                textfont=dict(size=6, family="Space Mono"),
                textposition="top center",
                hovertext=[f"<b>{nd}</b><br>Baseline" for nd in sub_r.nodes() if nd in pos_r],
                hoverinfo="text")
            fig_pre = go.Figure(data=edge_r + [node_pre])
            fig_pre.update_layout(
                title=dict(text="Baseline (nodos uniformes)",
                           font=dict(family="Space Mono", size=11, color="#94a3b8")),
                showlegend=False, hovermode="closest",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=480, **PLOTLY_THEME)
            st.plotly_chart(fig_pre, use_container_width=True)

        with col_post:
            st.markdown("**Post-Shock: Desplazamiento por Contagio**")
            max_dmg = max(Damage_dict_r.values()) if Damage_dict_r else 1
            post_sizes = [
                max(8, 8 + 50 * (Damage_dict_r.get(nd, 0) / max_dmg))
                for nd in sub_r.nodes() if nd in pos_r
            ]
            node_colors_post = [int(partition_r[nd]) for nd in sub_r.nodes() if nd in pos_r]
            is_origin = [nd == shock_sector_r for nd in sub_r.nodes() if nd in pos_r]

            node_post = go.Scatter(
                x=[pos_r[nd][0] for nd in sub_r.nodes() if nd in pos_r],
                y=[pos_r[nd][1] for nd in sub_r.nodes() if nd in pos_r],
                mode="markers+text",
                marker=dict(size=post_sizes,
                    color=node_colors_post, colorscale="Set2", showscale=False,
                            line=dict(width=1.5, color="white"),
                    opacity=0.90),
                text=[nd[:14] for nd in sub_r.nodes() if nd in pos_r],
                textfont=dict(size=6, family="Space Mono"),
                textposition="top center",
                hovertext=[f"<b>{nd}</b><br>Pérdida: {Damage_dict_r.get(nd,0):,.2f} MXN M"
                           for nd in sub_r.nodes() if nd in pos_r],
                hoverinfo="text")
            fig_post = go.Figure(data=edge_r + [node_post])
            fig_post.update_layout(
                title=dict(text="Post-Shock (tamaño ∝ pérdida financiera | ⭕ = origen)",
                           font=dict(family="Space Mono", size=11, color="#94a3b8")),
                showlegend=False, hovermode="closest",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=480, **PLOTLY_THEME)
            st.plotly_chart(fig_post, use_container_width=True)

        st.markdown('<div class="section-title">TOP SECTORES IMPACTADOS</div>', unsafe_allow_html=True)
        df_damage_r = pd.DataFrame({
            "sector":     [nd for nd in Gr.nodes()],
            "cluster":    [f"C{partition_r[nd]}" for nd in Gr.nodes()],
            "pérdida_abs": [Damage_dict_r.get(nd, 0) for nd in Gr.nodes()],
            "Δx":          [Delta_x_dict_r.get(nd, 0) for nd in Gr.nodes()],
        }).sort_values("pérdida_abs", ascending=False)

        fig_bar_r = px.bar(df_damage_r.head(15),
            x="pérdida_abs", y="sector", orientation="h", color="cluster",
            title=f"Top 15 sectores más afectados por shock en '{shock_sector_r}'",
            labels={"pérdida_abs": "Pérdida absoluta (MXN M)", "sector": ""})
        fig_bar_r.update_layout(yaxis=dict(autorange="reversed"), height=440, **PLOTLY_THEME)
        st.plotly_chart(fig_bar_r, use_container_width=True)

    with rtab3:
        df_cluster_r = pd.DataFrame({
            "sector":  list(partition_r.keys()),
            "cluster": [f"C{v}" for v in partition_r.values()],
            "pérdida": [Damage_dict_r.get(s, 0) for s in partition_r.keys()],
        })
        cluster_dmg = (df_cluster_r.groupby("cluster")["pérdida"]
                       .sum().reset_index().sort_values("pérdida", ascending=False))

        ce1, ce2 = st.columns(2)
        with ce1:
            fig_eco = px.bar(cluster_dmg, x="cluster", y="pérdida", color="cluster",
                title="Daño sistémico por comunidad (Ecosystem Contagion)",
                labels={"pérdida": "Pérdida total MXN M", "cluster": "Comunidad Louvain"})
            fig_eco.update_layout(showlegend=False, height=380, **PLOTLY_THEME)
            st.plotly_chart(fig_eco, use_container_width=True)
        with ce2:
            fig_pie_r = px.pie(cluster_dmg, values="pérdida", names="cluster",
                color_discrete_sequence=px.colors.sequential.Turbo,
                title="Distribución del shock por comunidad")
            fig_pie_r.update_layout(height=380, **PLOTLY_THEME)
            st.plotly_chart(fig_pie_r, use_container_width=True)

        st.markdown('<div class="section-title">MÉTRICAS DE CONTAGIO</div>', unsafe_allow_html=True)
        km1, km2, km3, km4 = st.columns(4)
        km1.metric("Pérdida total regional", f"{total_loss_r:,.1f} M")
        km2.metric("Multiplicador regional", f"{multiplier_r:.4f}")
        km3.metric("Cluster más afectado",   cluster_dmg.iloc[0]["cluster"])
        km4.metric("Pérdida en cluster top", f"{cluster_dmg.iloc[0]['pérdida']:,.1f} M")

    with rtab4:
        st.markdown("**Tabla completa — daño financiero por sector y comunidad**")
        df_full_r = pd.DataFrame({
            "sector":      [nd for nd in Gr.nodes()],
            "cluster":     [f"C{partition_r[nd]}" for nd in Gr.nodes()],
            "pérdida_abs": [Damage_dict_r.get(nd, 0) for nd in Gr.nodes()],
            "Δx":          [Delta_x_dict_r.get(nd, 0) for nd in Gr.nodes()],
            "grado":       [Gr.degree(nd) for nd in Gr.nodes()],
        }).sort_values("pérdida_abs", ascending=False).reset_index(drop=True)

        rf1, rf2 = st.columns(2)
        f_cl_r   = rf1.multiselect("Filtrar cluster", sorted(df_full_r["cluster"].unique()), key="m3_fcl")
        f_sort_r = rf2.selectbox("Ordenar por", ["pérdida_abs", "Δx", "grado"], key="m3_fs")

        df_view_r = df_full_r.copy()
        if f_cl_r: df_view_r = df_view_r[df_view_r["cluster"].isin(f_cl_r)]
        df_view_r = df_view_r.sort_values(f_sort_r, ascending=False).reset_index(drop=True)

        st.dataframe(df_view_r, use_container_width=True, height=480)
        st.download_button(
            "⬇ Descargar CSV de contagio",
            df_view_r.to_csv(index=False).encode(),
            f"contagio_{(selected_state_name or 'regional').replace(' ','_')[:20]}.csv",
            "text/csv",
        )


# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown("""
<hr style="margin-top:40px; border-color:#e2e8f0;">
<div style="text-align:center; padding:16px 0; font-family:'Space Mono',monospace;
            font-size:10px; letter-spacing:2px; color:#64748b;">
  CLUSTER INTELLIGENCE v2.1 · MIP · LEONTIEF + WIOD + LOUVAIN · ANÁLISIS REGIONAL DE CONTAGIO
</div>
""", unsafe_allow_html=True)
