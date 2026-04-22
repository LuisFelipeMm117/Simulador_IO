# pages/3_Panorama_Estatal.py
"""
Página 3: Radiografía económica de un estado (sin simulación).
"""
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
from loader import ModeloEconomico

st.set_page_config(page_title="Panorama Estatal", layout="wide", page_icon="🔍")

@st.cache_resource(show_spinner=False)
def cargar_modelo():
    return ModeloEconomico("data/")

modelo = cargar_modelo()

st.title("🔍 Panorama Económico Estatal")
st.caption("Estructura sectorial, multiplicadores y coeficientes técnicos de cada estado.")
st.divider()

estado_nombre = st.selectbox("🗺 Estado", sorted(modelo.mapa_estados.keys()))
estado_key    = modelo.mapa_estados[estado_nombre]

d    = modelo._load_estado(estado_key)
info = modelo.info_estado(estado_key)
res_row = modelo.df_resumen[modelo.df_resumen["estado"] == estado_key]

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.subheader(f"📌 {estado_nombre} — Indicadores clave")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Producción total (Mmdp)", f"${info['X_total_mmdp']:,.0f}")
k2.metric("Valor agregado (Mmdp)",   f"${info['VA_total_mmdp']:,.0f}")
k3.metric("Empleo (puestos)",        f"{info['PO_total']:,}")
k4.metric("Sectores activos",        f"{info['sectores_activos']}/78")
if info["mult_promedio"]:
    k5.metric("Mult. promedio",      f"{info['mult_promedio']:.4f}")

st.divider()

t1, t2, t3, t4 = st.tabs(["🏭 Estructura productiva", "🏆 Multiplicadores", "🔗 Coeficientes FLQ", "📊 Comparativa nacional"])

# ── TAB 1: Estructura productiva ──────────────────────────────────────────────
with t1:
    activos = d["VA_r"] > 0
    df_struct = pd.DataFrame({
        "scian":   modelo.sectores,
        "nombre":  [modelo.sector_names[s] for s in modelo.sectores],
        "X_pesos": d["X"] / 1e-6,
        "VA_pesos":d["VA_r"] / 1e-6,
        "PO":      d["PO_r"],
        "activo":  activos,
    }).query("activo").sort_values("X_pesos", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top 15 sectores por producción")
        ch1 = alt.Chart(df_struct.head(15)).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("X_pesos:Q", title="Producción (MXN)"),
            y=alt.Y("nombre:N", sort="-x", title=None),
            color=alt.Color("X_pesos:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=[alt.Tooltip("nombre:N"), alt.Tooltip("X_pesos:Q", format="$,.0f", title="Producción (MXN)")]
        ).properties(height=380)
        st.altair_chart(ch1, use_container_width=True)

    with col2:
        st.markdown("#### Top 15 sectores por empleo")
        ch2 = alt.Chart(df_struct.sort_values("PO", ascending=False).head(15)).mark_bar(
            cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("PO:Q", title="Personal ocupado"),
            y=alt.Y("nombre:N", sort="-x", title=None),
            color=alt.Color("PO:Q", scale=alt.Scale(scheme="purples"), legend=None),
            tooltip=[alt.Tooltip("nombre:N"), alt.Tooltip("PO:Q", format=",.0f", title="Empleo")]
        ).properties(height=380)
        st.altair_chart(ch2, use_container_width=True)

    # Tabla estructural
    st.markdown("#### Tabla de estructura sectorial")
    df_tabla_s = df_struct[["scian","nombre","X_pesos","VA_pesos","PO"]].rename(columns={
        "scian":"SCIAN","nombre":"Sector","X_pesos":"Producción (MXN)",
        "VA_pesos":"Valor Agregado (MXN)","PO":"Empleo (puestos)"
    })
    st.dataframe(df_tabla_s.style.format({
        "Producción (MXN)":      "${:,.0f}",
        "Valor Agregado (MXN)":  "${:,.0f}",
        "Empleo (puestos)":      "{:,.0f}",
    }), use_container_width=True, height=380)

# ── TAB 2: Multiplicadores ────────────────────────────────────────────────────
with t2:
    mult_prod = d["L"].sum(axis=0)
    mult_ing  = modelo.v_n * mult_prod

    df_mult = pd.DataFrame({
        "scian":    modelo.sectores,
        "nombre":   [modelo.sector_names[s] for s in modelo.sectores],
        "mult_X":   mult_prod,
        "mult_VA":  mult_ing,
        "e_j":      d["e"],
        "activo":   d["VA_r"] > 0,
    }).query("activo").sort_values("mult_X", ascending=False)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Multiplicadores de producción")
        ch3 = alt.Chart(df_mult.head(20)).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("mult_X:Q", title="Multiplicador producción",
                     scale=alt.Scale(domain=[1, df_mult["mult_X"].max()*1.05])),
            y=alt.Y("nombre:N", sort="-x", title=None),
            color=alt.Color("mult_X:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=[alt.Tooltip("nombre:N"), alt.Tooltip("mult_X:Q", format=".4f")]
        ).properties(height=500)
        st.altair_chart(ch3, use_container_width=True)

    with col4:
        st.markdown("#### Multiplicadores de ingreso (VA)")
        ch4 = alt.Chart(df_mult.sort_values("mult_VA", ascending=False).head(20)).mark_bar(
            cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("mult_VA:Q", title="Multiplicador ingreso"),
            y=alt.Y("nombre:N", sort="-x", title=None),
            color=alt.Color("mult_VA:Q", scale=alt.Scale(scheme="greens"), legend=None),
            tooltip=[alt.Tooltip("nombre:N"), alt.Tooltip("mult_VA:Q", format=".4f")]
        ).properties(height=500)
        st.altair_chart(ch4, use_container_width=True)

# ── TAB 3: FLQ ───────────────────────────────────────────────────────────────
with t3:
    FLQ = d["FLQ"]
    activos_idx = np.where(d["VA_r"] > 0)[0]
    FLQ_sub = FLQ[np.ix_(activos_idx, activos_idx)]
    nombres_activos = [
        f"{modelo.sector_names[modelo.sectores[i]][:20]}_{i}"
        for i in activos_idx
]

    st.markdown("#### Matriz FLQ (sectores activos)")
    st.caption("FLQ_ij: fracción del coeficiente técnico nacional satisfecha localmente. 1 = abastecimiento local completo.")

    df_flq_heat = pd.DataFrame(FLQ_sub, index=nombres_activos, columns=nombres_activos)
    df_flq_heat = df_flq_heat.fillna(0)

    st.dataframe(df_flq_heat, use_container_width=True, height=500)

    # ── Diagonal ─────────────────────────
    st.markdown("#### LQ diagonal promedio por sector")

    diag = pd.DataFrame({
        "nombre": nombres_activos,
        "FLQ_diag": np.diag(FLQ_sub)
    }).sort_values("FLQ_diag", ascending=False)

    ch5 = alt.Chart(diag.head(20)).mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    ).encode(
        x=alt.X("FLQ_diag:Q", title="FLQ diagonal", scale=alt.Scale(domain=[0,1])),
        y=alt.Y("nombre:N", sort="-x", title=None),
        color=alt.Color("FLQ_diag:Q", scale=alt.Scale(scheme="oranges"), legend=None),
        tooltip=[
            alt.Tooltip("nombre:N"),
            alt.Tooltip("FLQ_diag:Q", format=".4f")
        ]
    ).properties(height=420)

    st.altair_chart(ch5, use_container_width=True)

# ── TAB 4: Comparativa nacional ───────────────────────────────────────────────
with t4:
    df_res = modelo.df_resumen.copy()
    df_res["nombre_legible"] = df_res["estado"].apply(
        lambda k: {v: u for u, v in modelo.mapa_estados.items()}.get(k, k)
    )
    df_res["es_seleccionado"] = df_res["estado"] == estado_key

    st.markdown("#### Multiplicador promedio — todos los estados")
    ch6 = alt.Chart(df_res).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("mult_promedio:Q", title="Mult. promedio producción",
                 scale=alt.Scale(domain=[1, df_res["mult_promedio"].max()*1.05])),
        y=alt.Y("nombre_legible:N", sort="-x", title=None),
        color=alt.condition(
            alt.datum.es_seleccionado,
            alt.value("#DC2626"),
            alt.Color("mult_promedio:Q", scale=alt.Scale(scheme="blues"), legend=None)
        ),
        tooltip=[
            alt.Tooltip("nombre_legible:N", title="Estado"),
            alt.Tooltip("mult_promedio:Q", title="Mult. promedio", format=".4f"),
            alt.Tooltip("sectores_activos:Q", title="Sectores activos"),
        ]
    ).properties(height=700)
    st.altair_chart(ch6, use_container_width=True)
    st.caption(f"🔴 Estado seleccionado: **{estado_nombre}**")
