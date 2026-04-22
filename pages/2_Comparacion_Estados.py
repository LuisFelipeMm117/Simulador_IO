# pages/2_Comparacion_Estados.py
"""
Página 2: Comparación del mismo shock en los 32 estados.
"""
import streamlit as st
import altair as alt
import pandas as pd
from loader import ModeloEconomico

st.set_page_config(page_title="Comparación por Estado", layout="wide", page_icon="🗺")

@st.cache_resource(show_spinner=False)
def cargar_modelo():
    return ModeloEconomico("data/")

modelo = cargar_modelo()

st.title("🗺 Comparación Regional — Mismo Shock en 32 Estados")
st.caption("¿Dónde tiene mayor impacto un mismo estímulo? Compara multiplicadores entre entidades.")
st.divider()

# ── Controles ─────────────────────────────────────────────────────────────────
df_sec = modelo.df_sectores
opciones = [f"{r.scian} — {r.nombre}" for _, r in df_sec.iterrows()]
sel = st.selectbox("🏭 Sector a comparar", opciones, index=30)
sector_idx = df_sec[df_sec.apply(lambda r: f"{r.scian} — {r.nombre}" == sel, axis=1)].index[0]
sector_scian = df_sec.loc[sector_idx, "scian"]
sector_name  = df_sec.loc[sector_idx, "nombre"]

monto_col, _ = st.columns([1, 2])
with monto_col:
    monto = st.number_input("Monto del shock (MXN)", value=100_000_000.0,
                             step=10_000_000.0, format="%.0f")

if st.button("▶ Comparar estados", type="primary"):
    with st.spinner("Ejecutando simulación en los 32 estados…"):
        st.session_state["df_comp"] = modelo.comparar_estados(sector_idx, monto)

# ── cargar datos persistentes ─────────────────────────
df_comp = st.session_state.get("df_comp")

if df_comp is None:
    st.info("Ejecuta la simulación para ver resultados")
elif df_comp.empty:
    st.warning("El sector seleccionado no tiene actividad en ningún estado.")
else:
    st.success(f"Comparando **{len(df_comp)} estados** con un shock de ${monto:,.0f} MXN en **{sector_name}**")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────
t1, t2, t3 = st.tabs(["📈 Multiplicadores", "💰 Impacto Absoluto", "📋 Tabla"])

if df_comp is None:
    st.info("Ejecuta la simulación para ver resultados")

elif df_comp.empty:
    st.warning("El sector seleccionado no tiene actividad en ningún estado.")
else:
    with t1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Multiplicador de producción por estado")

            min_v = df_comp["mult_produccion"].min()
            max_v = df_comp["mult_produccion"].max()
            padding = (max_v - min_v) * 0.05 if max_v > min_v else 0.05

            ch = alt.Chart(df_comp).mark_bar(
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4
            ).encode(
                x=alt.X(
                    "mult_produccion:Q",
                    title="Multiplicador producción",
                    scale=alt.Scale(domain=[max(0, min_v - padding), max_v + padding])
                ),
                y=alt.Y("estado:N", sort="-x", title=None),
                color=alt.Color("mult_produccion:Q", scale=alt.Scale(scheme="blues"), legend=None),
                tooltip=[
                    alt.Tooltip("estado:N", title="Estado"),
                    alt.Tooltip("mult_produccion:Q", format=".4f"),
                    alt.Tooltip("mult_ingreso:Q", format=".4f"),
                ]
            ).properties(height=680)

            st.altair_chart(ch, use_container_width=True)
        with col2:
            st.markdown("#### Multiplicador de ingreso (valor agregado) por estado")

            ch2 = alt.Chart(df_comp).mark_bar(
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4
            ).encode(
                x=alt.X("mult_ingreso:Q", title="Multiplicador ingreso"),
                y=alt.Y("estado:N", sort="-x", title=None),
                color=alt.Color(
                    "mult_ingreso:Q",
                    scale=alt.Scale(scheme="greens"),
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip("estado:N"),
                    alt.Tooltip("mult_ingreso:Q", format=".4f"),
                ]
            ).properties(height=680)

            st.altair_chart(ch2, use_container_width=True)
    with t2:
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Producción total inducida (MXN)")
            ch3 = alt.Chart(df_comp).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("delta_X_pesos:Q", title="ΔX Producción (MXN)"),
                y=alt.Y("estado:N", sort="-x", title=None),
                color=alt.Color("delta_X_pesos:Q", scale=alt.Scale(scheme="blues"), legend=None),
                tooltip=[alt.Tooltip("estado:N"), alt.Tooltip("delta_X_pesos:Q", format="$,.0f")]
            ).properties(height=680)
            st.altair_chart(ch3, use_container_width=True)

        with col4:
            st.markdown("#### Empleo generado (puestos de trabajo)")
            ch4 = alt.Chart(df_comp).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("delta_E:Q", title="Puestos generados"),
                y=alt.Y("estado:N", sort="-x", title=None),
                color=alt.Color("delta_E:Q", scale=alt.Scale(scheme="purples"), legend=None),
                tooltip=[alt.Tooltip("estado:N"), alt.Tooltip("delta_E:Q", format=",.0f")]
            ).properties(height=680)
            st.altair_chart(ch4, use_container_width=True)

    with t3:
        st.markdown("#### Resultados completos")
        df_show = df_comp.drop(columns=["estado_key"]).rename(columns={
            "estado":           "Estado",
            "mult_produccion":  "Mult. Producción",
            "mult_ingreso":     "Mult. Ingreso",
            "mult_empleo_1M":   "Empleo / 1M MXN",
            "delta_X_pesos":    "ΔX Producción (MXN)",
            "delta_VA_pesos":   "ΔVA Ingreso (MXN)",
            "delta_E":          "ΔEmpleo (puestos)",
        })
        styled = (
            df_show.style
            .format({
                "Mult. Producción":      "{:.4f}",
                "Mult. Ingreso":         "{:.4f}",
                "Empleo / 1M MXN":       "{:.2f}",
                "ΔX Producción (MXN)":   "${:,.0f}",
                "ΔVA Ingreso (MXN)":     "${:,.0f}",
                "ΔEmpleo (puestos)":     "{:,.0f}",
            })
            .background_gradient(cmap="Blues",   subset=["Mult. Producción"])
            .background_gradient(cmap="Greens",  subset=["Mult. Ingreso"])
            .background_gradient(cmap="Purples", subset=["ΔEmpleo (puestos)"])
        )
        st.dataframe(styled, use_container_width=True, height=500)
        csv = df_show.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Descargar tabla", csv,
                           f"comparacion_{sector_scian}.csv", "text/csv")
