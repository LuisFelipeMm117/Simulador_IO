# app.py
"""
Simulador Económico Regional — MIP México
Correcciones v4.2:
  BUG-1: Sector inactivo → L[:,j] casi diagonal → barras vacías en gráfico.
  BUG-2: Modo porcentaje con sector inactivo → X[j]=1e-10 → monto=$0.
  BUG-3: Página "Simulacion de Shocks" en blanco (era placeholder vacío).
  BUG-4: Filtro de gráfico basado en base (X) excluía sectores con impacto real.
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from loader import ModeloEconomico

st.set_page_config(
    page_title="Simulador MIP Regional",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; color: #555; }
    .block-container { padding-top: 1.5rem; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Oculta TODO el toolbar */
    header [data-testid="stToolbar"] * {
        display: none !important;
    }

    /* Muestra SOLO el primer elemento (Share) */
    header [data-testid="stToolbar"] > div:first-child {
        display: flex !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource(show_spinner="Cargando modelo económico…")
def cargar_modelo() -> ModeloEconomico:
    return ModeloEconomico("data/")

modelo = cargar_modelo()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Simulador MIP\nRegional México")
    st.caption("Modelo Leontief · FLQ + RAS · Base 2018")
    st.divider()

    # Estado
    nombres_estados = sorted(modelo.mapa_estados.keys())
    estado_nombre   = st.selectbox("🗺 Estado", nombres_estados, index=0)
    estado_key      = modelo.mapa_estados[estado_nombre]

    info = modelo.info_estado(estado_key)
    st.caption(
        f"**X total:** ${info['X_total_mmdp']:,.0f} Mmdp  \n"
        f"**Empleo:** {info['PO_total']:,} puestos  \n"
        f"**Sectores activos:** {info['sectores_activos']}/{modelo.n}"
    )
    st.divider()

    # ── Sector ────────────────────────────────────────────────────────────
    d_estado  = modelo._load_estado(estado_key)
    va_estado = d_estado["VA_r"]
    X_estado  = d_estado["X"]

    df_sec = modelo.df_sectores.copy()
    df_sec["activo_estado"] = [bool(va_estado[i] > 0) for i in df_sec["indice"]]

    # BUG-3 FIX: checkbox para filtrar sectores activos (default=True para que
    # por defecto el usuario vea sectores con resultados significativos)
    solo_activos = st.checkbox("Solo sectores activos en este estado", value=True)
    df_filtrado  = df_sec[df_sec["activo_estado"]] if solo_activos else df_sec

    opciones_sector = [
        f"{row.scian} — {row.nombre}"
        for _, row in df_filtrado.iterrows()
    ]
    sel_sector = st.selectbox("🏭 Sector económico", opciones_sector, index=0)

    # Obtener el índice del modelo a partir del código SCIAN
    scian_sel   = sel_sector.split(" — ")[0]
    sector_row  = df_filtrado[df_filtrado["scian"].astype(str) == str(scian_sel)].iloc[0]
    sector_idx  = int(sector_row["indice"])     # posición real 0..77
    sector_scian = str(sector_row["scian"])
    sector_name  = sector_row["nombre"]
    sector_activo = bool(va_estado[sector_idx] > 0)

    if not sector_activo:
        st.warning(
            f"⚠ **{sector_name}** no tiene actividad registrada en {estado_nombre}. "
            "Los resultados reflejarán impacto directo únicamente (mult ≈ 1.0)."
        )

    st.divider()

    # ── Tipo y monto ──────────────────────────────────────────────────────
    tipo_shock = st.radio(
        "💥 Tipo de shock",
        ["Monto en pesos (MXN)", "Porcentaje sobre base (%)"],
        index=0
    )

    if tipo_shock == "Monto en pesos (MXN)":
        monto_pesos = st.number_input(
            "Monto (MXN)", value=100_000_000.0,
            min_value=-1e12, max_value=1e12,
            step=10_000_000.0, format="%.0f"
        )
    else:
        pct_input = st.number_input(
            "Porcentaje (%)", value=10.0,
            min_value=-100.0, max_value=500.0,
            step=1.0, format="%.1f"
        )

        # ── CORRECCIÓN BUG-2 ─────────────────────────────────────────────
        # X_estado[sector_idx] = 1e-10 cuando el sector es inactivo
        # → % de 1e-10 ≈ $0 → resultados en cero.
        # Fix: usar X_n (nacional) escalado al tamaño relativo del estado.
        if sector_activo:
            base_mxn = float(X_estado[sector_idx]) / 1e-6
        else:
            ratio_estado = float(va_estado.sum()) / max(float(modelo.X_n.sum()), 1e-10)
            base_mxn     = float(modelo.X_n[sector_idx]) / 1e-6 * ratio_estado

        monto_pesos = base_mxn * (pct_input / 100.0)
        st.caption(
            f"Base: ${base_mxn:,.0f} MXN  \n"
            f"Shock: ${monto_pesos:,.0f} MXN"
        )

    st.divider()
    simular = st.button("▶ Simular shock", use_container_width=True, type="primary")


# ── Cuerpo principal ──────────────────────────────────────────────────────────
st.title("📊 Simulador Económico Regional — MIP México")
st.caption(
    "Modelo de Insumo-Producto regionalizado (Flegg FLQ + RAS) · "
    f"{modelo.n} sectores SCIAN · {len(modelo.estados_raw)} estados · Base INEGI 2018"
)
st.markdown("""
### 🧠 ¿Qué hace esta herramienta?

Este simulador estima el impacto económico total (producción, empleo e ingreso)
de invertir o reducir recursos en un sector específico dentro de un estado.

Permite identificar qué sectores generan mayor efecto multiplicador en la economía.
""")

if not simular:
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.info("**Paso 1:** Selecciona un estado en el panel izquierdo.")
    c2.info("**Paso 2:** Elige el sector económico a perturbar.")
    c3.info("**Paso 3:** Define el monto y presiona **▶ Simular shock**.")
    st.divider()
    st.subheader(f"🏆 Top 10 sectores por multiplicador — {estado_nombre}")
    df_top = modelo.top_multiplicadores(estado_key, top_n=10)
    ch_top = alt.Chart(df_top).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("mult:Q", title="Multiplicador de producción",
                 scale=alt.Scale(domain=[1, df_top["mult"].max() * 1.05])),
        y=alt.Y("nombre:N", sort="-x", title=None),
        color=alt.Color("mult:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=[
            alt.Tooltip("nombre:N", title="Sector"),
            alt.Tooltip("mult:Q",   title="Multiplicador", format=".4f"),
            alt.Tooltip("X_mmdp:Q",title="Producción (Mmdp)", format=",.1f"),
        ]
    ).properties(height=340)
    st.altair_chart(ch_top, use_container_width=True)
    st.stop()


if monto_pesos == 0:
    st.warning("El monto es cero. No se genera ningún efecto.")
    st.stop()

# Simulación
with st.spinner("Calculando impactos…"):
    res = modelo.simular(estado_key, sector_idx, monto_pesos)

df = res["df_detalle"]

# Encabezado resultado
signo      = "+" if monto_pesos > 0 else ""
tipo_label = "expansión" if monto_pesos > 0 else "contracción"
st.success(
    f"**Shock de {tipo_label}:** {signo}${monto_pesos:,.0f} MXN en "
    f"**{sector_name}** ({sector_scian}) — {estado_nombre}"
)
if not sector_activo:
    st.warning(
        "⚠ Sector **sin actividad local** en este estado. "
        "Se muestra el impacto directo únicamente (sin encadenamientos)."
    )

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.subheader("Impactos agregados")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Producción inducida", f"${res['delta_X_total_pesos']:,.0f}",
          help="Producción total adicional en la economía estatal.")
k2.metric("Valor agregado",      f"${res['delta_VA_total_pesos']:,.0f}",
          help="Remuneraciones + excedente de operación (proxy PIB estatal).")
k3.metric("Empleo generado",     f"{res['delta_E_total']:,.0f} puestos",
          help="Puestos directos e indirectos estimados.")
k4.metric("Mult. producción",    f"{res['mult_produccion']:.4f}",
          help="Pesos de producción total por cada peso de shock.")
k5.metric("Mult. ingreso",       f"{res['mult_ingreso']:.4f}",
          help="Pesos de valor agregado por cada peso de shock.")

mp = res["mult_produccion"]
if   mp >= 1.4: st.success(f"🟢 **Alto encadenamiento** (mult. {mp:.3f}).")
elif mp >= 1.2: st.info(   f"🔵 **Encadenamiento moderado-alto** (mult. {mp:.3f}).")
elif mp >= 1.1: st.warning(f"🟡 **Encadenamiento moderado** (mult. {mp:.3f}).")
else:           st.error(  f"🔴 **Bajo encadenamiento** (mult. {mp:.3f}): sector inactivo o alta importación.")

st.divider()

# ── CORRECCIÓN BUG-4: filtro basado en impacto, no en base ───────────────────
# Antes: percentil 25 de df["base"] → excluía sectores con alta demanda pero
#        poca producción (servicios, por ej.), dejando el gráfico vacío.
# Ahora: filtrar por impacto absoluto > 1 MXN, mostrando todo lo real.
df_con_impacto = df[
    (df["activo"]) &
    (df["indice"] != sector_idx) &
    (df["delta_X_pesos"].abs() >= 1.0)          # al menos $1 MXN de impacto
].copy()

# FIX: sort by |ΔX| so negative shocks show largest contractions first.
# ascending=False on raw value showed near-zero sectors at top for neg. shocks.
df_top12 = (
    df_con_impacto
    .assign(_abs_dx=df_con_impacto["delta_X_pesos"].abs())
    .sort_values("_abs_dx", ascending=False)
    .drop(columns=["_abs_dx"])
    .head(12)
)

# ── Pestañas ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Producción", "💰 Ingreso y Empleo", "📋 Detalle"])

# ═════════════ TAB 1 ══════════════════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns([3, 1])

    with col_a:
        st.markdown("#### Impacto en producción — top sectores")
        if df_top12.empty:
            st.info(
                "No hay encadenamientos locales significativos para este sector en "
                f"{estado_nombre}. Verifica la pestaña **Detalle** para ver el "
                "impacto directo."
            )
        else:
            # FIX: use abs_delta for Y-axis sort order so bars rank by magnitude,
            # not by sign. Without this, negative shocks show reversed Y ordering.
            df_top12_chart = df_top12.copy()
            df_top12_chart["_abs_dx"] = df_top12_chart["delta_X_pesos"].abs()
            ch = alt.Chart(df_top12_chart).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4
            ).encode(
                x=alt.X("delta_X_pesos:Q", title="ΔX Producción (MXN)"),
                y=alt.Y("nombre:N", sort=alt.EncodingSortField(field="_abs_dx", order="descending"), title=None),
                color=alt.condition(
                    alt.datum.delta_X_pesos > 0,
                    alt.value("#2563EB"), alt.value("#DC2626")
                ),
                tooltip=[
                    alt.Tooltip("nombre:N",          title="Sector"),
                    alt.Tooltip("scian:N",            title="SCIAN"),
                    alt.Tooltip("delta_X_pesos:Q",   title="ΔX (MXN)",         format="$,.0f"),
                    alt.Tooltip("share_produccion:Q",title="Share del impacto", format=".2%"),
                    alt.Tooltip("variacion_pct:Q",   title="Var. base (%)",     format=".2f"),
                ]
            ).properties(height=380)
            st.altair_chart(ch, use_container_width=True)

    with col_b:
        st.markdown("#### Share del impacto")
        if not df_top12.empty and df_top12["share_produccion"].sum() > 0:
            df_pie = df_top12.head(6).copy()
            df_pie["share_pct"] = df_pie["share_produccion"] * 100
            pie = alt.Chart(df_pie).mark_arc(innerRadius=45).encode(
                theta=alt.Theta("share_pct:Q"),
                color=alt.Color("nombre:N",
                                legend=alt.Legend(orient="bottom", labelLimit=160)),
                tooltip=[
                    alt.Tooltip("nombre:N",    title="Sector"),
                    alt.Tooltip("share_pct:Q", title="Share (%)", format=".1f"),
                ]
            ).properties(height=280)
            st.altair_chart(pie, use_container_width=True)
        else:
            st.info("Sin impactos indirectos para mostrar.")

    if not df_top12.empty:
        st.markdown("#### Top 5 sectores receptores")
        cols = st.columns(5)
        for i, (_, row) in enumerate(df_top12.head(5).iterrows()):
            cols[i].metric(
                label=row["nombre"][:28],
                value=f"${row['delta_X_pesos']:,.0f}",
                delta=f"{row['share_produccion']:.1%} del total",
            )
    else:
        # Mostrar impacto directo en el sector seleccionado
        direct = df[df["indice"] == sector_idx]
        if not direct.empty:
            row = direct.iloc[0]
            st.markdown("#### Impacto directo")
            st.metric(
                label=row["nombre"][:40],
                value=f"${row['delta_X_pesos']:,.0f}",
                delta="Impacto directo (sector sin encadenamientos locales)",
            )


# ═════════════ TAB 2 ══════════════════════════════════════════════════════════
with tab2:
    # FIX: sort by |ΔVA| so negative shocks show largest income contractions first
    df_ie  = (
        df_con_impacto[df_con_impacto["delta_VA_pesos"].abs() >= 1]
        .assign(_abs_va=lambda d: d["delta_VA_pesos"].abs())
        .sort_values("_abs_va", ascending=False)
        .drop(columns=["_abs_va"])
        .head(12)
    )
    # FIX: sort by |ΔE| so negative shocks show largest job losses first
    df_emp = (
        df_con_impacto[df_con_impacto["delta_E"].abs() > 0]
        .assign(_abs_e=lambda d: d["delta_E"].abs())
        .sort_values("_abs_e", ascending=False)
        .drop(columns=["_abs_e"])
        .head(12)
    )

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### Impacto en valor agregado por sector")
        if df_ie.empty:
            st.info("Sin impacto en ingreso en sectores secundarios.")
        else:
            ch_va = alt.Chart(df_ie).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4
            ).encode(
                x=alt.X("delta_VA_pesos:Q", title="ΔVA (MXN)"),
                y=alt.Y("nombre:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum.delta_VA_pesos > 0,
                    alt.value("#059669"), alt.value("#DC2626")
                ),
                tooltip=[
                    alt.Tooltip("nombre:N",        title="Sector"),
                    alt.Tooltip("delta_VA_pesos:Q",title="ΔVA (MXN)",  format="$,.0f"),
                    alt.Tooltip("v_j:Q",           title="Coef. VA",   format=".4f"),
                ]
            ).properties(height=360)
            st.altair_chart(ch_va, use_container_width=True)

    with col_d:
        st.markdown("#### Impacto en empleo por sector")
        if df_emp.empty:
            st.info("Sin impacto en empleo en sectores secundarios.")
        else:
            ch_e = alt.Chart(df_emp).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4
            ).encode(
                x=alt.X("delta_E:Q", title="Puestos de trabajo"),
                y=alt.Y("nombre:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum.delta_E > 0,
                    alt.value("#7C3AED"), alt.value("#DC2626")
                ),
                tooltip=[
                    alt.Tooltip("nombre:N",  title="Sector"),
                    alt.Tooltip("delta_E:Q", title="ΔEmpleo", format=",.0f"),
                    alt.Tooltip("e_j:Q",     title="Coef. empleo", format=".4f"),
                ]
            ).properties(height=360)
            st.altair_chart(ch_e, use_container_width=True)

    st.markdown("#### Resumen")
    ci1, ci2, ci3, ci4 = st.columns(4)
    ci1.metric("VA total generado",  f"${res['delta_VA_total_pesos']:,.0f}")
    ci2.metric("Empleo total",       f"{res['delta_E_total']:,.0f} puestos")
    ci3.metric("Mult. ingreso",      f"{res['mult_ingreso']:.4f}",
               help="ΔVA / shock")
    ci4.metric("Empleo por Mmdp",    f"{res['mult_empleo']:.2f} puestos",
               help="Puestos por millón de pesos invertido")


# ═════════════ TAB 3 ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Tabla completa — sectores activos")

    df_tabla = df[df["activo"]].copy()
    df_tabla = df_tabla[[
        "scian", "nombre", "base_pesos",
        "delta_X_pesos", "delta_VA_pesos", "delta_E",
        "variacion_pct", "share_produccion", "v_j", "e_j"
    ]].rename(columns={
        "scian":           "SCIAN",
        "nombre":          "Sector",
        "base_pesos":      "Base (MXN)",
        "delta_X_pesos":   "ΔX Producción (MXN)",
        "delta_VA_pesos":  "ΔVA Ingreso (MXN)",
        "delta_E":         "ΔEmpleo (puestos)",
        "variacion_pct":   "Var. % base",
        "share_produccion":"Share prod.",
        "v_j":             "Coef. VA",
        "e_j":             "Coef. empleo",
    })

    styled = (
        df_tabla.style
        .format({
            "Base (MXN)":           "${:,.0f}",
            "ΔX Producción (MXN)":  "${:,.0f}",
            "ΔVA Ingreso (MXN)":    "${:,.0f}",
            "ΔEmpleo (puestos)":    "{:,.0f}",
            "Var. % base":          lambda x: f"{x:.2f}%" if pd.notnull(x) else "—",
            "Share prod.":          "{:.2%}",
            "Coef. VA":             "{:.4f}",
            "Coef. empleo":         "{:.4f}",
        })
        .background_gradient(cmap="Blues",   subset=["ΔX Producción (MXN)"])
        .background_gradient(cmap="Greens",  subset=["ΔVA Ingreso (MXN)"])
        .background_gradient(cmap="Purples", subset=["ΔEmpleo (puestos)"])
    )
    st.dataframe(styled, use_container_width=True, height=480)

    csv = df_tabla.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Descargar resultados (.csv)",
        data=csv,
        file_name=f"impacto_{estado_key}_{sector_scian}.csv",
        mime="text/csv",
    )
