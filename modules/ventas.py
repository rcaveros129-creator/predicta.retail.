"""
Sección 3: VENTAS Y TEMPORADAS (análisis de estacionalidad).
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import calculations as calc
from utils import temporadas as tmp
from utils.componentes import (AZUL, GRIS_GRILLA, GRIS_TEXTO, NARANJA, estilo_grafico, formato_soles, fuente,
                               tarjeta)

ICONO_CONCLUSION = {"Aumenta las ventas": "🔵 Aumenta las ventas",
                    "Disminuye las ventas": "🟠 Disminuye las ventas",
                    "Sin efecto comprobado": "⚪ Sin efecto comprobado"}


@st.cache_data(show_spinner="Analizando temporadas...")
def temporadas_cacheadas(ventas):
    return tmp.analizar_temporadas(ventas)


@st.cache_data(show_spinner="Analizando estaciones...")
def estaciones_cacheadas(ventas):
    return tmp.analizar_estaciones(ventas)


def mostrar(datos, filtros):
    st.title("Ventas y temporadas")
    ventas = calc.filtrar_ventas(datos["ventas"], filtros["fecha_inicio"], filtros["fecha_fin"],
                                 filtros["categorias"], filtros["codigos"])
    st.caption(f"Periodo: {filtros['fecha_inicio']:%d/%m/%Y} al {filtros['fecha_fin']:%d/%m/%Y}  ·  "
               f"{fuente(datos)}  ·  Los filtros del menú se aplican a toda la página")
    if ventas["fecha"].nunique() < 60:
        st.warning("Elige un periodo de al menos 2 meses para analizar temporadas.")
        return

    mensual = tmp.promedio_diario_por_mes(ventas)
    semana = tmp.promedio_por_dia_semana(ventas)
    completos = mensual[mensual["dias"] >= 20]
    mejor, peor = completos.loc[completos["soles_dia"].idxmax()], completos.loc[completos["soles_dia"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    tarjeta(c1, "📈 Mes con más ventas", mejor["mes"], f"{formato_soles(mejor['soles_dia'])} por día en promedio")
    tarjeta(c2, "📉 Mes con menos ventas", peor["mes"], f"{formato_soles(peor['soles_dia'])} por día en promedio")
    dia_max = semana.loc[semana["unidades_promedio"].idxmax()]
    dia_min = semana.loc[semana["unidades_promedio"].idxmin()]
    tarjeta(c3, "Día con más demanda", dia_max["dia"], f"{dia_max['unidades_promedio']:.1f} unidades en promedio")
    tarjeta(c4, "Día con menos demanda", dia_min["dia"], f"{dia_min['unidades_promedio']:.1f} unidades en promedio")
    st.caption("Los meses se comparan por **venta diaria promedio**, para que los meses incompletos no "
               "parezcan peores. Se excluyen de las tarjetas los meses con menos de 20 días de datos.")

    # ---------------- Meses y días ----------------
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader("Venta diaria promedio por mes")
        fig = px.bar(mensual, x="mes", y="soles_dia", custom_data=["soles_total", "dias"])
        fig.update_traces(hovertemplate="<b>%{x}</b><br>S/ %{y:,.0f} por día<br>Total: S/ %{customdata[0]:,.0f}"
                                        " en %{customdata[1]} días<extra></extra>")
        fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f")
        st.plotly_chart(estilo_grafico(fig, alto=320), width="stretch")
    with col_b:
        st.subheader("Unidades por día de la semana")
        fig = px.bar(semana, x="dia", y="unidades_promedio")
        fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:.1f} unidades<extra></extra>")
        st.plotly_chart(estilo_grafico(fig, alto=320), width="stretch")

    # ---------------- Mapa de calor ----------------
    st.subheader("¿En qué meses se vende más cada categoría?")
    st.caption("Índice estacional: venta diaria de la categoría en el mes ÷ su venta diaria del año. "
               "**1.0 = mes normal**; azul = se vende más de lo normal; naranja = menos.")
    indice = tmp.indice_categoria_mes(ventas)
    fig = go.Figure(go.Heatmap(
        z=indice.values, x=list(indice.columns), y=list(indice.index), zmid=1, zmin=0, zmax=2,
        colorscale=[[0, NARANJA], [0.5, "#f0efec"], [1, AZUL]], xgap=2, ygap=2,
        text=indice.round(1).values, texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="<b>%{y}</b> en %{x}<br>Índice: %{z:.2f}<extra></extra>",
        colorbar=dict(title="Índice", tickvals=[0, 0.5, 1, 1.5, 2])))
    fig.update_layout(template="plotly_white", height=max(300, 32 * len(indice) + 80),
                      margin=dict(l=10, r=10, t=10, b=10), font=dict(color=GRIS_TEXTO, size=12))
    st.plotly_chart(fig, width="stretch")

    # ---------------- Temporadas comerciales ----------------
    st.subheader("¿Las temporadas comerciales realmente aumentan las ventas?")
    st.markdown("Cada temporada se compara con los **días normales de las 4 semanas anteriores y posteriores**, "
                "ajustando por día de la semana. Una **prueba de permutación** indica si la diferencia es real "
                "(p < 0.05) o podría ser casualidad. **No se asume ningún efecto de antemano.**")
    resultado = temporadas_cacheadas(ventas)
    if resultado.empty:
        st.info("El periodo elegido no contiene temporadas completas.")
    else:
        mostrar_resultado_temporadas(resultado)

    # ---------------- Verano / invierno ----------------
    st.subheader("Cambio de estación climática")
    estaciones = estaciones_cacheadas(ventas)
    if estaciones.empty:
        st.info("El periodo elegido no permite comparar estaciones.")
    else:
        vista = estaciones.copy()
        vista["Conclusión"] = vista["conclusion"].map(ICONO_CONCLUSION)
        estacion = st.radio("Estación", list(tmp.ESTACIONES), horizontal=True, label_visibility="collapsed")
        vista = vista[vista["estacion"] == estacion].sort_values("diferencia_pct", ascending=False)
        st.dataframe(vista[["categoria", "unidades_dia_estacion", "unidades_dia_resto", "diferencia_pct",
                            "p_valor", "Conclusión"]], hide_index=True, width="stretch", column_config={
            "categoria": "Categoría",
            "unidades_dia_estacion": st.column_config.NumberColumn("Unid./día en la estación", format="%.2f"),
            "unidades_dia_resto": st.column_config.NumberColumn("Unid./día resto del año", format="%.2f"),
            "diferencia_pct": st.column_config.NumberColumn("Diferencia", format="%+.0f %%"),
            "p_valor": st.column_config.NumberColumn("p-valor", format="%.3f")})
        st.caption("Se compara por semanas (no por días) para no exagerar la seguridad del resultado.")

    # ---------------- Productos por temporada ----------------
    st.subheader("Productos más solicitados por temporada")
    opciones = list(tmp.TEMPORADAS) + list(tmp.ESTACIONES)
    elegida = st.selectbox("Temporada", opciones, index=opciones.index("Navidad"))
    top = tmp.productos_en_temporada(ventas, elegida)
    if top.empty:
        st.info("Esa temporada no está dentro del periodo elegido.")
    else:
        st.dataframe(top, hide_index=True, width="stretch", column_config={
            "codigo": "Código", "producto": "Producto", "categoria": "Categoría",
            "unidades_temporada": st.column_config.NumberColumn("Unidades vendidas en la temporada", format="%d"),
            "dia_en_temporada": st.column_config.NumberColumn("Unid./día en temporada", format="%.2f"),
            "dia_fuera": st.column_config.NumberColumn("Unid./día fuera", format="%.2f"),
            "veces_normal": st.column_config.NumberColumn("Veces su venta normal", format="%.1f×")})
        st.caption("Ordenados por cuántas veces se multiplica su venta en esa temporada. Ojo: si la temporada "
                   "coincide con una estación (ej. Día del Padre en invierno), el aumento puede deberse a la "
                   "estación y no a la fecha. Para eso sirve la prueba de la tabla de temporadas.")

    # ---------------- Cambios de comportamiento ----------------
    st.subheader("Cambios en el comportamiento de compra")
    tabla, total, (inicio, mitad, fin) = tmp.comparar_semestres(ventas)
    st.markdown(f"Primera mitad del periodo (**{inicio:%d/%m/%Y} – {mitad:%d/%m/%Y}**) vs. segunda mitad "
                f"(**{mitad:%d/%m/%Y} – {fin:%d/%m/%Y}**). Venta diaria total: **{total:+.1f} %**.")
    fig = px.bar(tabla.sort_values("cambio_pct"), x="cambio_pct", y="categoria", orientation="h",
                 custom_data=["primera_mitad", "segunda_mitad"])
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Cambio: %{x:+.0f} %<br>%{customdata[0]:.2f} → "
                                    "%{customdata[1]:.2f} unid./día<extra></extra>")
    fig = estilo_grafico(fig, alto=380)
    fig.update_traces(marker_color=[AZUL if v >= 0 else NARANJA for v in tabla.sort_values("cambio_pct")["cambio_pct"]])
    fig.update_xaxes(ticksuffix=" %", showgrid=True, gridcolor=GRIS_GRILLA, zeroline=True, zerolinecolor="#a8a7a2")
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Azul = la categoría vende más en la segunda mitad; naranja = vende menos. Con solo 12 meses, "
               "cada mitad cae en estaciones distintas, así que buena parte del cambio es estacional y no "
               "necesariamente una tendencia. Con 2 años se podrá comparar el mismo periodo de cada año.")

    with st.expander("ℹ️ ¿Cómo se interpretan estos resultados? (método y limitaciones)"):
        st.markdown(f"""
**Prueba de permutación (temporadas comerciales)**
1. Se calcula la venta diaria promedio en la temporada y en los días normales cercanos
   ({tmp.DIAS_COMPARACION} días antes y después, sin contar otras temporadas).
2. Cada día se ajusta por su día de la semana (los sábados venden más), para comparar de forma justa.
3. Se mezclan al azar los días {tmp.PERMUTACIONES:,} veces y se cuenta cuántas veces aparece por
   casualidad una diferencia igual o mayor a la real. Esa proporción es el **p-valor**.
4. Si p < 0.05 (menos del 5 % de las veces), se concluye que la temporada **sí** cambia las ventas.

**Cómo leerlo:** "Sin efecto comprobado" no significa que la temporada no influya; significa que con
estos datos **no se puede demostrar**. Una temporada puede afectar solo a algunas categorías (por
ejemplo, el Día de la Madre a vestidos y blusas): usa el filtro de **Categorías** del menú para verlo.

**Limitaciones**
- Con 12 meses, cada temporada aparece **una sola vez**: un año atípico puede engañar.
- Las fechas de cada temporada son fijas (ej. Navidad del 10 al 24 de diciembre).
- Los días seguidos se parecen entre sí, lo que puede hacer la prueba algo optimista.
""" + ("- Son datos **ficticios**: en la simulación, el Día del Padre se configuró **sin** efecto a "
       "propósito, para comprobar que el análisis no inventa resultados.\n" if datos["ficticios"] else ""))


def mostrar_resultado_temporadas(resultado):
    vista = resultado.copy()
    vista["Conclusión"] = vista["conclusion"].map(ICONO_CONCLUSION)
    vista["Fechas"] = vista["inicio"].dt.strftime("%d/%m/%Y") + " – " + vista["fin"].dt.strftime("%d/%m/%Y")

    colores = {"Aumenta las ventas": AZUL, "Disminuye las ventas": NARANJA, "Sin efecto comprobado": "#a8a7a2"}
    fig = go.Figure(go.Bar(
        x=vista["diferencia_pct"], y=vista["temporada"], orientation="h",
        marker_color=[colores[c] for c in vista["conclusion"]],
        text=[f"{d:+.0f} %  {'(p=' + format(p, '.3f') + ')'}" for d, p in zip(vista["diferencia_pct"], vista["p_valor"])],
        textposition="outside", customdata=vista[["Conclusión", "promedio_temporada", "promedio_normal"]],
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>%{customdata[1]:.1f} vs %{customdata[2]:.1f} "
                      "unid./día<extra></extra>"))
    fig.update_layout(template="plotly_white", height=60 * len(vista) + 60, barcornerradius=4,
                      margin=dict(l=10, r=60, t=10, b=10), font=dict(color=GRIS_TEXTO, size=13),
                      hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(ticksuffix=" %", gridcolor=GRIS_GRILLA, zeroline=True, zerolinecolor="#a8a7a2",
                     range=[min(0, vista["diferencia_pct"].min()) - 20, vista["diferencia_pct"].max() + 35])
    fig.update_yaxes(autorange="reversed", showgrid=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("🔵 Azul: aumento comprobado · 🟠 Naranja: disminución comprobada · ⚪ Gris: sin efecto comprobado.")

    st.dataframe(vista[["temporada", "Fechas", "promedio_temporada", "promedio_normal", "diferencia_pct",
                        "p_valor", "Conclusión"]], hide_index=True, width="stretch", column_config={
        "temporada": "Temporada",
        "promedio_temporada": st.column_config.NumberColumn("Unid./día en temporada", format="%.1f"),
        "promedio_normal": st.column_config.NumberColumn("Unid./día normales", format="%.1f"),
        "diferencia_pct": st.column_config.NumberColumn("Diferencia", format="%+.0f %%",
                                                        help="Ya ajustada por día de la semana."),
        "p_valor": st.column_config.NumberColumn("p-valor", format="%.3f",
                                                 help="Menor a 0.05 = la diferencia no parece casualidad.")})
