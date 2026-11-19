"""
Sección 1: DASHBOARD PRINCIPAL.

Muestra los indicadores clave y gráficos generales de la tienda.
"""

import plotly.express as px
import streamlit as st

from modules.inventario import obtener_tabla
from utils import calculations as calc
from utils.componentes import etiqueta_estado, estilo_grafico, formato_soles, fuente, tarjeta_producto

def mostrar(datos, filtros):
    productos = calc.filtrar_productos(datos["productos"], filtros["categorias"], filtros["codigos"])
    ventas = calc.filtrar_ventas(datos["ventas"], filtros["fecha_inicio"], filtros["fecha_fin"],
                                 filtros["categorias"], filtros["codigos"])

    st.title("Panel de control")
    st.caption(f"Periodo: {filtros['fecha_inicio']:%d/%m/%Y} al {filtros['fecha_fin']:%d/%m/%Y}  ·  "
               f"{fuente(datos)}")

    if ventas.empty:
        st.warning("No hay ventas con los filtros elegidos. Cambia las fechas o categorías.")
        return

    # ---------------- Indicadores (tarjetas) ----------------
    actual, anterior, variacion, ini_mes, fin_mes = calc.ventas_del_mes(
        calc.filtrar_ventas(datos["ventas"], "1900-01-01", filtros["fecha_fin"],
                            filtros["categorias"], filtros["codigos"]),
        filtros["fecha_fin"])
    costo_inv, venta_inv = calc.valor_inventario(productos)
    ranking = calc.ranking_productos(ventas, productos)
    try:
        stock, _ = obtener_tabla(datos, st.session_state.get("nivel_servicio", 95),
                                 st.session_state.get("dias_a_cubrir", 30),
                                 st.session_state.get("dias_exceso", 90))
        stock = stock[stock["codigo"].isin(productos["codigo"])]
    except ValueError:
        stock = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Ventas del mes ({calc.MESES[fin_mes.month - 1]})", formato_soles(actual),
              delta=f"{variacion:+.1f} % vs. mes anterior" if variacion is not None else None,
              help=f"Del {ini_mes:%d/%m} al {fin_mes:%d/%m}. Se compara con los mismos días "
                   f"del mes anterior ({formato_soles(anterior)}).",
              border=True)
    c2.metric("Unidades vendidas", f"{int(ventas['cantidad'].sum()):,}",
              help="Total de prendas vendidas en el periodo filtrado.", border=True)
    c3.metric("Productos registrados", len(productos), border=True)
    c4.metric("Valor del inventario", formato_soles(costo_inv),
              help=f"Stock actual × precio de compra. A precio de venta serían {formato_soles(venta_inv)}.",
              border=True)

    c5, c6, c7, c8 = st.columns(4)
    mas = ranking.iloc[0]
    menos = ranking.iloc[-1]
    tarjeta_producto(c5, "Más vendido", mas)
    tarjeta_producto(c6, "Menos vendido", menos)
    if stock is not None:
        c7.metric(etiqueta_estado("Stock crítico"), int((stock["estado"] == "Stock crítico").sum()),
                  help="Se agotarán antes de que pueda llegar un pedido hecho hoy.", border=True)
        c8.metric(etiqueta_estado("Próximo a agotarse"), int((stock["estado"] == "Próximo a agotarse").sum()),
                  help="Llegaron a su punto de pedido.", border=True)

    # ---------------- Gráficos ----------------
    st.subheader("Ventas por mes")
    mensual = calc.ventas_por_mes(ventas)
    fig = px.bar(mensual, x="mes", y="ventas_soles",
                 custom_data=["unidades", "dias_con_datos"])
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Ventas: S/ %{y:,.0f}<br>"
                                    "Unidades: %{customdata[0]:,}<br>Días con ventas: %{customdata[1]}<extra></extra>")
    fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f")
    st.plotly_chart(estilo_grafico(fig), width="stretch")
    incompletos = mensual[mensual["dias_con_datos"] < 25]["mes"].tolist()
    if incompletos:
        st.caption(f"Meses incompletos (con menos días de datos): {', '.join(incompletos)}. "
                   "Pasa el cursor sobre cada barra para ver cuántos días tiene.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Ventas por categoría")
        categorias = calc.ventas_por_categoria(ventas).sort_values("ventas_soles")
        fig = px.bar(categorias, x="ventas_soles", y="categoria", orientation="h",
                     custom_data=["unidades"])
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Ventas: S/ %{x:,.0f}<br>"
                                        "Unidades: %{customdata[0]:,}<extra></extra>")
        fig.update_xaxes(tickprefix="S/ ", tickformat=",.0f", gridcolor="#ececea", showgrid=True)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(estilo_grafico(fig, alto=400), width="stretch")
    with col_b:
        st.subheader("Unidades promedio por día")
        semana = calc.ventas_por_dia_semana(ventas)
        fig = px.bar(semana, x="nombre_dia", y="unidades_promedio")
        fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:.1f} unidades en promedio<extra></extra>")
        st.plotly_chart(estilo_grafico(fig, alto=400), width="stretch")

    # ---------------- Rankings ----------------
    st.subheader("Productos más y menos vendidos")
    st.caption("Ordenados por unidades vendidas en el periodo. La clasificación por rotación con "
               "Machine Learning se agregará en la Etapa 4.")
    columnas = {"codigo": "Código", "producto": "Producto",
                "unidades": "Unidades", "ventas_soles": "Ventas (S/)"}
    formato = {"Unidades": st.column_config.NumberColumn(format="%d"),
               "Ventas (S/)": st.column_config.NumberColumn(format="S/ %,.0f")}
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Top 5 más vendidos**")
        st.dataframe(ranking.head(5)[list(columnas)].rename(columns=columnas),
                     hide_index=True, width="stretch", column_config=formato)
    with col_b:
        st.markdown("**Top 5 menos vendidos**")
        st.dataframe(ranking.tail(5).iloc[::-1][list(columnas)].rename(columns=columnas),
                     hide_index=True, width="stretch", column_config=formato)

    # ---------------- Alertas de stock ----------------
    st.subheader("Alertas de abastecimiento")
    if stock is None:
        st.info("Las alertas predictivas necesitan al menos 6 meses de ventas.")
        return
    alertas = stock[stock["estado"] != "Stock adecuado"].copy()
    if alertas.empty:
        st.success("✅ Todos los productos tienen stock adecuado.")
        return
    st.caption("Alertas predictivas: usan la demanda estimada por el modelo y el tiempo de cada proveedor. "
               "Más detalle en la sección **📦 Control de stock**.")
    alertas["Estado"] = alertas["estado"].map(etiqueta_estado)
    alertas["dias_hasta_agotar"] = alertas["dias_hasta_agotar"].clip(upper=999)
    st.dataframe(
        alertas[["Estado", "codigo", "producto", "stock_actual", "en_transito", "dias_hasta_agotar", "motivo"]],
        hide_index=True, width="stretch", column_config={
            "codigo": "Código", "producto": "Producto",
            "stock_actual": st.column_config.NumberColumn("Stock", format="%d"),
            "en_transito": st.column_config.NumberColumn("En camino", format="%d"),
            "dias_hasta_agotar": st.column_config.NumberColumn("Días hasta agotar", format="%.0f"),
            "motivo": "Motivo"})
