"""
Sección 7: ABASTECIMIENTO INTELIGENTE (recomendaciones de compra).
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.inventario import obtener_tabla
from utils.abastecimiento import ACCIONES, recomendar
from utils.componentes import estilo_grafico, formato_soles, fuente, tarjeta

ICONO_RIESGO = {"Alto": "🔴 Alto", "Medio": "🟠 Medio", "Bajo": "🟢 Bajo"}


def mostrar(datos, filtros):
    st.title("Abastecimiento inteligente")
    nivel = st.session_state.get("nivel_servicio", 95)
    dias_cubrir = st.session_state.get("dias_a_cubrir", 30)
    dias_exceso = st.session_state.get("dias_exceso", 90)
    try:
        stock, res = obtener_tabla(datos, nivel, dias_cubrir, dias_exceso)
    except ValueError as e:
        st.error(str(e))
        return
    plan = recomendar(stock, datos["productos"], res, dias_cubrir, dias_exceso)
    hoy = pd.Timestamp(res["ultimo_dia_datos"])

    st.caption(f"Recomendaciones calculadas con la demanda PREDICHA por el modelo y el stock al "
               f"{hoy:%d/%m/%Y}  ·  Nivel de servicio {nivel} %, cada pedido cubre {dias_cubrir} días "
               f"(se cambian en ⚙️ Parámetros de *Control de stock*)  ·  {fuente(datos)}")

    if filtros["categorias"]:
        plan = plan[plan["categoria"].isin(filtros["categorias"])]
    if filtros["codigos"]:
        plan = plan[plan["codigo"].isin(filtros["codigos"])]

    dias = st.radio("Planificar los pedidos de los próximos", [7, 15, 30, 60], index=2, horizontal=True,
                    format_func=lambda d: f"{d} días")
    limite = hoy + pd.Timedelta(days=dias)
    pedidos = plan[plan["fecha_pedido"].notna() & (plan["fecha_pedido"] <= limite)]
    exceso = plan[plan["accion"] == "No comprar: exceso"]

    # ---------------- Tarjetas ----------------
    c1, c2, c3, c4 = st.columns(4)
    hoy_pedir = pedidos[pedidos["accion"] == "Pedir hoy"]
    tarjeta(c1, "🛒 Pedir hoy", f"{len(hoy_pedir)} productos",
            formato_soles(hoy_pedir["costo"].sum()) if len(hoy_pedir) else "Nada urgente")
    tarjeta(c2, f"📅 Pedidos en {dias} días", f"{len(pedidos)} productos",
            f"Inversión estimada: {formato_soles(pedidos['costo'].sum())} (a precio de compra)")
    tarjeta(c3, "Unidades a comprar", f"{int(pedidos['cantidad'].sum()):,}",
            f"En {pedidos['proveedor'].nunique()} proveedores")
    tarjeta(c4, "⏸️ No comprar (exceso)", f"{len(exceso)} productos",
            f"{formato_soles(exceso['valor_exceso'].sum())} en unidades que sobran")

    # ---------------- Plan de compras ----------------
    st.subheader(f"Plan de compras: próximos {dias} días")
    if pedidos.empty:
        st.success(f"✅ No hay pedidos que hacer en los próximos {dias} días.")
    else:
        vista = pedidos.copy()
        vista["Acción"] = vista["accion"].map(lambda a: f"{ACCIONES[a]} {a}")
        vista["Riesgo"] = vista["riesgo_quiebre"].map(ICONO_RIESGO)
        st.dataframe(
            vista[["Acción", "fecha_pedido", "codigo", "producto", "proveedor", "cantidad",
                   "fecha_llegada", "costo", "Riesgo", "motivo"]],
            hide_index=True, width="stretch", column_config={
                "fecha_pedido": st.column_config.DateColumn("Pedir el", format="DD/MM/YYYY"),
                "codigo": "Código", "producto": "Producto", "proveedor": "Proveedor",
                "cantidad": st.column_config.NumberColumn("Cantidad", format="%d"),
                "fecha_llegada": st.column_config.DateColumn("Llega aprox.", format="DD/MM/YYYY"),
                "costo": st.column_config.NumberColumn("Costo (S/)", format="S/ %,.0f"),
                "motivo": st.column_config.TextColumn("Motivo", width="large")})
        st.download_button("⬇️ Descargar plan de compras (CSV)",
                           pedidos.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"plan_compras_{hoy:%Y%m%d}_{dias}dias.csv", mime="text/csv")

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown("**Inversión por semana**")
            semanal = pedidos.copy()
            semanal["semana"] = ((semanal["fecha_pedido"] - hoy).dt.days - 1) // 7
            semanal = semanal.groupby("semana").agg(costo=("costo", "sum"), productos=("codigo", "count"))
            semanal = semanal.reindex(range(int(semanal.index.max()) + 1), fill_value=0).reset_index()
            semanal["etiqueta"] = [f"Sem. {s + 1}<br>{hoy + pd.Timedelta(days=7 * s + 1):%d/%m}"
                                   for s in semanal["semana"]]
            fig = px.bar(semanal, x="etiqueta", y="costo", custom_data=["productos"])
            fig.update_traces(hovertemplate="%{x}<br>S/ %{y:,.0f} · %{customdata[0]} pedidos<extra></extra>")
            fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f")
            st.plotly_chart(estilo_grafico(fig, alto=280), width="stretch")
        with col_b:
            st.markdown("**Agrupado por proveedor**")
            proveedores = (pedidos.groupby("proveedor")
                           .agg(productos=("codigo", "count"), unidades=("cantidad", "sum"),
                                costo=("costo", "sum"), primer_pedido=("fecha_pedido", "min"))
                           .sort_values("primer_pedido").reset_index())
            st.dataframe(proveedores, hide_index=True, width="stretch", column_config={
                "proveedor": "Proveedor",
                "productos": st.column_config.NumberColumn("Productos", format="%d"),
                "unidades": st.column_config.NumberColumn("Unidades", format="%d"),
                "costo": st.column_config.NumberColumn("Costo", format="S/ %,.0f"),
                "primer_pedido": st.column_config.DateColumn("Primer pedido", format="DD/MM")})
            st.caption("Agrupar pedidos del mismo proveedor en una fecha puede ahorrar costos de envío.")

    # ---------------- Riesgo de quedarse sin stock ----------------
    st.subheader("Productos con riesgo de quedarse sin stock")
    riesgo = plan[plan["riesgo_quiebre"].isin(["Alto", "Medio"])].copy()
    if riesgo.empty:
        st.success("✅ Ningún producto tiene riesgo alto o medio de quedarse sin stock.")
    else:
        riesgo["Riesgo"] = riesgo["riesgo_quiebre"].map(ICONO_RIESGO)
        st.dataframe(riesgo[["Riesgo", "codigo", "producto", "stock_actual", "en_transito",
                             "fecha_pedido", "cantidad"]],
                     hide_index=True, width="stretch", column_config={
                         "codigo": "Código", "producto": "Producto",
                         "stock_actual": st.column_config.NumberColumn("Stock", format="%d"),
                         "en_transito": st.column_config.NumberColumn("En camino", format="%d"),
                         "fecha_pedido": st.column_config.DateColumn("Pedir el", format="DD/MM/YYYY"),
                         "cantidad": st.column_config.NumberColumn("Cantidad", format="%d")})
        st.caption("🔴 Alto: stock crítico. 🟠 Medio: hay que pedir en los próximos 7 días; "
                   "si el pedido se retrasa, el producto puede agotarse.")

    # ---------------- Exceso ----------------
    st.subheader("Productos con exceso de inventario")
    if exceso.empty:
        st.success("✅ Ningún producto tiene exceso de inventario.")
    else:
        st.dataframe(exceso[["codigo", "producto", "stock_actual", "en_transito", "exceso_unidades",
                             "valor_exceso", "motivo"]],
                     hide_index=True, width="stretch", column_config={
                         "codigo": "Código", "producto": "Producto",
                         "stock_actual": st.column_config.NumberColumn("Stock", format="%d"),
                         "en_transito": st.column_config.NumberColumn("En camino", format="%d"),
                         "exceso_unidades": st.column_config.NumberColumn("Unidades que sobran", format="%d"),
                         "valor_exceso": st.column_config.NumberColumn("Valor (S/)", format="S/ %,.0f"),
                         "motivo": st.column_config.TextColumn("Recomendación", width="large")})

    # ---------------- Todos los productos ----------------
    with st.expander("📋 Ver la recomendación de todos los productos"):
        todo = plan.copy()
        todo["Acción"] = todo["accion"].map(lambda a: f"{ACCIONES[a]} {a}")
        st.dataframe(todo[["Acción", "codigo", "producto", "fecha_pedido", "cantidad", "costo", "motivo"]],
                     hide_index=True, width="stretch", column_config={
                         "codigo": "Código", "producto": "Producto",
                         "fecha_pedido": st.column_config.DateColumn("Pedir el", format="DD/MM/YYYY"),
                         "cantidad": st.column_config.NumberColumn("Cantidad", format="%d"),
                         "costo": st.column_config.NumberColumn("Costo", format="S/ %,.0f"),
                         "motivo": st.column_config.TextColumn("Motivo", width="large")})

    with st.expander("ℹ️ ¿Cómo se generan las recomendaciones?"):
        st.markdown(f"""
1. **Proyección del inventario.** Desde el stock actual + lo que está en camino, se resta día a día
   la demanda **predicha** por el modelo.
2. **¿Cuándo pedir?** El día en que esa proyección llega al **punto de pedido** (venta esperada mientras
   llega el proveedor + stock de seguridad). Si ya está por debajo, la acción es *Pedir hoy*.
3. **¿Cuánto pedir?** Lo necesario para cubrir, desde la fecha del pedido, los días del proveedor +
   **{dias_cubrir} días** de venta + el stock de seguridad, menos lo que quede en ese momento.
4. **¿Qué no comprar?** Productos cuyo stock alcanza para más de **{dias_exceso} días**. Se calcula
   cuántas unidades sobran frente a la venta prevista para ese periodo.
5. **Riesgo de quedarse sin stock:** Alto si está en estado crítico; Medio si hay que pedir en los
   próximos 7 días; Bajo en los demás casos.

**Limitaciones:** no considera pedidos mínimos del proveedor, descuentos por volumen, presupuesto
disponible ni espacio en la tienda. Las cantidades son una **guía**: el dueño decide. Se planifica un
solo pedido por producto dentro de los 90 días que cubre la predicción.
""")
