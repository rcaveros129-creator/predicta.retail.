"""
Sección 6: CONTROL PREDICTIVO DE STOCK.
"""

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.prediccion import cargar_modelo
from utils import stock as calc_stock
from utils.componentes import (ESTADOS, GRIS_GRILLA, GRIS_TEXTO, etiqueta_estado, formato_soles, fuente,
                               tarjeta)

TOPE_GRAFICO = 120   # días máximos que se dibujan en el gráfico


def parametros():
    """Controles de la política de inventario. Se guardan para usarlos en otras secciones."""
    with st.expander("⚙️ Parámetros de la política de inventario", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Nivel de servicio", list(calc_stock.NIVELES_SERVICIO), index=1,
                     format_func=lambda n: f"{n} %", key="nivel_servicio",
                     help="Probabilidad de NO quedarse sin stock mientras llega el pedido. "
                          "Más alto = más stock de seguridad (y más dinero en inventario).")
        c2.slider("Días que debe cubrir cada pedido", 7, 60, 30, key="dias_a_cubrir",
                  help="Además del tiempo del proveedor, cuántos días de venta debe alcanzar cada compra.")
        c3.slider("Exceso si el stock alcanza para más de (días)", 45, 180, 90, step=15, key="dias_exceso")
    return (st.session_state.get("nivel_servicio", 95), st.session_state.get("dias_a_cubrir", 30),
            st.session_state.get("dias_exceso", 90))


@st.cache_data(show_spinner=False)
def analizar(productos, ventas, compras, _resultados, firma, nivel, dias_cubrir, dias_exceso):
    return calc_stock.calcular_stock(productos, ventas, compras, _resultados, nivel, dias_cubrir, dias_exceso)


def obtener_tabla(datos, nivel=95, dias_cubrir=30, dias_exceso=90):
    """Análisis de stock de todos los productos (lo usan también el Dashboard y Abastecimiento)."""
    resultados = cargar_modelo(datos)
    return analizar(datos["productos"], datos["ventas"], datos["compras"], resultados,
                    resultados["firma"], nivel, dias_cubrir, dias_exceso), resultados


def mostrar(datos, filtros):
    st.title("Control de stock")
    nivel, dias_cubrir, dias_exceso = parametros()
    try:
        tabla, res = obtener_tabla(datos, nivel, dias_cubrir, dias_exceso)
    except ValueError as e:
        st.error(str(e))
        return
    st.caption(f"Situación al {res['ultimo_dia_datos']:%d/%m/%Y} (último día con ventas)  ·  Usa la demanda "
               f"PREDICHA por el modelo  ·  {fuente(datos)}")

    visibles = tabla
    if filtros["categorias"]:
        visibles = visibles[visibles["categoria"].isin(filtros["categorias"])]
    if filtros["codigos"]:
        visibles = visibles[visibles["codigo"].isin(filtros["codigos"])]

    # ---------------- Tarjetas ----------------
    columnas = st.columns(4)
    notas = {
        "Stock crítico": "Pedir hoy: el proveedor no llegaría a tiempo",
        "Próximo a agotarse": "Llegó al punto de pedido",
        "Stock adecuado": "No requiere acción",
        "Exceso de inventario": "",
    }
    for col, estado in zip(columnas, ESTADOS):
        grupo = visibles[visibles["estado"] == estado]
        nota = notas[estado]
        if estado == "Exceso de inventario":
            nota = f"{formato_soles(grupo['valor_stock'].sum())} inmovilizados (a costo)"
        tarjeta(col, etiqueta_estado(estado), f"{len(grupo)} productos", nota)

    urgentes = visibles[visibles["necesita_pedido"]]
    con_pedido = visibles[(visibles["en_transito"] > 0)]
    if urgentes.empty:
        st.success(f"✅ Ningún producto está en estado crítico ni próximo a agotarse. "
                   f"{len(con_pedido)} productos tienen pedidos en camino (ver columna «En camino»).")

    # ---------------- Gráfico ----------------
    st.subheader("¿Cuántos días alcanza el stock?")
    st.caption(f"Barra = días hasta agotarse con el stock actual (según la demanda predicha). "
               f"Marca negra = días que tarda el proveedor. ◆ = día en que llega un pedido que ya está en camino. "
               f"Si la barra no llega a la marca negra y no hay un ◆ antes de que termine, el producto es "
               f"crítico. Se dibuja hasta {TOPE_GRAFICO} días.")
    dibujar_grafico(visibles, res["ultimo_dia_datos"])

    # ---------------- Tabla ----------------
    st.subheader("Detalle por producto")
    elegidos = st.multiselect("Mostrar estados", list(ESTADOS), default=list(ESTADOS),
                              format_func=etiqueta_estado)
    detalle = visibles[visibles["estado"].isin(elegidos)].copy()
    detalle["Estado"] = detalle["estado"].map(etiqueta_estado)
    detalle["dias_hasta_agotar"] = detalle["dias_hasta_agotar"].clip(upper=999)
    st.dataframe(
        detalle[["Estado", "codigo", "producto", "stock_actual", "en_transito", "llegada_pedido",
                 "promedio_vendido_diario", "demanda_diaria_estimada", "dias_hasta_agotar",
                 "fecha_agotamiento", "tiempo_reposicion_dias", "stock_seguridad", "punto_pedido",
                 "cantidad_reponer"]],
        hide_index=True, width="stretch", column_config={
            "codigo": "Código", "producto": "Producto",
            "stock_actual": st.column_config.NumberColumn("Stock", format="%d"),
            "en_transito": st.column_config.NumberColumn("En camino", format="%d"),
            "llegada_pedido": st.column_config.DateColumn("Llega el", format="DD/MM/YYYY"),
            "promedio_vendido_diario": st.column_config.NumberColumn(
                "Vendido/día (últ. 28 d)", format="%.2f", help="Promedio real de los últimos 28 días."),
            "demanda_diaria_estimada": st.column_config.NumberColumn(
                "Demanda/día estimada", format="%.2f", help="Promedio de la predicción para los próximos 30 días."),
            "dias_hasta_agotar": st.column_config.NumberColumn("Días hasta agotar", format="%.0f"),
            "fecha_agotamiento": st.column_config.DateColumn("Se agota aprox.", format="DD/MM/YYYY"),
            "tiempo_reposicion_dias": st.column_config.NumberColumn("Proveedor (días)", format="%d"),
            "stock_seguridad": st.column_config.NumberColumn("Stock seguridad", format="%d"),
            "punto_pedido": st.column_config.NumberColumn("Punto de pedido", format="%d"),
            "cantidad_reponer": st.column_config.NumberColumn(
                "Cantidad a reponer", format="%d",
                help="Solo para productos críticos o próximos a agotarse."),
        })
    st.caption("Los días hasta agotar se calculan solo con el stock actual. Los pedidos en camino se "
               "consideran para decidir el estado: si llegan antes de agotarse, el producto no es crítico.")

    # ---------------- Explicación de un producto ----------------
    st.subheader("Explicación paso a paso")
    codigo = st.selectbox("Elige un producto", visibles["codigo"],
                          format_func=dict(zip(visibles["codigo"], visibles["codigo"] + " · " +
                                               visibles["producto"])).get)
    if codigo:
        explicar(visibles.set_index("codigo").loc[codigo], nivel, dias_cubrir)

    # ---------------- Calculadora ----------------
    st.subheader("🧮 Prueba con tus propios números")
    calculadora(nivel, dias_cubrir, dias_exceso)

    with st.expander("ℹ️ ¿Cómo se calcula? (fórmulas y limitaciones)"):
        st.markdown(f"""
| Concepto | Cálculo |
|---|---|
| Demanda diaria estimada | Promedio de la predicción del modelo para los próximos 30 días |
| Días hasta agotar | Día en que la demanda predicha acumulada alcanza el stock actual |
| Demanda durante la reposición | Suma de la demanda predicha en los días que tarda el proveedor |
| Stock de seguridad | Z × desviación de las ventas diarias (últimos 90 días) × √(días del proveedor). Con nivel de servicio {nivel} %, Z = {calc_stock.NIVELES_SERVICIO[nivel]} |
| Punto de pedido | Demanda durante la reposición + stock de seguridad |
| Stock objetivo | Demanda predicha para (días del proveedor + {dias_cubrir} días) + stock de seguridad |
| Cantidad a reponer | Stock objetivo − (stock actual + unidades en camino) |

**Estados**
- ⛔ **Stock crítico**: se agotará antes de que pueda llegar un pedido hecho hoy (días hasta agotar < días del proveedor) y no hay un pedido en camino que llegue a tiempo.
- ⚠️ **Próximo a agotarse**: stock actual + en camino ≤ punto de pedido.
- 📦 **Exceso de inventario**: el stock (+ en camino) alcanza para más de {dias_exceso} días.
- ✅ **Stock adecuado**: ninguno de los anteriores.

**Limitaciones**
- Depende de la calidad de la predicción (ver su error en *Predicción de demanda*).
- El stock de seguridad supone que las ventas diarias varían de forma "normal"; en productos que se venden muy poco es solo aproximado.
- Supone que el proveedor cumple su plazo. Si suele atrasarse, aumenta los días de reposición en `productos.csv`.
- El stock actual es el registrado en `productos.csv`: si no se actualiza, las alertas serán incorrectas.
""")


def dibujar_grafico(tabla, hoy):
    datos = tabla.sort_values("dias_hasta_agotar", ascending=False).copy()
    datos["dias_grafico"] = datos["dias_hasta_agotar"].clip(upper=TOPE_GRAFICO)
    datos["etiqueta"] = datos["codigo"] + " · " + datos["producto"]
    fig = go.Figure()
    for estado, info in ESTADOS.items():
        parte = datos[datos["estado"] == estado]
        if parte.empty:
            continue
        fig.add_trace(go.Bar(
            x=parte["dias_grafico"], y=parte["etiqueta"], orientation="h", name=etiqueta_estado(estado),
            marker_color=info["color"],
            customdata=parte[["dias_hasta_agotar", "tiempo_reposicion_dias", "stock_actual", "en_transito"]],
            hovertemplate="<b>%{y}</b><br>Se agota en %{customdata[0]:.0f} días<br>"
                          "Proveedor: %{customdata[1]} días<br>Stock: %{customdata[2]}"
                          " · En camino: %{customdata[3]}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=datos["tiempo_reposicion_dias"], y=datos["etiqueta"], mode="markers", name="Días del proveedor",
        marker=dict(symbol="line-ns", size=14, line=dict(width=3, color="#0b0b0b")),
        hovertemplate="Proveedor: %{x} días<extra></extra>"))
    camino = datos[datos["en_transito"] > 0].copy()
    if not camino.empty:
        camino["dias_llegada"] = (pd.to_datetime(camino["llegada_pedido"]) - pd.Timestamp(hoy)).dt.days
        fig.add_trace(go.Scatter(
            x=camino["dias_llegada"], y=camino["etiqueta"], mode="markers", name="Llega pedido en camino",
            marker=dict(symbol="diamond", size=11, color="#2a78d6", line=dict(width=2, color="white")),
            customdata=camino[["en_transito"]],
            hovertemplate="Llegan %{customdata[0]} unidades en %{x} días<extra></extra>"))
    fig.update_layout(template="plotly_white", height=max(300, 24 * len(datos) + 80), barcornerradius=4,
                      margin=dict(l=10, r=10, t=40, b=10), font=dict(color=GRIS_TEXTO, size=12),
                      hoverlabel=dict(bgcolor="white"), legend=dict(orientation="h", y=1.04, x=0),
                      yaxis=dict(categoryorder="array", categoryarray=list(datos["etiqueta"])))
    fig.update_xaxes(title="Días", gridcolor=GRIS_GRILLA, range=[0, TOPE_GRAFICO + 5])
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, width="stretch")


def explicar(f, nivel, dias_cubrir):
    dias = f["dias_hasta_agotar"]
    dias_txt = f"{dias:.1f} días" if math.isfinite(dias) else "nunca (no se esperan ventas)"
    fecha = f"{f['fecha_agotamiento']:%d/%m/%Y}" if pd.notna(f["fecha_agotamiento"]) else "—"
    camino = (f"Hay **{f['en_transito']} unidades en camino** que llegan el **{f['llegada_pedido']:%d/%m/%Y}**."
              if f["en_transito"] else "No hay pedidos en camino.")
    st.markdown(f"""
1. **Stock actual:** {f['stock_actual']} unidades. {camino}
2. **Demanda estimada:** {f['demanda_diaria_estimada']:.2f} unidades/día (el modelo); en los últimos 28 días
   se vendieron {f['promedio_vendido_diario']:.2f}/día en promedio.
3. **Se agota en:** {dias_txt} → fecha aproximada **{fecha}**.
4. **El proveedor tarda:** {f['tiempo_reposicion_dias']} días. Durante ese tiempo se espera vender
   {f['demanda_en_reposicion']:.1f} unidades.
5. **Stock de seguridad ({nivel} %):** {f['stock_seguridad']} unidades → **punto de pedido:** {f['punto_pedido']} unidades.
6. **Resultado:** {etiqueta_estado(f['estado'])}. {f['motivo']}
""")
    if f["necesita_pedido"]:
        st.markdown(f"7. **Cantidad a reponer:** {f['cantidad_reponer']} unidades, para cubrir "
                    f"{f['tiempo_reposicion_dias']} días del proveedor + {dias_cubrir} días de venta + stock de seguridad.")


def calculadora(nivel, dias_cubrir, dias_exceso):
    st.caption("Escribe un caso con venta diaria constante. Usa exactamente las mismas reglas que el análisis.")
    c1, c2, c3, c4, c5 = st.columns(5)
    stock = c1.number_input("Stock actual", min_value=0, value=20, step=1)
    venta = c2.number_input("Venta por día", min_value=0.0, value=4.0, step=0.5)
    proveedor = c3.number_input("Días del proveedor", min_value=1, value=7, step=1)
    camino = c4.number_input("Unidades en camino", min_value=0, value=0, step=1)
    llegada = c5.number_input("Llegan en (días)", min_value=1, value=3, step=1, disabled=camino == 0)
    f = calc_stock.evaluar_caso(stock, venta, proveedor, camino, llegada, nivel, dias_cubrir, dias_exceso)
    dias = f["dias_hasta_agotar"]
    dias_txt = f"{dias:.1f} días" if math.isfinite(dias) else "no se agota"
    mensaje = (f"**{etiqueta_estado(f['estado'])}** · Inventario para **{dias_txt}** "
               f"({stock} ÷ {venta:g} por día). {f['motivo']}")
    if f["necesita_pedido"]:
        mensaje += f" Reponer **{f['cantidad_reponer']} unidades**."
    {"Stock crítico": st.error, "Próximo a agotarse": st.warning,
     "Stock adecuado": st.success, "Exceso de inventario": st.info}[f["estado"]](mensaje)
