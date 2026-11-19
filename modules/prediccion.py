"""
Sección 5: PREDICCIÓN DE DEMANDA.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.componentes import fuente, tarjeta
from utils import ml_demanda as ml

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLOR_MODELO = {"random_forest": AZUL, "promedio_28": NARANJA, "promedio_7": AQUA}


@st.cache_resource(show_spinner=False)
def obtener_resultados(ventas, productos, compras, forzar=False):
    return ml.ejecutar_todo(ventas, productos, compras, forzar=forzar)


def cargar_modelo(datos):
    """Carga el modelo guardado o lo entrena (la primera vez tarda unos segundos)."""
    con_archivo = ml.RUTA_MODELO.exists()
    mensaje = ("Cargando el modelo guardado..." if con_archivo else
               "Entrenando y evaluando los modelos por primera vez. Esto puede tardar hasta 1 minuto...")
    with st.spinner(mensaje):
        return obtener_resultados(datos["ventas"], datos["productos"], datos["compras"])


def mostrar(datos, filtros):
    st.title("Predicción de demanda")
    try:
        res = cargar_modelo(datos)
    except ValueError as e:
        st.error(str(e))
        return

    st.caption(f"Predicciones GENERADAS POR EL MODELO  ·  {fuente(datos)}  ·  "
               f"Último día con ventas: {res['ultimo_dia_datos']:%d/%m/%Y}  ·  "
               f"Modelo entrenado: {res['fecha_entrenamiento']}")
    st.info("La predicción usa **todo el historial** de ventas; el filtro de fechas del menú no la modifica. "
            "Los filtros de categoría y producto solo limitan qué productos puedes elegir.")

    productos = datos["productos"].copy()
    productos["etiqueta"] = (productos["codigo"] + " · " + productos["nombre"] + " " +
                             productos["talla"] + " " + productos["color"])
    elegibles = productos
    if filtros["categorias"]:
        elegibles = elegibles[elegibles["categoria"].isin(filtros["categorias"])]
    if filtros["codigos"]:
        elegibles = elegibles[elegibles["codigo"].isin(filtros["codigos"])]
    if elegibles.empty:
        st.warning("No hay productos con los filtros elegidos.")
        return

    # ---------------- Producto elegido ----------------
    codigo = st.selectbox("Elige un producto", elegibles["codigo"],
                          format_func=dict(zip(productos["codigo"], productos["etiqueta"])).get)
    fila = res["predicciones"].set_index("codigo").loc[codigo]
    evaluacion = res["evaluacion"]

    st.subheader("Unidades que se estima vender")
    columnas = st.columns(len(ml.HORIZONTES))
    for col, h in zip(columnas, ml.HORIZONTES):
        modelo_h = res["mejor_modelo"][h]
        wape = evaluacion.query("horizonte == @h and modelo == @modelo_h")["wape"].iloc[0]
        tarjeta(col, f"Próximos {h} días", f"{fila[f'pred_{h}']:.0f} unid.",
                f"Rango: {fila[f'min_{h}']:.0f} a {fila[f'max_{h}']:.0f}  ·  error típico ±{wape:.0f} %")

    dias_agotado = res["dias_agotado"].get(codigo, 0)
    if dias_agotado:
        st.caption(f"ℹ️ Este producto estuvo agotado {dias_agotado} días del historial. Esos días no se "
                   "usaron para aprender (no se vendió por falta de stock, no por falta de demanda), "
                   "así que la predicción estima la **demanda**, que puede ser mayor a lo que se vendió.")

    mostrar_grafico(datos["ventas"], res, codigo)

    # ---------------- Evaluación ----------------
    st.subheader("¿Qué tan confiables son las predicciones?")
    mostrar_evaluacion(res)

    # ---------------- Tabla de todos los productos ----------------
    st.subheader("Predicción para todos los productos")
    tabla = res["predicciones"].merge(productos[["codigo", "etiqueta", "categoria", "stock_actual"]], on="codigo")
    tabla = tabla[tabla["codigo"].isin(elegibles["codigo"])].sort_values("pred_30", ascending=False)
    tabla["rango_30"] = tabla.apply(lambda f: f"{f['min_30']:.0f} – {f['max_30']:.0f}", axis=1)
    config = {"etiqueta": "Producto", "categoria": "Categoría",
              "stock_actual": st.column_config.NumberColumn("Stock actual", format="%d"),
              "rango_30": "Rango 30 días"}
    for h in ml.HORIZONTES:
        config[f"pred_{h}"] = st.column_config.NumberColumn(f"{h} días", format="%.0f")
    st.dataframe(tabla[["etiqueta", "categoria", "stock_actual"] + [f"pred_{h}" for h in ml.HORIZONTES]
                       + ["rango_30"]],
                 hide_index=True, width="stretch", column_config=config)

    # ---------------- Explicación ----------------
    mostrar_explicacion(res, datos["ficticios"])

    st.divider()
    if st.button("🔄 Reentrenar el modelo",
                 help="Úsalo si cambiaste los archivos de datos. Tarda hasta 1 minuto."):
        obtener_resultados.clear()
        with st.spinner("Reentrenando y evaluando los modelos..."):
            obtener_resultados(datos["ventas"], datos["productos"], datos["compras"], forzar=True)
        st.rerun()


def mostrar_grafico(ventas, res, codigo):
    """Historial semanal (últimas 26 semanas) + predicción semanal (próximas 12)."""
    ultimo = res["ultimo_dia_datos"]
    diario = (ventas[ventas["codigo"] == codigo].groupby("fecha")["cantidad"].sum()
              .reindex(pd.date_range(ultimo - pd.Timedelta(days=26 * 7 - 1), ultimo), fill_value=0))
    # Bloques de 7 días contados hacia atrás desde el último día (todas las semanas completas)
    historia = diario.groupby(pd.Series(range(len(diario)), index=diario.index) // 7).agg(["sum"])
    historia["inicio"] = [diario.index[i * 7] for i in historia.index]

    modelo_largo = res["mejor_modelo"][90]
    futuro_diario = res["pronostico_diario"][modelo_largo][codigo].iloc[:84]
    futuro = futuro_diario.groupby(pd.Series(range(84), index=futuro_diario.index) // 7).sum().to_frame("sum")
    futuro["inicio"] = [futuro_diario.index[i * 7] for i in futuro.index]
    errores = res["evaluacion"].set_index(["horizonte", "modelo"])["wape"]

    def error_semana(i):
        dias = (i + 1) * 7
        h = next(h for h in ml.HORIZONTES if h >= dias)
        return errores[(h, res["mejor_modelo"][h])] / 100
    futuro["min"] = [max(0, v * (1 - error_semana(i))) for i, v in zip(futuro.index, futuro["sum"])]
    futuro["max"] = [v * (1 + error_semana(i)) for i, v in zip(futuro.index, futuro["sum"])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(futuro["inicio"]) + list(futuro["inicio"][::-1]),
                             y=list(futuro["max"]) + list(futuro["min"][::-1]), fill="toself",
                             fillcolor="rgba(235,104,52,0.15)", line=dict(width=0), hoverinfo="skip",
                             name="Rango probable"))
    fig.add_trace(go.Scatter(x=historia["inicio"], y=historia["sum"], mode="lines+markers",
                             name="Ventas reales", line=dict(color=AZUL, width=2), marker=dict(size=8),
                             hovertemplate="Semana del %{x|%d/%m/%Y}<br>%{y:.0f} unidades vendidas<extra></extra>"))
    fig.add_trace(go.Scatter(x=futuro["inicio"], y=futuro["sum"], mode="lines+markers",
                             name="Predicción del modelo", line=dict(color=NARANJA, width=2, dash="dash"),
                             marker=dict(size=8),
                             hovertemplate="Semana del %{x|%d/%m/%Y}<br>%{y:.1f} unidades estimadas<extra></extra>"))
    fig.add_vline(x=ultimo + pd.Timedelta(hours=12), line_dash="dot", line_color="#a8a7a2")
    fig.add_annotation(x=ultimo, y=1, yref="paper", text="Hoy", showarrow=False, xanchor="right",
                       font=dict(color="#52514e"))
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=30, b=10),
                      font=dict(color="#52514e", size=13), hoverlabel=dict(bgcolor="white"),
                      legend=dict(orientation="h", y=1.12, x=0), yaxis_title="Unidades por semana")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#ececea", rangemode="tozero")
    st.markdown("**Ventas semanales: últimos 6 meses y próximas 12 semanas**")
    st.plotly_chart(fig, width="stretch")


def mostrar_evaluacion(res):
    ev = res["evaluacion"].copy()
    ev["Modelo"] = ev["modelo"].map(ml.MODELOS)
    st.markdown(
        f"Para medir el error **sin mirar el futuro**, el Random Forest se entrenó solo con las ventas "
        f"hasta el **{res['fecha_corte_evaluacion'] - pd.Timedelta(days=1):%d/%m/%Y}** y se le pidió predecir "
        f"los días siguientes, que ya conocemos. Luego se comparó lo predicho con lo realmente vendido.")

    fig = px.bar(ev, x="horizonte", y="wape", color="Modelo", barmode="group",
                 color_discrete_map={ml.MODELOS[k]: v for k, v in COLOR_MODELO.items()},
                 category_orders={"Modelo": list(ml.MODELOS.values())},
                 labels={"horizonte": "Horizonte (días)", "wape": "Error (%)"})
    fig.update_traces(hovertemplate="%{fullData.name}<br>%{x} días: error %{y:.1f} %<extra></extra>")
    fig.update_xaxes(type="category", showgrid=False)
    fig.update_yaxes(ticksuffix=" %", gridcolor="#ececea")
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=30, b=10),
                      barcornerradius=4, bargap=0.25, bargroupgap=0.08,
                      font=dict(color="#52514e", size=13), hoverlabel=dict(bgcolor="white"),
                      legend=dict(orientation="h", y=1.15, x=0, title=None))
    st.plotly_chart(fig, width="stretch")

    tabla = ev.pivot(index="horizonte", columns="Modelo", values="wape")[list(ml.MODELOS.values())]
    tabla["Modelo usado"] = [ml.MODELOS[res["mejor_modelo"][h]] for h in tabla.index]
    tabla["Pruebas"] = ev.groupby("horizonte")["pruebas"].first()
    tabla.index = [f"{h} días" for h in tabla.index]
    st.dataframe(tabla, width="stretch", column_config={
        **{m: st.column_config.NumberColumn(m, format="%.1f %%") for m in ml.MODELOS.values()},
        "Pruebas": st.column_config.NumberColumn(
            "Pruebas", format="%d", help="Cuántas veces se repitió la prueba desde distintas fechas de inicio.")})

    gana_rf = sum(1 for m in res["mejor_modelo"].values() if m == "random_forest")
    st.caption(
        f"**Error (WAPE)** = suma de las diferencias entre lo predicho y lo vendido ÷ total vendido. "
        f"Un 30 % significa que, en promedio, la predicción se desvía 30 unidades por cada 100 vendidas. "
        f"Menos es mejor. El Random Forest tuvo el menor error en {gana_rf} de {len(ml.HORIZONTES)} horizontes; "
        f"en cada horizonte el sistema usa el modelo que menos se equivocó. "
        f"El horizonte de 90 días se probó una sola vez, así que su error es el menos seguro.")


def mostrar_explicacion(res, ficticios):
    with st.expander("ℹ️ ¿Cómo funciona la predicción? (algoritmo, motivos y limitaciones)"):
        variables_txt = ", ".join(res["variables"])
        estacional = ("Sí: hay 2 años o más de historia." if res["usa_estacionalidad_anual"] else
                      "**No todavía**: con solo 12 meses, cada mes se vio una sola vez y los meses a "
                      "predecir pueden no haberse visto nunca durante el entrenamiento. En las pruebas, "
                      "agregar el mes y la semana del año **empeoró** el error. Se activarán solos cuando "
                      "haya 2 años de datos.")
        st.markdown(f"""
**1. Datos.** Se arma una tabla de ventas diarias por producto ({res['filas_entrenamiento']:,} filas producto-día).
Los días en que el producto estaba **agotado** (stock reconstruido con ventas y compras) se excluyen,
porque ese día no hubo venta por falta de stock, no por falta de interés.

**2. Variables.** Para predecir un día se usan solo datos **anteriores** a ese día:
- Calendario: día de la semana.
- Fechas comerciales (fijas del calendario peruano): campaña escolar, Día de la Madre,
  Día del Padre, Fiestas Patrias, Navidad y Año Nuevo. El modelo **no asume** que vendan más:
  aprende de los datos si tienen efecto.
- Producto: categoría y precio.
- Historial: venta de hace 7 días y promedios de los últimos 7, 28 y 91 días.
- ¿Usa mes y semana del año? {estacional}

**3. Modelo: Random Forest Regressor.** Es un conjunto de 150 árboles de decisión; cada árbol
aprende reglas del tipo *"si es sábado y es Fiestas Patrias, se vende 1.6 veces el ritmo normal"*
y la predicción final es el promedio de todos los árboles. Predice un **multiplicador** sobre el
ritmo de las últimas 4 semanas, para que un mismo modelo sirva para productos que venden 0.1 o
3 unidades al día. Para horizontes largos predice día por día, usando sus propias predicciones
como historial de los días siguientes.

**4. Modelos de referencia.** "Promedio de los últimos 28 días" y "promedio de los últimos 7 días":
suponen que se seguirá vendiendo al mismo ritmo. Si el Random Forest no les gana en un horizonte,
el sistema usa el de referencia en ese horizonte.

**5. Rango.** El rango mostrado es la predicción ± el error típico (WAPE) que tuvo el modelo
en la evaluación para ese horizonte. Es una guía aproximada, no una garantía.

**Variables usadas:** {variables_txt}

**¿Por qué Random Forest?** Captura combinaciones (día de semana + temporada + nivel de venta)
sin programarlas a mano, funciona bien con pocos datos, no se altera con valores atípicos y
permite ver qué variables pesan más.

**Limitaciones**
- Solo hay 12 meses: cada temporada se vio **una vez**. Si una campaña fue atípica ese año,
  el modelo lo repetirá. A 60 y 90 días el error es mayor.
- No conoce factores externos: promociones, precios de la competencia, clima real, cambios en la economía.
- No puede predecir bien productos nuevos con menos de 4 semanas de ventas.
- Las ventas perdidas por falta de stock se estiman excluyendo los días agotados, pero el
  verdadero interés de los clientes en esos días es desconocido.
""" + ("- Son datos **ficticios**: el error real con datos de una tienda puede ser distinto.\n"
       if ficticios else ""))
        st.markdown("**Peso de cada variable en el Random Forest**")
        imp = res["importancia"].sort_values().reset_index()
        imp.columns = ["variable", "peso"]
        fig = px.bar(imp, x="peso", y="variable", orientation="h")
        fig.update_traces(marker_color=AZUL, hovertemplate="%{y}: %{x:.1%}<extra></extra>")
        fig.update_xaxes(tickformat=".0%", gridcolor="#ececea")
        fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=10, b=10),
                          barcornerradius=4, font=dict(color="#52514e", size=13),
                          xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption("Las variables de historial pesan más porque fijan el nivel de cada producto; el día de la "
                   "semana y las fechas comerciales ajustan ese nivel hacia arriba o abajo.")
