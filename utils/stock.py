"""
CONTROL PREDICTIVO DE STOCK.

Combina el stock actual, los pedidos en camino y la demanda PREDICHA por el
modelo (Etapa 5) para saber cuándo se agotará cada producto y cuánto reponer.

Fórmulas (todas en unidades):
  demanda en reposición  = suma de la demanda predicha durante los días que tarda el proveedor
  stock de seguridad     = Z x desviación diaria de ventas x raíz(días de reposición)
  punto de pedido        = demanda en reposición + stock de seguridad
  stock objetivo         = demanda predicha (reposición + días a cubrir) + stock de seguridad
  cantidad a reponer     = stock objetivo - (stock actual + en tránsito)
"""

import math

import numpy as np
import pandas as pd

# Nivel de servicio -> valor Z de la distribución normal
NIVELES_SERVICIO = {90: 1.28, 95: 1.65, 98: 2.05, 99: 2.33}


def dias_hasta_agotar(stock, pronostico):
    """
    Días hasta que la demanda acumulada predicha alcanza el stock.
    Si no se agota en el horizonte del pronóstico, se extrapola con el promedio.
    """
    if stock <= 0:
        return 0.0
    acumulado = np.cumsum(pronostico)
    if acumulado[-1] >= stock:
        dia = int(np.argmax(acumulado >= stock))
        previo = acumulado[dia - 1] if dia > 0 else 0
        fraccion = (stock - previo) / pronostico[dia] if pronostico[dia] > 0 else 1
        return dia + fraccion
    promedio = acumulado[-1] / len(pronostico)
    if promedio <= 0:
        return math.inf
    return len(pronostico) + (stock - acumulado[-1]) / promedio


def calcular_stock(productos, ventas, compras, resultados_modelo,
                   nivel_servicio=95, dias_a_cubrir=30, dias_exceso=90):
    """Devuelve una tabla con el análisis de stock de cada producto."""
    z = NIVELES_SERVICIO[nivel_servicio]
    hoy = pd.Timestamp(resultados_modelo["ultimo_dia_datos"])
    modelo_30 = resultados_modelo["mejor_modelo"][30]
    pronostico = resultados_modelo["pronostico_diario"][modelo_30]

    # Ventas de los últimos 91 días para el promedio y la variabilidad
    recientes = ventas[ventas["fecha"] > hoy - pd.Timedelta(days=91)]
    diario = (recientes.pivot_table(index="fecha", columns="codigo", values="cantidad", aggfunc="sum")
              .reindex(pd.date_range(hoy - pd.Timedelta(days=90), hoy)).fillna(0))

    en_camino = compras[compras["estado"].str.lower() != "recibido"]
    en_camino = en_camino.groupby("codigo").agg(en_transito=("cantidad", "sum"),
                                                llegada=("fecha_recepcion", "min"))

    filas = []
    for _, p in productos.iterrows():
        codigo = p["codigo"]
        stock = float(p["stock_actual"])
        reposicion = int(p["tiempo_reposicion_dias"])
        serie = pronostico[codigo].values if codigo in pronostico else np.zeros(90)
        historial = diario[codigo] if codigo in diario else pd.Series(0, index=diario.index)

        transito = float(en_camino.loc[codigo, "en_transito"]) if codigo in en_camino.index else 0.0
        llegada = en_camino.loc[codigo, "llegada"] if codigo in en_camino.index else pd.NaT
        posicion = stock + transito

        demanda_diaria = serie[:30].mean()
        demanda_reposicion = serie[:reposicion].sum()
        desviacion = historial.std()
        seguridad = z * desviacion * math.sqrt(reposicion)
        punto_pedido = demanda_reposicion + seguridad
        objetivo = serie[:reposicion + dias_a_cubrir].sum() + seguridad
        cantidad = max(0, math.ceil(objetivo - posicion))

        dias = dias_hasta_agotar(stock, serie)
        fecha_agota = hoy + pd.Timedelta(days=math.ceil(dias)) if math.isfinite(dias) else pd.NaT
        cobertura = posicion / demanda_diaria if demanda_diaria > 0 else math.inf
        llega_a_tiempo = pd.notna(llegada) and (pd.isna(fecha_agota) or llegada <= fecha_agota)

        # ---------- Estado ----------
        if stock <= 0 and not llega_a_tiempo:
            estado, motivo = "Stock crítico", "Producto agotado y sin pedido en camino."
        elif dias < reposicion and not llega_a_tiempo:
            estado = "Stock crítico"
            motivo = (f"Se agotaría en {dias:.0f} días, pero el proveedor tarda {reposicion} días: "
                      f"aunque se pida hoy, llegaría tarde.")
        elif posicion <= punto_pedido:
            estado = "Próximo a agotarse"
            motivo = (f"El stock disponible{' + en camino' if transito else ''} ({posicion:.0f}) ya está "
                      f"en el punto de pedido ({punto_pedido:.0f}) o por debajo.")
        elif demanda_diaria <= 0 and posicion > 0:
            estado = "Exceso de inventario"
            motivo = "No se esperan ventas: todo el stock queda inmovilizado."
        elif cobertura > dias_exceso:
            estado = "Exceso de inventario"
            motivo = (f"Alcanza para {cobertura:.0f} días, más que el límite de "
                      f"{dias_exceso} días.")
        else:
            estado, motivo = "Stock adecuado", "Cubre el tiempo de reposición y el stock de seguridad."

        necesita_pedido = estado in ("Stock crítico", "Próximo a agotarse")
        filas.append({
            "codigo": codigo,
            "producto": f"{p['nombre']} {p['talla']} {p['color']}",
            "categoria": p["categoria"],
            "estado": estado,
            "motivo": motivo,
            "stock_actual": int(stock),
            "en_transito": int(transito),
            "llegada_pedido": llegada,
            "promedio_vendido_diario": historial.iloc[-28:].mean(),
            "demanda_diaria_estimada": demanda_diaria,
            "dias_hasta_agotar": dias,
            "fecha_agotamiento": fecha_agota,
            "tiempo_reposicion_dias": reposicion,
            "demanda_en_reposicion": demanda_reposicion,
            "stock_seguridad": math.ceil(seguridad),
            "punto_pedido": math.ceil(punto_pedido),
            "punto_pedido_exacto": punto_pedido,
            "stock_objetivo": math.ceil(objetivo),
            "cantidad_reponer": cantidad if necesita_pedido else 0,
            "necesita_pedido": necesita_pedido,
            "cobertura_dias": cobertura,
            "precio_compra": p["precio_compra"],
            "valor_stock": stock * p["precio_compra"],
        })

    tabla = pd.DataFrame(filas)
    orden = {"Stock crítico": 0, "Próximo a agotarse": 1, "Stock adecuado": 2, "Exceso de inventario": 3}
    tabla["orden"] = tabla["estado"].map(orden)
    tabla = tabla.sort_values(["orden", "dias_hasta_agotar"]).drop(columns="orden").reset_index(drop=True)
    return tabla


def evaluar_caso(stock, venta_diaria, dias_proveedor, en_camino=0, dias_llegada=None,
                 nivel_servicio=95, dias_a_cubrir=30, dias_exceso=90):
    """
    Calculadora para un caso escrito a mano (venta diaria constante).
    Usa las MISMAS reglas que el análisis de los productos. Como la venta es
    constante, no hay variabilidad y el stock de seguridad es 0.
    """
    productos = pd.DataFrame([{
        "codigo": "PRUEBA", "nombre": "Caso de prueba", "talla": "", "color": "", "categoria": "-",
        "stock_actual": stock, "tiempo_reposicion_dias": dias_proveedor, "precio_compra": 0}])
    hoy = pd.Timestamp("2000-01-01")
    pronostico = pd.DataFrame({"PRUEBA": [venta_diaria] * 90},
                              index=pd.date_range(hoy + pd.Timedelta(days=1), periods=90))
    ventas = pd.DataFrame({"fecha": [hoy], "codigo": ["PRUEBA"], "cantidad": [0]})
    compras = pd.DataFrame({"codigo": ["PRUEBA"] if en_camino else [],
                            "cantidad": [en_camino] if en_camino else [],
                            "estado": ["En tránsito"] if en_camino else [],
                            "fecha_recepcion": [hoy + pd.Timedelta(days=dias_llegada or dias_proveedor)]
                            if en_camino else []})
    compras["estado"] = compras["estado"].astype(str)
    resultados = {"ultimo_dia_datos": hoy, "mejor_modelo": {30: "manual"},
                  "pronostico_diario": {"manual": pronostico}}
    return calcular_stock(productos, ventas, compras, resultados,
                          nivel_servicio, dias_a_cubrir, dias_exceso).iloc[0]
