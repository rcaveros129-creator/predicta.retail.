"""
RECOMENDACIONES DE ABASTECIMIENTO.

Parte del análisis de stock (Etapa 6) y proyecta el inventario día por día con
la demanda PREDICHA para responder:
  - ¿Cuándo hay que hacer el pedido?  -> el día en que (stock + en camino) baja
                                         hasta el punto de pedido.
  - ¿Cuánto pedir?                    -> lo necesario para cubrir el tiempo del
                                         proveedor + los días a cubrir + stock de
                                         seguridad, en la fecha del pedido.
  - ¿Qué productos sobran?            -> los que tienen stock para más días que
                                         el límite de exceso.
"""

import math

import numpy as np
import pandas as pd

ACCIONES = {
    "Pedir hoy": "🛒",
    "Programar pedido": "📅",
    "No comprar: exceso": "⏸️",
    "Sin pedido próximo": "✔️",
}


def recomendar(tabla_stock, productos, resultados_modelo, dias_a_cubrir=30, dias_exceso=90):
    hoy = pd.Timestamp(resultados_modelo["ultimo_dia_datos"])
    modelo_30 = resultados_modelo["mejor_modelo"][30]
    pronostico = resultados_modelo["pronostico_diario"][modelo_30]
    info = productos.set_index("codigo")
    filas = []

    for _, s in tabla_stock.iterrows():
        codigo = s["codigo"]
        serie = pronostico[codigo].values if codigo in pronostico else np.zeros(90)
        horizonte = len(serie)
        reposicion = int(s["tiempo_reposicion_dias"])
        posicion_hoy = s["stock_actual"] + s["en_transito"]
        acumulada = np.concatenate([[0], np.cumsum(serie)])     # acumulada[d] = venta de los días 1..d

        # Día en que (stock + en camino) proyectado llega al punto de pedido
        dia_pedido = None
        for d in range(horizonte - reposicion):
            if posicion_hoy - acumulada[d] <= s["punto_pedido_exacto"]:
                dia_pedido = d
                break

        precio = float(info.loc[codigo, "precio_compra"])
        proveedor = info.loc[codigo, "proveedor"] if "proveedor" in info.columns else "—"
        exceso_unidades = 0

        if s["estado"] == "Exceso de inventario":
            accion = "No comprar: exceso"
            necesario = serie[:dias_exceso].sum() if dias_exceso <= horizonte else \
                serie.mean() * dias_exceso
            exceso_unidades = max(0, math.floor(posicion_hoy - necesario))
            cantidad, fecha_pedido = 0, pd.NaT
            if exceso_unidades > 0:
                motivo = (f"Al ritmo actual tiene stock para ~{min(s['cobertura_dias'], 999):.0f} días. "
                          f"Sobran unas {exceso_unidades} unidades respecto a la venta prevista para "
                          f"{dias_exceso} días: conviene una promoción o liquidación.")
            else:
                motivo = (f"Al ritmo actual tiene stock para ~{min(s['cobertura_dias'], 999):.0f} días, pero "
                          f"el modelo prevé más ventas en las próximas semanas que absorberán ese stock. "
                          f"No comprar por ahora; no hace falta liquidar.")
        elif dia_pedido is None:
            accion, cantidad, fecha_pedido = "Sin pedido próximo", 0, pd.NaT
            motivo = (f"El stock alcanza más allá de los próximos {horizonte - reposicion} días "
                      f"analizados; no hace falta pedir por ahora.")
        else:
            fecha_pedido = hoy + pd.Timedelta(days=dia_pedido + 1)   # día siguiente al último dato
            posicion = posicion_hoy - acumulada[dia_pedido]
            fin = min(horizonte, dia_pedido + reposicion + dias_a_cubrir)
            demanda_cubrir = acumulada[fin] - acumulada[dia_pedido]
            if fin - dia_pedido < reposicion + dias_a_cubrir:      # completar si pasa el horizonte
                demanda_cubrir += serie.mean() * (reposicion + dias_a_cubrir - (fin - dia_pedido))
            cantidad = max(1, math.ceil(demanda_cubrir + s["stock_seguridad"] - posicion))
            accion = "Pedir hoy" if dia_pedido == 0 else "Programar pedido"
            if s["estado"] == "Stock crítico":
                motivo = "URGENTE: " + s["motivo"]
            elif dia_pedido == 0:
                motivo = "Ya llegó a su punto de pedido: hay que pedir hoy."
            else:
                motivo = (f"En {dia_pedido} días el stock (+ en camino) llegará al punto de pedido "
                          f"({s['punto_pedido']} unid.).")

        llegada = fecha_pedido + pd.Timedelta(days=reposicion) if pd.notna(fecha_pedido) else pd.NaT
        dias_para_pedir = (fecha_pedido - hoy).days - 1 if pd.notna(fecha_pedido) else None

        if s["estado"] == "Stock crítico":
            riesgo = "Alto"
        elif s["estado"] == "Próximo a agotarse" or (dias_para_pedir is not None and dias_para_pedir <= 7):
            riesgo = "Medio"
        else:
            riesgo = "Bajo"

        filas.append({
            "accion": accion,
            "codigo": codigo,
            "producto": s["producto"],
            "categoria": s["categoria"],
            "proveedor": proveedor,
            "estado": s["estado"],
            "riesgo_quiebre": riesgo,
            "stock_actual": s["stock_actual"],
            "en_transito": s["en_transito"],
            "fecha_pedido": fecha_pedido,
            "dias_para_pedir": dias_para_pedir,
            "cantidad": cantidad,
            "fecha_llegada": llegada,
            "precio_compra": precio,
            "costo": cantidad * precio,
            "exceso_unidades": exceso_unidades,
            "valor_exceso": exceso_unidades * precio,
            "motivo": motivo,
        })

    tabla = pd.DataFrame(filas)
    orden = {"Pedir hoy": 0, "Programar pedido": 1, "No comprar: exceso": 2, "Sin pedido próximo": 3}
    tabla["_orden"] = tabla["accion"].map(orden)
    return (tabla.sort_values(["_orden", "fecha_pedido", "codigo"], na_position="last")
            .drop(columns="_orden").reset_index(drop=True))
