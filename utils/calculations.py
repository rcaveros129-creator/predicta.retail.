"""
Cálculos de indicadores que usan varias secciones del aplicativo.

Aquí NO hay Machine Learning todavía: son sumas, promedios y conteos sobre los
datos. Los modelos se agregarán en las Etapas 4 y 5.
"""

import pandas as pd

DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Set", "Oct", "Nov", "Dic"]


def filtrar_ventas(ventas, fecha_inicio, fecha_fin, categorias=None, codigos=None):
    """Devuelve solo las ventas dentro del rango de fechas y filtros elegidos."""
    filtro = (ventas["fecha"] >= pd.Timestamp(fecha_inicio)) & \
             (ventas["fecha"] <= pd.Timestamp(fecha_fin))
    if categorias:
        filtro &= ventas["categoria"].isin(categorias)
    if codigos:
        filtro &= ventas["codigo"].isin(codigos)
    return ventas[filtro]


def filtrar_productos(productos, categorias=None, codigos=None):
    filtro = pd.Series(True, index=productos.index)
    if categorias:
        filtro &= productos["categoria"].isin(categorias)
    if codigos:
        filtro &= productos["codigo"].isin(codigos)
    return productos[filtro]


def ventas_del_mes(ventas, fecha_fin):
    """
    Ventas desde el día 1 del mes de 'fecha_fin' hasta 'fecha_fin', y las del
    mismo tramo del mes anterior (para comparar de forma justa).
    """
    fin = pd.Timestamp(fecha_fin)
    inicio = fin.replace(day=1)
    inicio_ant = inicio - pd.DateOffset(months=1)
    fin_ant = min(inicio_ant + pd.Timedelta(days=fin.day - 1), inicio - pd.Timedelta(days=1))

    actual = ventas[(ventas["fecha"] >= inicio) & (ventas["fecha"] <= fin)]["total_venta"].sum()
    anterior = ventas[(ventas["fecha"] >= inicio_ant) & (ventas["fecha"] <= fin_ant)]["total_venta"].sum()
    variacion = (actual - anterior) / anterior * 100 if anterior > 0 else None
    return actual, anterior, variacion, inicio, fin


def ventas_por_mes(ventas):
    tabla = (ventas.groupby(ventas["fecha"].dt.to_period("M"))
             .agg(ventas_soles=("total_venta", "sum"), unidades=("cantidad", "sum"),
                  dias_con_datos=("fecha", "nunique"))
             .reset_index())
    tabla["mes"] = [f"{MESES[p.month - 1]} {p.year}" for p in tabla["fecha"]]
    return tabla


def ventas_por_categoria(ventas):
    return (ventas.groupby("categoria")
            .agg(ventas_soles=("total_venta", "sum"), unidades=("cantidad", "sum"))
            .reset_index()
            .sort_values("ventas_soles", ascending=False))


def ventas_por_dia_semana(ventas):
    """Unidades promedio vendidas por cada día de la semana."""
    diario = ventas.groupby("fecha")["cantidad"].sum().reset_index()
    diario["dia"] = diario["fecha"].dt.day_of_week
    tabla = diario.groupby("dia")["cantidad"].mean().reindex(range(7), fill_value=0).reset_index()
    tabla["nombre_dia"] = DIAS_SEMANA
    tabla.columns = ["dia", "unidades_promedio", "nombre_dia"]
    return tabla


def ranking_productos(ventas, productos):
    """Unidades y soles vendidos por producto, incluyendo los que no vendieron nada."""
    resumen = ventas.groupby("codigo").agg(
        unidades=("cantidad", "sum"), ventas_soles=("total_venta", "sum")).reset_index()
    tabla = productos[["codigo", "nombre", "talla", "color", "categoria"]].merge(
        resumen, on="codigo", how="left").fillna({"unidades": 0, "ventas_soles": 0})
    tabla["producto"] = tabla["nombre"] + " " + tabla["talla"] + " " + tabla["color"]
    return tabla.sort_values("unidades", ascending=False)


def valor_inventario(productos):
    """Valor del stock actual a precio de compra (costo) y a precio de venta."""
    a_costo = (productos["stock_actual"] * productos["precio_compra"]).sum()
    a_venta = (productos["stock_actual"] * productos["precio_venta"]).sum()
    return a_costo, a_venta
