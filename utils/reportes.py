"""
REPORTES: arma las tablas finales (con nombres de columnas en español) y
genera el archivo Excel con una hoja por reporte.
"""

from io import BytesIO

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from utils import calculations as calc
from utils.ml_demanda import HORIZONTES, MODELOS


def _redondear(tabla, decimales=2):
    """Redondea solo las columnas numéricas (las fechas no se tocan)."""
    numericas = tabla.select_dtypes("number").columns
    tabla = tabla.copy()
    tabla[numericas] = tabla[numericas].round(decimales)
    return tabla


def ventas_por_producto(ventas, productos):
    v = ventas.assign(costo=ventas["cantidad"] * ventas["precio_compra"]) if "precio_compra" in ventas else \
        ventas.merge(productos[["codigo", "precio_compra"]], on="codigo").assign(
            costo=lambda t: t["cantidad"] * t["precio_compra"])
    tabla = v.groupby("codigo").agg(unidades=("cantidad", "sum"), ventas=("total_venta", "sum"),
                                    costo=("costo", "sum"), dias_con_venta=("fecha", "nunique")).reset_index()
    tabla = productos[["codigo", "nombre", "categoria", "talla", "color"]].merge(tabla, on="codigo", how="left")
    tabla = tabla.fillna({"unidades": 0, "ventas": 0, "costo": 0, "dias_con_venta": 0})
    tabla["margen"] = tabla["ventas"] - tabla["costo"]
    tabla["margen_pct"] = (tabla["margen"] / tabla["ventas"].where(tabla["ventas"] > 0) * 100).round(1)
    tabla = tabla.sort_values("ventas", ascending=False)
    return tabla.rename(columns={
        "codigo": "Código", "nombre": "Producto", "categoria": "Categoría", "talla": "Talla", "color": "Color",
        "unidades": "Unidades vendidas", "ventas": "Ventas (S/)", "costo": "Costo (S/)",
        "dias_con_venta": "Días con venta", "margen": "Margen bruto (S/)", "margen_pct": "Margen (%)"})


def ventas_por_categoria(ventas):
    tabla = calc.ventas_por_categoria(ventas)
    tabla["participacion"] = (tabla["ventas_soles"] / tabla["ventas_soles"].sum() * 100).round(1)
    return tabla.rename(columns={"categoria": "Categoría", "ventas_soles": "Ventas (S/)",
                                 "unidades": "Unidades vendidas", "participacion": "Participación (%)"})


def ventas_por_mes(ventas):
    tabla = calc.ventas_por_mes(ventas)
    tabla["promedio_diario"] = (tabla["ventas_soles"] / tabla["dias_con_datos"]).round(2)
    return tabla[["mes", "ventas_soles", "unidades", "dias_con_datos", "promedio_diario"]].rename(columns={
        "mes": "Mes", "ventas_soles": "Ventas (S/)", "unidades": "Unidades vendidas",
        "dias_con_datos": "Días con ventas", "promedio_diario": "Venta promedio por día (S/)"})


def rotacion(tabla_rotacion):
    t = tabla_rotacion.sort_values("unidades_por_dia", ascending=False)
    return t[["codigo", "producto", "categoria", "rotacion", "unidades_vendidas", "unidades_por_dia",
              "frecuencia_venta", "dias_entre_ventas", "dias_desde_ultima_venta", "cobertura_dias",
              "historial_corto"]].pipe(_redondear).rename(columns={
        "codigo": "Código", "producto": "Producto", "categoria": "Categoría", "rotacion": "Rotación (K-Means)",
        "unidades_vendidas": "Unidades vendidas", "unidades_por_dia": "Unidades por día",
        "frecuencia_venta": "% de días con venta", "dias_entre_ventas": "Días entre ventas",
        "dias_desde_ultima_venta": "Días desde última venta", "cobertura_dias": "Cobertura (días)",
        "historial_corto": "Poco historial"})


def predicciones(resultados, productos):
    t = resultados["predicciones"].merge(productos[["codigo", "nombre", "talla", "color", "categoria"]], on="codigo")
    t["Producto"] = t["nombre"] + " " + t["talla"] + " " + t["color"]
    columnas = {"codigo": "Código", "Producto": "Producto", "categoria": "Categoría"}
    for h in HORIZONTES:
        columnas[f"pred_{h}"] = f"Predicción {h} días"
        columnas[f"min_{h}"] = f"Mínimo {h} días"
        columnas[f"max_{h}"] = f"Máximo {h} días"
    t = t[list(columnas)].rename(columns=columnas).pipe(_redondear, 1)
    return t


def evaluacion_modelos(resultados):
    ev = resultados["evaluacion"].copy()
    ev["modelo"] = ev["modelo"].map(MODELOS)
    tabla = ev.pivot(index="horizonte", columns="modelo", values="wape").round(1).reset_index()
    tabla["Modelo usado"] = [MODELOS[resultados["mejor_modelo"][h]] for h in tabla["horizonte"]]
    return tabla.rename(columns={"horizonte": "Horizonte (días)"})


def inventario(tabla_stock):
    t = tabla_stock.copy()
    t["dias_hasta_agotar"] = t["dias_hasta_agotar"].clip(upper=999).round(1)
    t["cobertura_dias"] = t["cobertura_dias"].clip(upper=999).round(1)
    return t[["codigo", "producto", "categoria", "estado", "stock_actual", "en_transito", "llegada_pedido",
              "promedio_vendido_diario", "demanda_diaria_estimada", "dias_hasta_agotar", "fecha_agotamiento",
              "tiempo_reposicion_dias", "stock_seguridad", "punto_pedido", "cantidad_reponer",
              "valor_stock", "motivo"]].pipe(_redondear).rename(columns={
        "codigo": "Código", "producto": "Producto", "categoria": "Categoría", "estado": "Estado",
        "stock_actual": "Stock actual", "en_transito": "En camino", "llegada_pedido": "Llega el",
        "promedio_vendido_diario": "Vendido por día (últ. 28 días)",
        "demanda_diaria_estimada": "Demanda diaria estimada", "dias_hasta_agotar": "Días hasta agotar",
        "fecha_agotamiento": "Fecha aprox. de agotamiento", "tiempo_reposicion_dias": "Proveedor (días)",
        "stock_seguridad": "Stock de seguridad", "punto_pedido": "Punto de pedido",
        "cantidad_reponer": "Cantidad a reponer", "valor_stock": "Valor del stock (S/)", "motivo": "Motivo"})


def abastecimiento(plan):
    return plan[["accion", "fecha_pedido", "codigo", "producto", "proveedor", "cantidad", "fecha_llegada",
                 "costo", "riesgo_quiebre", "exceso_unidades", "valor_exceso", "motivo"]].rename(columns={
        "accion": "Acción", "fecha_pedido": "Pedir el", "codigo": "Código", "producto": "Producto",
        "proveedor": "Proveedor", "cantidad": "Cantidad", "fecha_llegada": "Llega aprox.",
        "costo": "Costo (S/)", "riesgo_quiebre": "Riesgo de quedarse sin stock",
        "exceso_unidades": "Unidades en exceso", "valor_exceso": "Valor del exceso (S/)", "motivo": "Motivo"})


def temporadas(resultado):
    if resultado.empty:
        return resultado
    t = resultado.copy()
    return t[["temporada", "anio", "inicio", "fin", "promedio_temporada", "promedio_normal",
              "diferencia_pct", "p_valor", "conclusion"]].pipe(_redondear, 3).rename(columns={
        "temporada": "Temporada", "anio": "Año", "inicio": "Desde", "fin": "Hasta",
        "promedio_temporada": "Unid./día en temporada", "promedio_normal": "Unid./día normales",
        "diferencia_pct": "Diferencia (%)", "p_valor": "p-valor", "conclusion": "Conclusión"})


def _a_fechas(tabla):
    """Quita la hora de las columnas de fecha para que Excel las muestre limpias."""
    tabla = tabla.copy()
    for col in tabla.columns:
        if pd.api.types.is_datetime64_any_dtype(tabla[col]):
            tabla[col] = tabla[col].dt.date
    return tabla


def generar_excel(reportes, informacion):
    """Crea un Excel en memoria: hoja 'Información' + una hoja por reporte."""
    salida = BytesIO()
    encabezado = PatternFill("solid", fgColor="2A78D6")
    with pd.ExcelWriter(salida, engine="openpyxl") as libro:
        pd.DataFrame(informacion, columns=["Dato", "Valor"]).to_excel(libro, sheet_name="Información", index=False)
        for nombre, tabla in reportes.items():
            _a_fechas(tabla).to_excel(libro, sheet_name=nombre[:31], index=False)

        for hoja in libro.book.worksheets:
            for celda in hoja[1]:
                celda.font = Font(bold=True, color="FFFFFF")
                celda.fill = encabezado
                celda.alignment = Alignment(wrap_text=True, vertical="center")
            hoja.freeze_panes = "A2"
            if hoja.max_row > 1:
                hoja.auto_filter.ref = hoja.dimensions
            for i, columna in enumerate(hoja.columns, start=1):
                largo = max(len(str(c.value)) if c.value is not None else 0 for c in columna)
                hoja.column_dimensions[get_column_letter(i)].width = min(max(10, largo + 2), 60)
                for celda in columna[1:]:
                    if hasattr(celda.value, "year"):
                        celda.number_format = "DD/MM/YYYY"
    return salida.getvalue()
