"""
Sección 8: REPORTES Y ESTADÍSTICAS (descarga en Excel o CSV).
"""

import pandas as pd
import streamlit as st

from modules.inventario import obtener_tabla
from modules.rotacion import ejecutar_modelo
from modules.ventas import temporadas_cacheadas
from utils import calculations as calc
from utils import reportes as rep
from utils.abastecimiento import recomendar
from utils.componentes import fuente


def mostrar(datos, filtros):
    st.title("Reportes")
    ventas = calc.filtrar_ventas(datos["ventas"], filtros["fecha_inicio"], filtros["fecha_fin"],
                                 filtros["categorias"], filtros["codigos"])
    productos = calc.filtrar_productos(datos["productos"], filtros["categorias"], filtros["codigos"])
    st.caption(f"Periodo de ventas: {filtros['fecha_inicio']:%d/%m/%Y} al {filtros['fecha_fin']:%d/%m/%Y}  ·  "
               f"Los filtros del menú se aplican a los reportes  ·  {fuente(datos)}")
    if ventas.empty:
        st.warning("No hay ventas con los filtros elegidos.")
        return

    nivel = st.session_state.get("nivel_servicio", 95)
    dias_cubrir = st.session_state.get("dias_a_cubrir", 30)
    dias_exceso = st.session_state.get("dias_exceso", 90)

    with st.spinner("Preparando reportes..."):
        reportes = {
            "Ventas por producto": rep.ventas_por_producto(ventas, productos),
            "Ventas por categoría": rep.ventas_por_categoria(ventas),
            "Ventas por mes": rep.ventas_por_mes(ventas),
        }
        origen = {nombre: "Datos de ventas" for nombre in reportes}

        try:
            rot, _, _ = ejecutar_modelo(datos["ventas"], datos["productos"],
                                        filtros["fecha_inicio"], filtros["fecha_fin"])
            rot = rot[rot["codigo"].isin(productos["codigo"])]
            reportes["Rotación"] = rep.rotacion(rot)
            origen["Rotación"] = "Modelo K-Means"
        except ValueError as e:
            st.info(f"Rotación no disponible: {e}")

        try:
            stock, res = obtener_tabla(datos, nivel, dias_cubrir, dias_exceso)
            codigos = set(productos["codigo"])
            reportes["Predicciones"] = rep.predicciones(res, productos)
            reportes["Error de los modelos"] = rep.evaluacion_modelos(res)
            reportes["Estado del inventario"] = rep.inventario(stock[stock["codigo"].isin(codigos)])
            plan = recomendar(stock, datos["productos"], res, dias_cubrir, dias_exceso)
            reportes["Abastecimiento"] = rep.abastecimiento(plan[plan["codigo"].isin(codigos)])
            origen.update({"Predicciones": "Modelo Random Forest / referencia",
                           "Error de los modelos": "Evaluación temporal",
                           "Estado del inventario": "Predicción + stock actual",
                           "Abastecimiento": "Predicción + stock actual"})
        except ValueError as e:
            st.info(f"Predicciones no disponibles: {e}")

        if ventas["fecha"].nunique() >= 60:
            reportes["Temporadas"] = rep.temporadas(temporadas_cacheadas(ventas))
            origen["Temporadas"] = "Prueba de permutación"

    informacion = [
        ("Sistema", "Predicta Retail"),
        ("Generado el", pd.Timestamp.now().strftime("%d/%m/%Y %H:%M")),
        ("Periodo de ventas", f"{filtros['fecha_inicio']:%d/%m/%Y} al {filtros['fecha_fin']:%d/%m/%Y}"),
        ("Categorías", ", ".join(filtros["categorias"]) or "Todas"),
        ("Productos", ", ".join(filtros["codigos"]) or "Todos"),
        ("Nivel de servicio", f"{nivel} %"),
        ("Días que cubre cada pedido", dias_cubrir),
        ("Exceso si cubre más de (días)", dias_exceso),
        ("Origen de los datos", fuente(datos)),
        ("Aviso", "Las predicciones son estimaciones de modelos y tienen error."),
    ]

    st.subheader("Descargar todo")
    st.download_button(
        "📥 Descargar reporte completo (Excel)", rep.generar_excel(reportes, informacion),
        file_name=f"reporte_predicta_{pd.Timestamp.now():%Y%m%d}.xlsx", type="primary",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.caption(f"Un archivo Excel con {len(reportes) + 1} hojas: Información + "
               f"{', '.join(reportes)}.")

    st.subheader("Reportes individuales")
    for nombre, tabla in reportes.items():
        with st.expander(f"📄 {nombre}  ·  {len(tabla)} filas  ·  {origen[nombre]}"):
            st.dataframe(tabla, hide_index=True, width="stretch")
            st.download_button(f"⬇️ Descargar «{nombre}» (CSV)", tabla.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{nombre.lower().replace(' ', '_')}.csv", mime="text/csv",
                               key=f"csv_{nombre}")
