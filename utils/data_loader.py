"""
Carga y VALIDACIÓN de los archivos de datos.

Todas las secciones del aplicativo obtienen los datos desde aquí, así las
validaciones se hacen en un solo lugar.

Acepta archivos .csv o .xlsx (Excel).
"""

from pathlib import Path

import pandas as pd
import streamlit as st

CARPETA_DATOS = Path(__file__).resolve().parent.parent / "data"

# Columnas obligatorias de cada archivo y su tipo esperado
COLUMNAS = {
    "productos": {
        "codigo": "texto", "nombre": "texto", "categoria": "texto",
        "talla": "texto", "color": "texto",
        "precio_compra": "numero", "precio_venta": "numero",
        "stock_actual": "numero", "stock_minimo": "numero",
        "tiempo_reposicion_dias": "numero",
    },
    "ventas": {
        "fecha": "fecha", "codigo": "texto", "cantidad": "numero",
        "precio_venta": "numero",
    },
    "compras": {
        "fecha_pedido": "fecha", "fecha_recepcion": "fecha", "codigo": "texto",
        "cantidad": "numero", "costo_unitario": "numero", "estado": "texto",
    },
}


def leer_archivo(origen):
    """Lee un CSV o Excel. 'origen' puede ser una ruta o un archivo subido."""
    nombre = getattr(origen, "name", str(origen)).lower()
    if nombre.endswith((".xlsx", ".xls")):
        return pd.read_excel(origen)
    # utf-8-sig acepta archivos guardados desde Excel con tildes
    return pd.read_csv(origen, encoding="utf-8-sig", sep=None, engine="python")


def convertir_fechas(columna):
    """
    Acepta fechas como 2026-11-18 (formato del sistema), 18/11/2026 (formato peruano)
    o fechas reales de Excel. Devuelve NaT en las que no se puedan leer.
    """
    if pd.api.types.is_datetime64_any_dtype(columna):
        return columna
    iso = pd.to_datetime(columna, errors="coerce", format="ISO8601")
    if iso.notna().sum() >= columna.notna().sum():
        return iso
    return pd.to_datetime(columna, errors="coerce", dayfirst=True, format="mixed")


def validar_tabla(df, tipo):
    """
    Revisa que la tabla tenga las columnas y tipos correctos.
    Devuelve (tabla_limpia, errores, advertencias).
    """
    errores, advertencias = [], []
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    faltantes = [c for c in COLUMNAS[tipo] if c not in df.columns]
    if faltantes:
        errores.append(f"{tipo}: faltan las columnas {', '.join(faltantes)}.")
        return df, errores, advertencias

    if df.empty:
        errores.append(f"{tipo}: el archivo no tiene filas.")
        return df, errores, advertencias

    for columna, clase in COLUMNAS[tipo].items():
        if clase == "texto":
            df[columna] = df[columna].astype(str).str.strip()
        elif clase == "numero":
            convertida = pd.to_numeric(df[columna], errors="coerce")
            malos = convertida.isna().sum()
            if malos:
                errores.append(f"{tipo}: la columna '{columna}' tiene {malos} valores que no son números.")
            elif (convertida < 0).any():
                errores.append(f"{tipo}: la columna '{columna}' tiene valores negativos.")
            df[columna] = convertida
        elif clase == "fecha":
            convertida = convertir_fechas(df[columna])
            malos = convertida.isna().sum()
            if malos:
                errores.append(f"{tipo}: la columna '{columna}' tiene {malos} fechas no válidas.")
            df[columna] = convertida

    if tipo == "productos" and "fecha_alta" in df.columns:
        df["fecha_alta"] = convertir_fechas(df["fecha_alta"])

    if tipo == "productos":
        duplicados = df["codigo"].duplicated().sum()
        if duplicados:
            errores.append(f"productos: hay {duplicados} códigos de producto repetidos.")
        if not errores and (df["precio_venta"] < df["precio_compra"]).any():
            advertencias.append("productos: hay productos con precio de venta menor al de compra.")

    if tipo == "ventas" and not errores:
        if (df["cantidad"] == 0).any():
            advertencias.append("ventas: hay ventas con cantidad 0; se ignorarán.")
            df = df[df["cantidad"] > 0]
        if "total_venta" not in df.columns:
            df["total_venta"] = df["cantidad"] * df["precio_venta"]

    return df, errores, advertencias


def _marca_archivos():
    """Nombre y fecha de modificación de cada archivo de datos: si alguno cambia, se recarga."""
    if not CARPETA_DATOS.exists():
        return ()
    return tuple(sorted((r.name, r.stat().st_mtime) for r in CARPETA_DATOS.iterdir() if r.is_file()))


def cargar_datos():
    """
    Carga productos, ventas y compras desde la carpeta 'data'.
    Devuelve un diccionario con las tablas, los errores y las advertencias.
    El resultado se guarda en memoria (caché) para que la app sea rápida, y se
    vuelve a leer automáticamente si algún archivo de la carpeta cambia.
    """
    return _cargar_datos(_marca_archivos())


@st.cache_data(show_spinner="Cargando datos...", max_entries=3)
def _cargar_datos(marca):  # "marca" cambia si cambian los archivos
    # Si existe el archivo de la simulación, los datos son los FICTICIOS de prueba
    resultado = {"errores": [], "advertencias": [],
                 "ficticios": (CARPETA_DATOS / "simulacion_referencia.csv").exists()}

    for tipo in COLUMNAS:
        ruta = next((CARPETA_DATOS / f"{tipo}{ext}" for ext in (".csv", ".xlsx")
                     if (CARPETA_DATOS / f"{tipo}{ext}").exists()), None)
        if ruta is None:
            resultado["errores"].append(
                f"No se encontró 'data/{tipo}.csv'. Ejecuta primero: python generar_datos.py")
            resultado[tipo] = pd.DataFrame()
            continue
        try:
            tabla = leer_archivo(ruta)
        except Exception as e:  # archivo dañado o con formato raro
            resultado["errores"].append(f"No se pudo leer '{ruta.name}': {e}")
            resultado[tipo] = pd.DataFrame()
            continue

        tabla, errores, advertencias = validar_tabla(tabla, tipo)
        resultado[tipo] = tabla
        resultado["errores"] += errores
        resultado["advertencias"] += advertencias

    # Revisión cruzada: toda venta debe corresponder a un producto del catálogo
    if not resultado["errores"]:
        codigos = set(resultado["productos"]["codigo"])
        desconocidos = set(resultado["ventas"]["codigo"]) - codigos
        if desconocidos:
            resultado["advertencias"].append(
                f"ventas: {len(desconocidos)} códigos no existen en productos "
                f"({', '.join(sorted(desconocidos)[:5])}...). Esas ventas se ignorarán.")
            resultado["ventas"] = resultado["ventas"][resultado["ventas"]["codigo"].isin(codigos)]

    # Completar nombre y categoría de cada venta desde el catálogo
    if not resultado["errores"]:
        info = resultado["productos"][["codigo", "nombre", "categoria", "talla", "color"]]
        ventas = resultado["ventas"].drop(
            columns=[c for c in ("nombre", "categoria", "talla", "color") if c in resultado["ventas"]])
        resultado["ventas"] = ventas.merge(info, on="codigo", how="left")

    return resultado
