"""
PREDICTA RETAIL - Etapa 2: generador de DATOS FICTICIOS.

Este script crea 12 meses de datos SIMULADOS de una tienda de ropa:

    data/productos.csv              -> catálogo y stock ACTUAL (una foto de hoy)
    data/ventas.csv                 -> historial de ventas (movimientos de salida)
    data/compras.csv                -> historial de pedidos al proveedor (entradas)
    data/simulacion_referencia.csv  -> "respuestas" de la simulación, SOLO para
                                       comparar después. El aplicativo NUNCA la
                                       usará para entrenar modelos.

Cómo ejecutarlo (con el entorno virtual activado):
    python generar_datos.py

Si lo ejecutas otra vez, los archivos se reemplazan. Como usamos una
"semilla" fija, siempre obtendrás los mismos números.

IMPORTANTE: todo lo que genera este archivo es INVENTADO para poder probar el
sistema. No son ventas reales ni predicciones.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. CONFIGURACIÓN GENERAL
# ------------------------------------------------------------------
SEMILLA = 2026                          # misma semilla = mismos datos siempre
DIAS_HISTORIAL = 365                    # 12 meses
FECHA_FIN = date.today() - timedelta(days=1)   # último día con ventas: ayer
FECHA_INICIO = FECHA_FIN - timedelta(days=DIAS_HISTORIAL - 1)

# Ajusta el tamaño del negocio: 1.0 = demanda del catálogo, 0.6 = 60 %.
# Con 0.6 la tienda vende unos S/ 40 000 al mes (tamaño de microempresa).
ESCALA_DEMANDA = 0.6

CARPETA_DATOS = Path(__file__).parent / "data"

rng = np.random.default_rng(SEMILLA)

# Proveedores (nombres ficticios) y días que tardan en entregar
PROVEEDORES = {
    "Polos":       ("Textiles Gamarra (ficticio)", 5),
    "Poleras":     ("Textiles Gamarra (ficticio)", 5),
    "Shorts":      ("Textiles Gamarra (ficticio)", 5),
    "Pantalones":  ("Confecciones Lima Norte (ficticio)", 7),
    "Camisas":     ("Confecciones Lima Norte (ficticio)", 7),
    "Blusas":      ("Confecciones Lima Norte (ficticio)", 7),
    "Vestidos":    ("Moda Femenina Perú (ficticio)", 8),
    "Casacas":     ("Abrigos del Sur (ficticio)", 12),
    "Chompas":     ("Abrigos del Sur (ficticio)", 12),
    "Buzos":       ("Abrigos del Sur (ficticio)", 12),
    "Zapatillas":  ("Calzados Trujillo (ficticio)", 10),
    "Accesorios":  ("Importaciones Mesa Redonda (ficticio)", 4),
}

# ------------------------------------------------------------------
# 2. CATÁLOGO DE PRODUCTOS
# ------------------------------------------------------------------
# Columnas:
#   codigo, nombre, categoria, talla, color, precio_compra, precio_venta,
#   demanda_base (unidades/día promedio que SE SIMULAN, no lo sabe la tienda),
#   temporadas (etiquetas que modifican la demanda en ciertas fechas),
#   politica (cómo repone el dueño: "normal", "descuidada" o "sobrestock"),
#   dias_desde_inicio (en qué día del año empezó a venderse; 0 = desde el inicio)
CATALOGO = [
    ("POL-001", "Polo básico algodón",        "Polos",      "M",     "Negro",       12, 29, 3.2, [],                     "normal",     0),
    ("POL-002", "Polo básico algodón",        "Polos",      "L",     "Blanco",      12, 29, 2.8, ["verano"],             "descuidada", 0),
    ("POL-003", "Polo básico algodón",        "Polos",      "S",     "Gris",        12, 29, 1.2, [],                     "normal",     0),
    ("POL-004", "Polo estampado",             "Polos",      "M",     "Azul",        15, 35, 2.2, ["verano"],             "normal",     0),
    ("POL-005", "Polo oversize",              "Polos",      "L",     "Beige",       18, 45, 0.9, [],                     "normal",     0),
    ("POL-006", "Polo dry-fit deportivo",     "Polos",      "M",     "Negro",       16, 39, 1.6, [],                     "normal",     235),
    ("POE-001", "Polera con capucha",         "Poleras",    "M",     "Negro",       32, 79, 1.5, ["invierno"],           "normal",     0),
    ("PAN-001", "Jean clásico",               "Pantalones", "32",    "Azul",        38, 89, 2.0, ["escolar"],            "normal",     0),
    ("PAN-002", "Jean clásico",               "Pantalones", "30",    "Azul",        38, 89, 1.3, [],                     "normal",     0),
    ("PAN-003", "Pantalón drill",             "Pantalones", "32",    "Beige",       35, 79, 0.8, [],                     "normal",     0),
    ("PAN-004", "Jogger deportivo",           "Pantalones", "M",     "Negro",       28, 65, 1.1, ["escolar"],            "normal",     0),
    ("PAN-005", "Pantalón de vestir",         "Pantalones", "34",    "Gris",        45, 110, 0.15, [],                   "sobrestock", 0),
    ("CAM-001", "Camisa manga larga",         "Camisas",    "M",     "Celeste",     30, 69, 0.9, ["padre"],              "normal",     0),
    ("CAM-002", "Camisa manga larga",         "Camisas",    "L",     "Blanco",      30, 69, 0.7, ["padre"],              "normal",     0),
    ("CAM-003", "Camisa casual a cuadros",    "Camisas",    "M",     "Rojo",        32, 75, 0.25, [],                    "normal",     0),
    ("CAM-004", "Camisa de lino",             "Camisas",    "L",     "Blanco",      40, 95, 0.2, ["verano"],             "sobrestock", 0),
    ("BLU-001", "Blusa manga corta",          "Blusas",     "M",     "Blanco",      22, 55, 1.0, ["madre", "verano"],    "normal",     0),
    ("VES-001", "Vestido casual",             "Vestidos",   "S",     "Rojo",        40, 99, 0.6, ["madre"],              "normal",     0),
    ("VES-002", "Vestido floral",             "Vestidos",   "M",     "Estampado",   42, 105, 0.7, ["madre", "verano"],   "normal",     0),
    ("VES-003", "Vestido de fiesta",          "Vestidos",   "M",     "Negro",       70, 169, 0.12, ["fiesta"],           "sobrestock", 0),
    ("CAS-001", "Casaca jean",                "Casacas",    "M",     "Azul",        55, 129, 0.6, ["invierno"],          "normal",     0),
    ("CAS-002", "Casaca cortaviento",         "Casacas",    "L",     "Negro",       48, 115, 0.7, ["invierno"],          "normal",     0),
    ("CAS-003", "Casaca de cuero sintético",  "Casacas",    "M",     "Marrón",      80, 189, 0.1, ["invierno"],          "sobrestock", 0),
    ("CHO-001", "Chompa tejida",              "Chompas",    "M",     "Gris",        35, 85, 0.6, ["invierno"],           "normal",     0),
    ("BUZ-001", "Buzo deportivo conjunto",    "Buzos",      "M",     "Azul marino", 50, 119, 0.7, ["invierno", "escolar"], "normal",   0),
    ("SHO-001", "Short drill",                "Shorts",     "32",    "Beige",       22, 49, 1.0, ["verano"],             "normal",     0),
    ("SHO-002", "Short de baño",              "Shorts",     "M",     "Azul",        18, 45, 0.8, ["verano"],             "normal",     0),
    ("ZAP-001", "Zapatillas urbanas",         "Zapatillas", "40",    "Blanco",      75, 159, 1.8, ["escolar"],           "descuidada", 0),
    ("ZAP-002", "Zapatillas urbanas",         "Zapatillas", "42",    "Negro",       75, 159, 1.0, ["escolar"],           "normal",     0),
    ("ZAP-003", "Zapatillas running",         "Zapatillas", "41",    "Gris",        95, 199, 0.3, [],                    "normal",     0),
    ("ACC-001", "Gorra",                      "Accesorios", "Única", "Negro",       10, 25, 1.0, ["verano"],             "normal",     0),
    ("ACC-002", "Correa de cuero",            "Accesorios", "Única", "Marrón",      15, 39, 0.2, ["padre"],              "sobrestock", 0),
    ("ACC-003", "Medias pack x3",             "Accesorios", "Única", "Blanco",      8, 19, 2.5, ["escolar"],             "normal",     0),
]

# ------------------------------------------------------------------
# 3. FACTORES QUE MODIFICAN LA DEMANDA SIMULADA
# ------------------------------------------------------------------
# Día de la semana (0 = lunes ... 6 = domingo): los sábados se vende más
FACTOR_DIA_SEMANA = [0.75, 0.80, 0.85, 0.90, 1.10, 1.45, 1.15]

# Efecto general de cada mes sobre TODA la tienda
FACTOR_MES = {1: 0.80, 2: 0.95, 3: 1.00, 4: 0.95, 5: 1.05, 6: 1.00,
              7: 1.15, 8: 0.95, 9: 0.82, 10: 0.88, 11: 1.00, 12: 1.30}


def domingo_numero(anio, mes, numero):
    """Devuelve el 'numero'-ésimo domingo de un mes (ej. 2do domingo de mayo)."""
    dia = date(anio, mes, 1)
    while dia.weekday() != 6:
        dia += timedelta(days=1)
    return dia + timedelta(weeks=numero - 1)


def entre(d, inicio, fin):
    return inicio <= d <= fin


def factor_general(d):
    """Efecto sobre todos los productos: mes + Fiestas Patrias + Navidad."""
    f = FACTOR_MES[d.month]
    if entre(d, date(d.year, 7, 15), date(d.year, 7, 28)):    # gratificación + Fiestas Patrias
        f *= 1.30
    if entre(d, date(d.year, 12, 14), date(d.year, 12, 24)):  # compras navideñas
        f *= 1.40
    if entre(d, date(d.year, 12, 26), date(d.year, 12, 31)):  # Año Nuevo
        f *= 1.15
    return f


def factor_temporada(d, etiqueta):
    """Efecto de cada etiqueta de temporada sobre un producto."""
    m = d.month
    if etiqueta == "verano":        # en Perú: diciembre a marzo
        return {12: 1.6, 1: 1.7, 2: 1.7, 3: 1.4, 4: 1.0, 11: 1.1,
                6: 0.5, 7: 0.5, 8: 0.55}.get(m, 0.8)
    if etiqueta == "invierno":      # junio a agosto
        return {6: 1.8, 7: 1.9, 8: 1.7, 5: 1.3, 9: 1.2,
                12: 0.35, 1: 0.3, 2: 0.3, 3: 0.4}.get(m, 0.8)
    if etiqueta == "escolar":       # campaña escolar
        return 2.2 if entre(d, date(d.year, 2, 10), date(d.year, 3, 20)) else 1.0
    if etiqueta == "madre":         # 10 días antes del Día de la Madre
        dia_madre = domingo_numero(d.year, 5, 2)
        return 2.5 if entre(d, dia_madre - timedelta(days=10), dia_madre) else 1.0
    if etiqueta == "fiesta":        # fiestas de fin de año
        return 3.0 if entre(d, date(d.year, 12, 15), date(d.year, 12, 31)) else 1.0
    if etiqueta == "padre":
        # PRUEBA INTENCIONAL: el Día del Padre NO aumenta las ventas en esta
        # simulación. Así podremos comprobar que el análisis de temporadas no
        # "inventa" efectos que no existen en los datos.
        return 1.0
    return 1.0


# ------------------------------------------------------------------
# 4. SIMULACIÓN DÍA POR DÍA
# ------------------------------------------------------------------
def simular():
    fechas = [FECHA_INICIO + timedelta(days=i) for i in range(DIAS_HISTORIAL)]
    ventas, compras, referencia, productos = [], [], [], []
    id_venta, id_compra = 1, 1

    for (codigo, nombre, categoria, talla, color, p_compra, p_venta,
         demanda_base, temporadas, politica, dia_alta) in CATALOGO:

        proveedor, tiempo_reposicion = PROVEEDORES[categoria]
        demanda_base = demanda_base * ESCALA_DEMANDA

        # Reglas de reposición que "usa" el dueño en la simulación
        tamano_pedido = max(6, int(round(demanda_base * 30)))       # ~1 mes de ventas
        if politica == "descuidada":       # pide tarde -> se queda sin stock a veces
            punto_pedido = demanda_base * tiempo_reposicion * 0.4
        else:
            punto_pedido = demanda_base * (tiempo_reposicion + 7)

        if politica == "sobrestock":       # compró demasiado: ~2 años de ventas
            stock = int(round(demanda_base * 365 * rng.uniform(1.8, 2.4)))
        else:
            stock = max(6, int(round(demanda_base * 30 * rng.uniform(1.0, 1.4))))
        if dia_alta > 0:                   # producto nuevo: llega su primer pedido
            stock = tamano_pedido

        stock_inicial = stock
        pedidos_pendientes = []            # lista de (fecha_llegada, cantidad)
        dias_sin_stock = 0
        demanda_perdida = 0

        for i, d in enumerate(fechas):
            if i < dia_alta:               # todavía no se vendía este producto
                continue

            # a) Llegan los pedidos que tocaban hoy
            for llegada, cantidad in [p for p in pedidos_pendientes if p[0] == d]:
                stock += cantidad
            pedidos_pendientes = [p for p in pedidos_pendientes if p[0] != d]

            # b) Demanda del día (número aleatorio con distribución de Poisson)
            tendencia = 1 + 0.10 * i / DIAS_HISTORIAL        # crecimiento suave de 10% al año
            lam = demanda_base * factor_general(d) * FACTOR_DIA_SEMANA[d.weekday()] * tendencia
            for t in temporadas:
                lam *= factor_temporada(d, t)
            demanda = int(rng.poisson(lam))

            # c) Solo se puede vender lo que hay en stock
            vendido = min(demanda, stock)
            stock -= vendido
            if demanda > vendido:
                demanda_perdida += demanda - vendido
                dias_sin_stock += 1

            # d) Se registran las ventas como tickets de 1 a 3 unidades
            restante = vendido
            while restante > 0:
                cantidad_ticket = min(restante, int(rng.choice([1, 1, 1, 1, 2, 2, 3])))
                ventas.append({
                    "id_venta": f"V{id_venta:06d}",
                    "fecha": d.isoformat(),
                    "codigo": codigo,
                    "nombre": nombre,
                    "categoria": categoria,
                    "talla": talla,
                    "color": color,
                    "cantidad": cantidad_ticket,
                    "precio_compra": p_compra,
                    "precio_venta": p_venta,
                    "total_venta": cantidad_ticket * p_venta,
                })
                id_venta += 1
                restante -= cantidad_ticket

            # e) Si el stock está bajo y no hay pedido en camino, se hace un pedido
            if stock <= punto_pedido and not pedidos_pendientes:
                demora = max(2, tiempo_reposicion + int(rng.integers(-1, 3)))
                llegada = d + timedelta(days=demora)
                pedidos_pendientes.append((llegada, tamano_pedido))
                compras.append({
                    "id_compra": f"C{id_compra:05d}",
                    "fecha_pedido": d.isoformat(),
                    "fecha_recepcion": llegada.isoformat(),
                    "codigo": codigo,
                    "nombre": nombre,
                    "categoria": categoria,
                    "proveedor": proveedor,
                    "cantidad": tamano_pedido,
                    "costo_unitario": p_compra,
                    "costo_total": tamano_pedido * p_compra,
                    "estado": "Recibido" if llegada <= FECHA_FIN else "En tránsito",
                })
                id_compra += 1

        productos.append({
            "codigo": codigo,
            "nombre": nombre,
            "categoria": categoria,
            "talla": talla,
            "color": color,
            "precio_compra": p_compra,
            "precio_venta": p_venta,
            "stock_actual": stock,
            "stock_minimo": max(2, int(np.ceil(demanda_base * 7))),
            "tiempo_reposicion_dias": tiempo_reposicion,
            "proveedor": proveedor,
            "fecha_alta": fechas[dia_alta].isoformat(),
        })
        referencia.append({
            "codigo": codigo,
            "demanda_base_simulada": round(demanda_base, 3),
            "temporadas_simuladas": ", ".join(temporadas) if temporadas else "ninguna",
            "politica_reposicion_simulada": politica,
            "stock_inicial": stock_inicial,
            "dias_sin_stock": dias_sin_stock,
            "demanda_perdida_por_falta_de_stock": demanda_perdida,
        })

    return (pd.DataFrame(productos), pd.DataFrame(ventas),
            pd.DataFrame(compras), pd.DataFrame(referencia))


# ------------------------------------------------------------------
# 5. VALIDACIONES: comprobar que los datos son coherentes
# ------------------------------------------------------------------
def validar(productos, ventas, compras, referencia):
    errores = []

    if (productos["stock_actual"] < 0).any():
        errores.append("Hay productos con stock negativo.")
    if (ventas["cantidad"] <= 0).any():
        errores.append("Hay ventas con cantidad cero o negativa.")
    if (productos["precio_venta"] <= productos["precio_compra"]).any():
        errores.append("Hay productos que se venden a pérdida.")

    # Cuadre de inventario: stock inicial + compras recibidas - ventas = stock actual
    vendido = ventas.groupby("codigo")["cantidad"].sum()
    recibido = compras[compras["estado"] == "Recibido"].groupby("codigo")["cantidad"].sum()
    for _, fila in referencia.iterrows():
        c = fila["codigo"]
        esperado = fila["stock_inicial"] + recibido.get(c, 0) - vendido.get(c, 0)
        actual = productos.loc[productos["codigo"] == c, "stock_actual"].iloc[0]
        if esperado != actual:
            errores.append(f"El stock de {c} no cuadra ({esperado} vs {actual}).")

    return errores


# ------------------------------------------------------------------
# 6. PROGRAMA PRINCIPAL
# ------------------------------------------------------------------
def main():
    CARPETA_DATOS.mkdir(exist_ok=True)
    print("Generando datos FICTICIOS...")
    productos, ventas, compras, referencia = simular()

    errores = validar(productos, ventas, compras, referencia)
    if errores:
        print("\n❌ Se encontraron problemas:")
        for e in errores:
            print("   -", e)
        return

    # Ordenar las ventas por fecha (como se registrarían en caja) y renumerarlas
    ventas = ventas.sort_values(["fecha", "codigo"], kind="stable").reset_index(drop=True)
    ventas["id_venta"] = [f"V{i:06d}" for i in range(1, len(ventas) + 1)]
    compras = compras.sort_values(["fecha_pedido", "codigo"], kind="stable").reset_index(drop=True)
    compras["id_compra"] = [f"C{i:05d}" for i in range(1, len(compras) + 1)]

    # utf-8-sig: para que Excel muestre bien las tildes y la ñ
    productos.to_csv(CARPETA_DATOS / "productos.csv", index=False, encoding="utf-8-sig")
    ventas.to_csv(CARPETA_DATOS / "ventas.csv", index=False, encoding="utf-8-sig")
    compras.to_csv(CARPETA_DATOS / "compras.csv", index=False, encoding="utf-8-sig")
    referencia.to_csv(CARPETA_DATOS / "simulacion_referencia.csv", index=False, encoding="utf-8-sig")

    # Resumen en pantalla
    ventas["fecha"] = pd.to_datetime(ventas["fecha"])
    por_mes = ventas.groupby(ventas["fecha"].dt.to_period("M"))["total_venta"].sum()

    print("\n✅ Datos generados correctamente en la carpeta 'data'")
    print(f"   Periodo:            {FECHA_INICIO} a {FECHA_FIN}")
    print(f"   Productos:          {len(productos)}")
    print(f"   Registros de venta: {len(ventas):,}")
    print(f"   Unidades vendidas:  {ventas['cantidad'].sum():,}")
    print(f"   Ventas totales:     S/ {ventas['total_venta'].sum():,.2f}")
    print(f"   Pedidos de compra:  {len(compras)} "
          f"({(compras['estado'] == 'En tránsito').sum()} en tránsito)")
    print("   Cuadre de inventario: OK (stock inicial + compras - ventas = stock actual)")

    print("\n   Ventas por mes (S/):")
    for periodo, total in por_mes.items():
        barra = "█" * int(total / por_mes.max() * 30)
        print(f"   {periodo}  {total:>10,.0f}  {barra}")

    print("\n   RECUERDA: estos datos son FICTICIOS, solo sirven para probar el sistema.")


if __name__ == "__main__":
    main()
