"""
Clasificación de ROTACIÓN de productos con Machine Learning (K-Means).

Idea general:
1. Para cada producto se calculan 5 variables a partir de su historial.
2. K-Means agrupa los productos en 3 grupos según qué tan parecidos son.
   Nadie le dice al algoritmo qué producto es "alto" o "bajo": lo descubre
   comparando las variables.
3. Al final se le pone nombre a cada grupo: el grupo cuyo centro vende más
   unidades por día es "Alta rotación", el que menos, "Baja rotación".
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

VARIABLES = {
    "unidades_por_dia": "Unidades vendidas por día",
    "frecuencia_venta": "% de días con venta",
    "dias_entre_ventas": "Días promedio entre ventas",
    "dias_desde_ultima_venta": "Días desde la última venta",
    "cobertura_dias": "Días de inventario (cobertura)",
}
CLASES = ["Alta rotación", "Rotación media", "Baja rotación"]
TOPE_DIAS = 365      # límite para que valores extremos no distorsionen el modelo
HISTORIAL_CORTO = 60  # productos con menos días de historial se marcan


def calcular_variables(ventas, productos, fecha_inicio, fecha_fin):
    """Calcula las 5 variables de comportamiento de cada producto."""
    inicio, fin = pd.Timestamp(fecha_inicio), pd.Timestamp(fecha_fin)
    ventas = ventas[(ventas["fecha"] >= inicio) & (ventas["fecha"] <= fin)]
    filas = []

    for _, p in productos.iterrows():
        # Solo se cuentan los días en que el producto existía en la tienda
        alta = pd.Timestamp(p["fecha_alta"]) if "fecha_alta" in p and pd.notna(p["fecha_alta"]) else inicio
        if alta > fin:      # el producto aún no existía en el periodo elegido
            continue
        desde = max(inicio, alta)
        dias = max(1, (fin - desde).days + 1)

        v = ventas[ventas["codigo"] == p["codigo"]]
        dias_venta = np.sort(v["fecha"].unique())
        unidades = v["cantidad"].sum()
        por_dia = unidades / dias

        if len(dias_venta) >= 2:
            entre = np.diff(dias_venta).astype("timedelta64[D]").astype(int).mean()
        else:
            entre = dias
        ultima = (fin - pd.Timestamp(dias_venta[-1])).days if len(dias_venta) else dias
        cobertura = p["stock_actual"] / por_dia if por_dia > 0 else TOPE_DIAS

        filas.append({
            "codigo": p["codigo"],
            "producto": f"{p['nombre']} {p['talla']} {p['color']}",
            "categoria": p["categoria"],
            "dias_historial": dias,
            "unidades_vendidas": int(unidades),
            "unidades_por_dia": por_dia,
            "frecuencia_venta": len(dias_venta) / dias * 100,
            "dias_entre_ventas": min(entre, TOPE_DIAS),
            "dias_desde_ultima_venta": min(ultima, TOPE_DIAS),
            "cobertura_dias": min(cobertura, TOPE_DIAS),
            "stock_actual": int(p["stock_actual"]),
        })
    tabla = pd.DataFrame(filas)
    tabla["historial_corto"] = tabla["dias_historial"] < HISTORIAL_CORTO
    return tabla


def preparar_matriz(tabla):
    """
    Transforma las variables antes de K-Means:
    - Logaritmo: las ventas tienen valores muy dispares (0.05 vs 2 por día);
      el logaritmo los pone en una escala comparable.
    - Estandarización: cada variable queda con promedio 0 y desviación 1,
      para que ninguna pese más solo por estar en otra unidad de medida.
    """
    matriz = tabla[list(VARIABLES)].copy()
    for col in ["unidades_por_dia", "dias_entre_ventas", "dias_desde_ultima_venta", "cobertura_dias"]:
        matriz[col] = np.log1p(matriz[col])
    return StandardScaler().fit_transform(matriz)


def clasificar(tabla, semilla=42):
    """
    Entrena K-Means con 3 grupos y pone nombre a cada grupo.
    Devuelve (tabla_con_clase, calidad_silhouette, resumen_por_grupo).
    """
    if len(tabla) < 6:
        raise ValueError("Se necesitan al menos 6 productos para formar 3 grupos.")
    if tabla["unidades_vendidas"].sum() == 0:
        raise ValueError("No hay ventas en el periodo elegido.")

    matriz = preparar_matriz(tabla)
    modelo = KMeans(n_clusters=3, n_init=20, random_state=semilla)
    grupos = modelo.fit_predict(matriz)

    tabla = tabla.copy()
    tabla["grupo"] = grupos
    # Nombre de cada grupo según su promedio de unidades por día
    orden = tabla.groupby("grupo")["unidades_por_dia"].mean().sort_values(ascending=False).index
    nombres = {g: CLASES[i] for i, g in enumerate(orden)}
    tabla["rotacion"] = tabla["grupo"].map(nombres)

    # Distancia de cada producto al centro de su grupo (qué tan "típico" es)
    distancias = modelo.transform(matriz)
    tabla["distancia_centro"] = distancias[np.arange(len(tabla)), grupos]

    calidad = silhouette_score(matriz, grupos)

    resumen = (tabla.groupby("rotacion")
               .agg(productos=("codigo", "count"),
                    unidades_por_dia=("unidades_por_dia", "mean"),
                    frecuencia_venta=("frecuencia_venta", "mean"),
                    dias_entre_ventas=("dias_entre_ventas", "mean"),
                    cobertura_dias=("cobertura_dias", "median"),
                    unidades_vendidas=("unidades_vendidas", "sum"))
               .reindex(CLASES).dropna(subset=["productos"]).reset_index())
    resumen["participacion_ventas"] = resumen["unidades_vendidas"] / resumen["unidades_vendidas"].sum() * 100

    return tabla.drop(columns="grupo"), calidad, resumen


def interpretar_silhouette(valor):
    """Traduce el puntaje silhouette (-1 a 1) a palabras."""
    if valor >= 0.5:
        return "grupos bien separados"
    if valor >= 0.25:
        return "grupos razonablemente separados"
    return "grupos poco separados: la clasificación debe tomarse con cautela"
