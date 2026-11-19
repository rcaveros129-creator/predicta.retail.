"""
ANÁLISIS DE TEMPORADAS Y ESTACIONALIDAD.

Idea clave: NO se asume que una temporada venda más. Para cada temporada se
comparan sus días con los días "normales" cercanos (4 semanas antes y 4 después)
y se mide si la diferencia es real o podría ser casualidad, con una prueba de
permutación:

  1. Se calcula la diferencia de venta diaria promedio: temporada vs. días normales.
  2. Se mezclan al azar las etiquetas "temporada" / "normal" 5 000 veces y se
     recalcula la diferencia cada vez.
  3. Si diferencias tan grandes como la real casi nunca aparecen al azar
     (menos del 5 % de las veces, p < 0.05), se concluye que el efecto es real.

Antes de comparar, cada día se ajusta por el día de la semana (un sábado vende
más que un lunes) para que la comparación sea justa.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from utils.calculations import DIAS_SEMANA, MESES

DIAS_COMPARACION = 28        # días normales antes y después de cada temporada
PERMUTACIONES = 5000
NIVEL_SIGNIFICANCIA = 0.05


def _domingo(anio, mes, numero):
    d = date(anio, mes, 1)
    while d.weekday() != 6:
        d += timedelta(days=1)
    return d + timedelta(weeks=numero - 1)


# Fechas de cada temporada comercial (calendario peruano). Devuelven (inicio, fin).
TEMPORADAS = {
    "Campaña escolar": lambda a: (date(a, 2, 1), date(a, 3, 15)),
    "Día de la Madre": lambda a: (_domingo(a, 5, 2) - timedelta(days=14), _domingo(a, 5, 2)),
    "Día del Padre": lambda a: (_domingo(a, 6, 3) - timedelta(days=14), _domingo(a, 6, 3)),
    "Fiestas Patrias": lambda a: (date(a, 7, 14), date(a, 7, 29)),
    "Navidad": lambda a: (date(a, 12, 10), date(a, 12, 24)),
    "Año Nuevo": lambda a: (date(a, 12, 26), date(a, 12, 31)),
}
ESTACIONES = {
    "Verano (dic–mar)": [12, 1, 2, 3],
    "Invierno (jun–ago)": [6, 7, 8],
}


def serie_diaria(ventas, columna="cantidad"):
    """Total diario (incluye días sin ventas como 0)."""
    fechas = pd.date_range(ventas["fecha"].min(), ventas["fecha"].max(), freq="D")
    return ventas.groupby("fecha")[columna].sum().reindex(fechas, fill_value=0)


def ajustar_por_dia_semana(serie):
    """Divide cada día entre el factor de su día de la semana (promedio del día / promedio general)."""
    dia = serie.index.day_of_week
    factor = serie.groupby(dia).mean() / serie.mean()
    factor = factor.replace(0, np.nan).fillna(1)
    return serie / factor.reindex(dia).values


def _dias_temporada(indice, inicio, fin):
    return (indice >= pd.Timestamp(inicio)) & (indice <= pd.Timestamp(fin))


def prueba_permutacion(grupo_a, grupo_b, semilla=42):
    """p-valor de dos colas para la diferencia de promedios entre dos grupos."""
    rng = np.random.default_rng(semilla)
    real = grupo_a.mean() - grupo_b.mean()
    todos = np.concatenate([grupo_a, grupo_b])
    n = len(grupo_a)
    extremos = 0
    for _ in range(PERMUTACIONES):
        rng.shuffle(todos)
        if abs(todos[:n].mean() - todos[n:].mean()) >= abs(real):
            extremos += 1
    return (extremos + 1) / (PERMUTACIONES + 1)


def analizar_temporadas(ventas):
    """
    Compara cada temporada comercial con los días normales cercanos.
    Devuelve una fila por temporada y año encontrado en los datos.
    """
    if ventas.empty:
        return pd.DataFrame()
    diario = serie_diaria(ventas)
    ajustada = ajustar_por_dia_semana(diario)
    indice = ajustada.index
    anios = sorted(set(indice.year))

    # Días que pertenecen a CUALQUIER temporada (no sirven como "días normales")
    en_alguna = pd.Series(False, index=indice)
    for fn in TEMPORADAS.values():
        for a in anios:
            en_alguna |= _dias_temporada(indice, *fn(a))

    filas = []
    for nombre, fn in TEMPORADAS.items():
        for a in anios:
            inicio, fin = fn(a)
            if pd.Timestamp(inicio) < indice[0] or pd.Timestamp(fin) > indice[-1]:
                continue   # la temporada no está completa en los datos
            dentro = _dias_temporada(indice, inicio, fin)
            cerca = _dias_temporada(indice, inicio - timedelta(days=DIAS_COMPARACION),
                                    fin + timedelta(days=DIAS_COMPARACION))
            normales = cerca & ~en_alguna
            if normales.sum() < 10:
                continue
            temporada, normal = ajustada[dentro].values, ajustada[normales].values
            diferencia = (temporada.mean() / normal.mean() - 1) * 100 if normal.mean() > 0 else np.nan
            p = prueba_permutacion(temporada, normal)
            filas.append({
                "temporada": nombre, "anio": a, "inicio": pd.Timestamp(inicio), "fin": pd.Timestamp(fin),
                "dias_temporada": int(dentro.sum()), "dias_normales": int(normales.sum()),
                "promedio_temporada": diario[dentro].mean(), "promedio_normal": diario[normales].mean(),
                "diferencia_pct": diferencia, "p_valor": p,
                "conclusion": conclusion(diferencia, p),
            })
    return pd.DataFrame(filas)


def conclusion(diferencia, p):
    if p >= NIVEL_SIGNIFICANCIA or np.isnan(diferencia):
        return "Sin efecto comprobado"
    return "Aumenta las ventas" if diferencia > 0 else "Disminuye las ventas"


def analizar_estaciones(ventas):
    """Compara verano e invierno contra el resto del año, por categoría (unidades por día)."""
    filas = []
    for categoria, grupo in [("Toda la tienda", ventas)] + list(ventas.groupby("categoria")):
        diario = serie_diaria(grupo) if not grupo.empty else None
        if diario is None:
            continue
        # Se usan semanas (no días) para que la prueba no exagere la seguridad
        semanal = diario.resample("W").mean()
        meses = semanal.index.month
        for nombre, lista in ESTACIONES.items():
            dentro = meses.isin(lista)
            if dentro.sum() < 4 or (~dentro).sum() < 4:
                continue
            a, b = semanal[dentro].values, semanal[~dentro].values
            diferencia = (a.mean() / b.mean() - 1) * 100 if b.mean() > 0 else np.nan
            p = prueba_permutacion(a, b)
            filas.append({"categoria": categoria, "estacion": nombre,
                          "unidades_dia_estacion": a.mean(), "unidades_dia_resto": b.mean(),
                          "diferencia_pct": diferencia, "p_valor": p, "conclusion": conclusion(diferencia, p)})
    return pd.DataFrame(filas)


def promedio_diario_por_mes(ventas):
    """Venta diaria promedio de cada mes (justo para meses incompletos)."""
    diario = serie_diaria(ventas, "total_venta")
    unidades = serie_diaria(ventas, "cantidad")
    tabla = pd.DataFrame({"soles": diario, "unidades": unidades})
    resumen = tabla.groupby(tabla.index.to_period("M")).agg(
        soles_dia=("soles", "mean"), unidades_dia=("unidades", "mean"),
        soles_total=("soles", "sum"), dias=("soles", "size")).reset_index(names="periodo")
    resumen["mes"] = [f"{MESES[p.month - 1]} {p.year}" for p in resumen["periodo"]]
    return resumen


def promedio_por_dia_semana(ventas):
    diario = serie_diaria(ventas)
    tabla = diario.groupby(diario.index.day_of_week).mean().reindex(range(7), fill_value=0)
    return pd.DataFrame({"dia": DIAS_SEMANA, "unidades_promedio": tabla.values})


def indice_categoria_mes(ventas):
    """
    Índice estacional: venta diaria de la categoría en cada mes ÷ su venta diaria del año.
    1.0 = mes normal; 1.5 = se vende 50 % más que su promedio; 0.5 = la mitad.
    """
    fechas = pd.date_range(ventas["fecha"].min(), ventas["fecha"].max(), freq="D")
    tabla = (ventas.pivot_table(index="fecha", columns="categoria", values="cantidad", aggfunc="sum")
             .reindex(fechas).fillna(0))
    mensual = tabla.groupby(tabla.index.to_period("M")).mean()
    indice = mensual / tabla.mean()
    indice.index = [f"{MESES[p.month - 1]} {str(p.year)[2:]}" for p in mensual.index]
    return indice.T


def productos_en_temporada(ventas, nombre, top=8):
    """Productos que más aumentan sus ventas durante una temporada o estación."""
    indice = pd.date_range(ventas["fecha"].min(), ventas["fecha"].max(), freq="D")
    dentro = pd.Series(False, index=indice)
    if nombre in TEMPORADAS:
        for a in set(indice.year):
            dentro |= _dias_temporada(indice, *TEMPORADAS[nombre](a))
    else:
        dentro = pd.Series(indice.month.isin(ESTACIONES[nombre]), index=indice)
    if dentro.sum() == 0:
        return pd.DataFrame()
    tabla = (ventas.pivot_table(index="fecha", columns="codigo", values="cantidad", aggfunc="sum")
             .reindex(indice).fillna(0))
    en_temp = tabla[dentro.values].mean()
    fuera = tabla[~dentro.values].mean()
    nombres = ventas.drop_duplicates("codigo").set_index("codigo")
    resultado = pd.DataFrame({
        "codigo": en_temp.index,
        "producto": [f"{nombres.loc[c, 'nombre']} {nombres.loc[c, 'talla']} {nombres.loc[c, 'color']}"
                     for c in en_temp.index],
        "categoria": [nombres.loc[c, "categoria"] for c in en_temp.index],
        "unidades_temporada": tabla[dentro.values].sum().values,
        "dia_en_temporada": en_temp.values, "dia_fuera": fuera.values})
    resultado["veces_normal"] = resultado["dia_en_temporada"] / resultado["dia_fuera"].replace(0, np.nan)
    resultado = resultado[resultado["unidades_temporada"] >= 3]
    return resultado.sort_values("veces_normal", ascending=False).head(top)


def comparar_semestres(ventas):
    """Cambios de comportamiento: primera mitad del periodo vs. segunda mitad, por categoría."""
    inicio, fin = ventas["fecha"].min(), ventas["fecha"].max()
    mitad = inicio + (fin - inicio) / 2
    primera = ventas[ventas["fecha"] <= mitad]
    segunda = ventas[ventas["fecha"] > mitad]
    dias1 = (mitad - inicio).days + 1
    dias2 = (fin - mitad).days
    t1 = primera.groupby("categoria")["cantidad"].sum() / dias1
    t2 = segunda.groupby("categoria")["cantidad"].sum() / dias2
    tabla = pd.DataFrame({"primera_mitad": t1, "segunda_mitad": t2}).fillna(0)
    tabla["cambio_pct"] = (tabla["segunda_mitad"] / tabla["primera_mitad"].replace(0, np.nan) - 1) * 100
    tabla["participacion_1"] = tabla["primera_mitad"] / tabla["primera_mitad"].sum() * 100
    tabla["participacion_2"] = tabla["segunda_mitad"] / tabla["segunda_mitad"].sum() * 100
    tabla = tabla.reset_index(names="categoria").sort_values("cambio_pct", ascending=False)
    total = (segunda["cantidad"].sum() / dias2) / (primera["cantidad"].sum() / dias1) * 100 - 100
    return tabla, total, (inicio, mitad, fin)
