# Predicta Retail

Sistema web de gestión de inventarios y predicción de demanda con Machine Learning para
microempresas de venta de ropa. Construido con Python y Streamlit.

> ⚠️ Los datos incluidos son **ficticios**, generados por `generar_datos.py` para probar el sistema.

## Secciones

| Sección | Qué hace | Método |
|---|---|---|
| Panel de control | Indicadores, gráficos y alertas | Análisis de datos |
| Productos | Catálogo, ficha integrada e importación de CSV/Excel con validaciones | — |
| Ventas y temporadas | Meses y días con más ventas, efecto de cada temporada comercial | Prueba de permutación |
| Rotación de inventario | Clasifica productos en alta, media y baja rotación | K-Means |
| Predicción de demanda | Unidades a vender en 7, 15, 30, 60 y 90 días, con su error | Random Forest + modelos de referencia, evaluación temporal |
| Control de stock | Días hasta agotarse, stock de seguridad, punto de pedido y estado | Demanda predicha |
| Abastecimiento inteligente | Qué comprar, cuánto y cuándo; productos con exceso | Proyección del inventario |
| Reportes | Descarga en Excel (una hoja por reporte) o CSV | — |

## Ejecutar en tu computadora

```bash
python -m venv venv
venv\Scripts\activate          # En Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Para verificar todo el sistema: `python verificar_sistema.py`

Para regenerar los datos ficticios: `python generar_datos.py`

## Estructura

```
app.py                 Menú, filtros y navegación
generar_datos.py       Crea los datos ficticios de prueba
verificar_sistema.py   Prueba completa del sistema
data/                  productos.csv, ventas.csv, compras.csv
models/                Modelo de demanda entrenado (se crea solo)
modules/               Una página por sección del menú
utils/                 Carga de datos, modelos y cálculos
```

## Requisitos

Python 3.13 y las librerías de `requirements.txt`.
