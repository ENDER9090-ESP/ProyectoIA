"""Exploración inicial del dataset Wine Quality (Red Wine).

Este módulo carga el dataset original (winequality-red.csv) e imprime
un resumen completo en consola para comprender la estructura, tipos de datos,
valores faltantes, registros duplicados y distribución de la calidad.
"""

from pathlib import Path
import pandas as pd

from utilidades import (
    COLUMNA_OBJETIVO,
    RUTA_DATASET_ORIGINAL,
    cargar_dataset,
    imprimir_encabezado,
    imprimir_subtitulo,
)


def explorar_dataset(ruta_archivo: Path = RUTA_DATASET_ORIGINAL) -> pd.DataFrame:
    """Realiza la exploración inicial del dataset original y devuelve el DataFrame."""
    imprimir_encabezado("EXPLORACION INICIAL DEL DATASET ORIGINAL")
    datos = cargar_dataset(ruta_archivo)
    print(f"Archivo cargado: {ruta_archivo.name}")

    # 1. Primeros cinco registros
    imprimir_subtitulo("1. Primeros cinco registros (head)")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(datos.head())

    # 2. Últimos cinco registros
    imprimir_subtitulo("2. Ultimos cinco registros (tail)")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(datos.tail())

    # 3. Dimensiones del dataset
    imprimir_subtitulo("3. Dimensiones del dataset (shape)")
    filas, columnas = datos.shape
    print(f"Shape: {datos.shape} -> {filas} registros y {columnas} columnas")

    # 4. Nombres de todas las columnas
    imprimir_subtitulo("4. Nombres de todas las columnas")
    for indice, columna in enumerate(datos.columns.tolist(), start=1):
        print(f"  {indice:2d}. {columna}")

    # 5. Información general del DataFrame
    imprimir_subtitulo("5. Informacion general (info)")
    datos.info()

    # 6. Tipos de datos por columna
    imprimir_subtitulo("6. Tipos de datos (dtypes)")
    print(datos.dtypes)

    # 7. Valores faltantes por columna y total
    imprimir_subtitulo("7. Valores faltantes (isnull)")
    faltantes_por_columna = datos.isnull().sum()
    total_faltantes = int(faltantes_por_columna.sum())
    print(faltantes_por_columna)
    print(f"\nTotal de valores faltantes en el dataset: {total_faltantes}")

    # 8. Registros duplicados y ejemplos
    imprimir_subtitulo("8. Registros duplicados (duplicated)")
    cantidad_duplicados = int(datos.duplicated().sum())
    print(f"Cantidad de filas completamente duplicadas: {cantidad_duplicados}")

    if cantidad_duplicados > 0:
        # Se muestran pares repetidos ordenados para visualizar ejemplos idénticos
        ejemplos_duplicados = (
            datos[datos.duplicated(keep=False)]
            .sort_values(by=list(datos.columns))
            .head(10)
        )
        print("\nEjemplos de registros duplicados (primeras 10 filas repetidas):")
        with pd.option_context("display.max_columns", None, "display.width", 1000):
            print(ejemplos_duplicados)
    else:
        print("No se encontraron registros duplicados.")

    # 9. Estadísticas descriptivas
    imprimir_subtitulo("9. Estadisticas descriptivas (describe)")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(datos.describe().round(4))

    # 10. Distribución de la variable objetivo quality
    imprimir_subtitulo("10. Distribucion de puntuaciones de calidad (value_counts)")
    if COLUMNA_OBJETIVO in datos.columns:
        distribucion = datos[COLUMNA_OBJETIVO].value_counts().sort_index()
        for calidad, cantidad in distribucion.items():
            porcentaje = (cantidad / len(datos)) * 100
            print(f"  Calidad {calidad}: {cantidad:4d} registros ({porcentaje:5.2f}%)")
    else:
        print(f"Advertencia: No se encontro la columna '{COLUMNA_OBJETIVO}'.")

    return datos


if __name__ == "__main__":
    try:
        explorar_dataset()
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error durante la exploracion de datos: {error}")
