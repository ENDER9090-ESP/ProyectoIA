"""Identificación de variables independientes (X) y variable dependiente (y).

Define las 11 variables fisicoquímicas de entrada (X) y la variable objetivo
numérica (y = quality) para el problema de aprendizaje supervisado de regresión.
"""

from pathlib import Path
import pandas as pd

from utilidades import (
    COLUMNA_OBJETIVO,
    RUTA_DATASET_LIMPIO,
    RUTA_DATASET_ORIGINAL,
    cargar_dataset,
    existe_dataset_limpio,
    imprimir_encabezado,
    imprimir_subtitulo,
)


def separar_variables(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa el DataFrame en variables independientes X y variable objetivo y."""
    if COLUMNA_OBJETIVO not in datos.columns:
        raise KeyError(
            f"No se encontro la columna objetivo '{COLUMNA_OBJETIVO}' en el dataset."
        )

    # Separación requerida para regresión (conservando 'quality' como valor numérico)
    X = datos.drop(columns=["quality"])
    y = datos["quality"]
    return X, y


def identificar_variables(
    ruta_archivo: Path = RUTA_DATASET_LIMPIO,
) -> tuple[pd.DataFrame, pd.Series]:
    """Carga el dataset indicado, separa X e y y muestra el resumen en consola."""
    imprimir_encabezado("IDENTIFICACION DE VARIABLES X Y y (REGRESION)")

    datos = cargar_dataset(ruta_archivo)
    print(f"Archivo cargado: {ruta_archivo.name}")
    print(f"Dimensiones totales del dataset: {datos.shape}")

    X, y = separar_variables(datos)

    # Mostrar las columnas correspondientes a X y la variable y
    imprimir_subtitulo("1. Columnas correspondientes a X y variable objetivo y")
    print("Variables independientes (X):")
    for indice, columna in enumerate(X.columns.tolist(), start=1):
        print(f"  {indice:2d}. {columna}")

    print(
        f"\nVariable dependiente (y): {y.name} "
        f"(tipo: {y.dtype}, problema de regresion)"
    )

    # Primeros cinco registros de X
    imprimir_subtitulo("2. Primeros cinco registros de X")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(X.head())

    # Primeros cinco registros de y
    imprimir_subtitulo("3. Primeros cinco registros de y")
    print(y.head())

    # Dimensiones de X y y
    imprimir_subtitulo("4. Dimensiones de X y y")
    print(f"Dimensiones de X: {X.shape}")
    print(f"Dimensiones de y: {y.shape}")

    return X, y


if __name__ == "__main__":
    try:
        if existe_dataset_limpio():
            identificar_variables(RUTA_DATASET_LIMPIO)
        else:
            print(
                "Nota: 'winequality-red-limpio.csv' aun no existe. "
                "Se usara temporalmente el dataset original 'winequality-red.csv'."
            )
            identificar_variables(RUTA_DATASET_ORIGINAL)
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        print(f"Error al identificar variables X y y: {error}")
