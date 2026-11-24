"""Limpieza y validación del dataset Wine Quality (Red Wine).

Este módulo verifica valores faltantes, tipos numéricos, rango válido de la
variable objetivo (0 a 10) y elimina filas completamente duplicadas sin
modificar el archivo CSV original ni eliminar valores atípicos reales.
"""

from pathlib import Path
import pandas as pd

from utilidades import (
    COLUMNA_OBJETIVO,
    RUTA_DATASET_LIMPIO,
    RUTA_DATASET_ORIGINAL,
    RUTA_TABLAS,
    asegurar_carpetas_resultados,
    cargar_dataset,
    imprimir_encabezado,
    imprimir_subtitulo,
)


def limpiar_dataset(
    ruta_entrada: Path = RUTA_DATASET_ORIGINAL,
    ruta_salida: Path = RUTA_DATASET_LIMPIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limpia el dataset original, guarda el resultado y devuelve (datos_limpios, comparacion)."""
    imprimir_encabezado("LIMPIEZA Y VALIDACION DEL DATASET")

    # Protección para nunca sobrescribir el dataset original
    if ruta_salida.resolve() == ruta_entrada.resolve():
        raise ValueError(
            "La ruta de salida no puede ser igual a la ruta del dataset original."
        )

    datos_originales = cargar_dataset(ruta_entrada)

    # Cálculo dinámico de métricas antes de la limpieza
    registros_antes = len(datos_originales)
    columnas_antes = datos_originales.shape[1]
    faltantes_antes = int(datos_originales.isnull().sum().sum())
    duplicados_antes = int(datos_originales.duplicated().sum())

    # 1. Revisión de valores faltantes
    imprimir_subtitulo("1. Verificacion de valores faltantes")
    if faltantes_antes > 0:
        faltantes_col = datos_originales.isnull().sum()
        print("Se detectaron valores faltantes inesperados en las siguientes columnas:")
        print(faltantes_col[faltantes_col > 0])
        print(
            "Justificacion: No se aplica eliminacion ni imputacion automatica sin "
            "analizar primero la causa de los valores ausentes."
        )
    else:
        print("No se encontraron valores faltantes (0 valores nulos).")

    # 2. Comprobación de columnas numéricas
    imprimir_subtitulo("2. Comprobacion de tipos de datos numericos")
    columnas_no_numericas = [
        col
        for col in datos_originales.columns
        if not pd.api.types.is_numeric_dtype(datos_originales[col])
    ]
    if columnas_no_numericas:
        print(f"Advertencia: Columnas no numericas detectadas: {columnas_no_numericas}")
    else:
        print(
            f"Todas las {columnas_antes} columnas tienen tipo de dato numerico."
        )

    # 3. Verificación de escala válida de calidad (0 a 10)
    imprimir_subtitulo("3. Verificacion de escala de la variable 'quality' (0 a 10)")
    if COLUMNA_OBJETIVO not in datos_originales.columns:
        raise KeyError(
            f"No se encontro la columna objetivo '{COLUMNA_OBJETIVO}' en el dataset."
        )

    calidad_min = datos_originales[COLUMNA_OBJETIVO].min()
    calidad_max = datos_originales[COLUMNA_OBJETIVO].max()
    escala_valida = bool(datos_originales[COLUMNA_OBJETIVO].between(0, 10).all())
    print(f"Rango observado en 'quality': minimo = {calidad_min}, maximo = {calidad_max}")

    if escala_valida:
        print("Todas las puntuaciones de calidad estan dentro de la escala valida [0, 10].")
    else:
        invalidos = datos_originales[~datos_originales[COLUMNA_OBJETIVO].between(0, 10)]
        print(
            f"Advertencia: Se detectaron {len(invalidos)} registros fuera del rango [0, 10]. "
            "Se conservan para revision antes de decidir un tratamiento."
        )

    # 4. Eliminación de filas completamente duplicadas y reinicio de índices
    imprimir_subtitulo("4. Eliminacion de filas duplicadas")
    print(f"Registros originales: {registros_antes}")
    print(f"Registros completamente duplicados detectados: {duplicados_antes}")
    print(
        "\nDecision metodologica: Se eliminan las filas identicas para reducir la "
        "posibilidad de que observaciones iguales aparezcan tanto en el conjunto de "
        "entrenamiento como en el de prueba (fuga de datos), aunque en el origen "
        "fisicoquimico podrian corresponder a muestras distintas con mediciones iguales."
    )
    print(
        "Criterio sobre valores atipicos: No se eliminan automaticamente los valores "
        "atipicos (outliers), ya que pueden representar mediciones fisicoquimicas reales."
    )

    datos_limpios = datos_originales.drop_duplicates().reset_index(drop=True)

    # Cálculo dinámico de métricas después de la limpieza
    registros_despues = len(datos_limpios)
    columnas_despues = datos_limpios.shape[1]
    faltantes_despues = int(datos_limpios.isnull().sum().sum())
    duplicados_despues = int(datos_limpios.duplicated().sum())

    # 5. Validación posterior a la limpieza
    imprimir_subtitulo("5. Validacion posterior a la limpieza")
    print(f"Duplicados restantes despues de limpiar: {duplicados_despues}")
    print(
        f"Columnas numericas despues de limpiar: "
        f"{all(pd.api.types.is_numeric_dtype(datos_limpios[c]) for c in datos_limpios.columns)}"
    )
    print(
        f"Escala valida en 'quality' [0, 10]: "
        f"{bool(datos_limpios[COLUMNA_OBJETIVO].between(0, 10).all())}"
    )

    # 6. Tabla comparativa Antes vs Después
    comparacion = pd.DataFrame(
        {
            "Característica": [
                "Registros",
                "Columnas",
                "Valores faltantes",
                "Duplicados",
            ],
            "Antes": [
                registros_antes,
                columnas_antes,
                faltantes_antes,
                duplicados_antes,
            ],
            "Después": [
                registros_despues,
                columnas_despues,
                faltantes_despues,
                duplicados_despues,
            ],
        }
    )

    imprimir_subtitulo("6. Comparacion entre el dataset original y el limpio")
    print(comparacion.to_string(index=False))

    # Guardar dataset limpio sin sobrescribir el original y conservando sep=';'
    datos_limpios.to_csv(ruta_salida, sep=";", index=False)
    print(f"\nDataset limpio guardado correctamente en: {ruta_salida}")

    # Guardar tabla comparativa en resultados/tablas
    asegurar_carpetas_resultados()
    ruta_tabla_comparacion = RUTA_TABLAS / "comparacion_limpieza.csv"
    comparacion.to_csv(ruta_tabla_comparacion, index=False, encoding="utf-8-sig")
    print(f"Tabla comparativa guardada en: {ruta_tabla_comparacion}")

    return datos_limpios, comparacion


if __name__ == "__main__":
    try:
        limpiar_dataset()
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        print(f"Error durante la limpieza de datos: {error}")
