"""Separación del dataset limpio en conjuntos de entrenamiento y prueba (80/20).

Carga winequality-red-limpio.csv, separa las variables X e y reutilizando
las funciones existentes, aplica train_test_split con semilla reproducible
(random_state=42), verifica la ausencia de superposición de índices y guarda
los conjuntos resultantes, la tabla resumen y dos gráficas a 300 DPI.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from Identificar_X_y_Y import separar_variables
from utilidades import (
    COLUMNA_OBJETIVO,
    COLUMNAS_INDEPENDIENTES,
    PROPORCION_PRUEBA,
    RUTA_CONFIG_MODELO,
    RUTA_DATASET_LIMPIO,
    RUTA_ESCALADOR_JOBLIB,
    RUTA_MODELO_JOBLIB,
    RUTA_TABLAS,
    RUTA_X_TEST,
    RUTA_X_TEST_ESCALADO,
    RUTA_X_TRAIN,
    RUTA_X_TRAIN_ESCALADO,
    RUTA_Y_TEST,
    RUTA_Y_TRAIN,
    SEMILLA_ALEATORIA,
    asegurar_carpetas_resultados,
    cargar_dataset,
    existe_dataset_limpio,
    guardar_y_mostrar_figura,
    imprimir_encabezado,
    imprimir_subtitulo,
    invalidar_archivos_desactualizados,
)


def graficar_particion_datos(
    cantidad_train: int,
    cantidad_test: int,
    mostrar: bool = True,
) -> Path:
    """Genera una gráfica de barras con la proporción entre entrenamiento y prueba."""
    sns.set_theme(style="whitegrid")
    total = cantidad_train + cantidad_test
    porc_train = (cantidad_train / total) * 100
    porc_test = (cantidad_test / total) * 100

    etiquetas = ["Entrenamiento (80%)", "Prueba (20%)"]
    cantidades = [cantidad_train, cantidad_test]
    porcentajes = [porc_train, porc_test]
    colores = ["#2b6cb0", "#dd6b20"]

    figura, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(
        etiquetas,
        cantidades,
        color=colores,
        edgecolor="#1a202c",
        linewidth=0.8,
        width=0.5,
    )

    for barra, cantidad, porcentaje in zip(barras, cantidades, porcentajes):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + total * 0.015,
            f"{cantidad:,} ({porcentaje:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title(
        "Proporción de datos en entrenamiento y prueba (Partición 80/20)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Conjunto de datos", fontsize=11)
    ax.set_ylabel("Cantidad de registros", fontsize=11)
    ax.set_ylim(0, max(cantidades) * 1.16)
    ax.grid(axis="x", visible=False)
    figura.tight_layout()

    return guardar_y_mostrar_figura(figura, "07_particion_datos.png", mostrar=mostrar)


def graficar_distribucion_train_test(
    y_train: pd.Series,
    y_test: pd.Series,
    mostrar: bool = True,
) -> Path:
    """Compara la distribución porcentual de la calidad entre entrenamiento y prueba."""
    sns.set_theme(style="whitegrid")
    calidades = sorted(set(y_train.unique()).union(set(y_test.unique())))

    dist_train = (
        y_train.value_counts(normalize=True).reindex(calidades, fill_value=0.0) * 100
    )
    dist_test = (
        y_test.value_counts(normalize=True).reindex(calidades, fill_value=0.0) * 100
    )

    posiciones = np.arange(len(calidades))
    ancho = 0.36

    figura, ax = plt.subplots(figsize=(9.5, 5.5))
    barras_train = ax.bar(
        posiciones - ancho / 2,
        dist_train.values,
        width=ancho,
        color="#2b6cb0",
        edgecolor="#1a365d",
        linewidth=0.7,
        label=f"Entrenamiento (n={len(y_train):,})",
    )
    barras_test = ax.bar(
        posiciones + ancho / 2,
        dist_test.values,
        width=ancho,
        color="#dd6b20",
        edgecolor="#7b341e",
        linewidth=0.7,
        label=f"Prueba (n={len(y_test):,})",
    )

    for barra in barras_train:
        altura = barra.get_height()
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            altura + 0.6,
            f"{altura:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    for barra in barras_test:
        altura = barra.get_height()
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            altura + 0.6,
            f"{altura:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_title(
        "Comparación de la distribución de calidad entre entrenamiento y prueba",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Puntuación de calidad (quality)", fontsize=11)
    ax.set_ylabel("Porcentaje de registros (%)", fontsize=11)
    ax.set_xticks(posiciones)
    ax.set_xticklabels([str(c) for c in calidades])
    ax.set_ylim(0, max(dist_train.max(), dist_test.max()) * 1.18)
    ax.legend()
    ax.grid(axis="x", visible=False)
    figura.tight_layout()

    return guardar_y_mostrar_figura(
        figura, "08_distribucion_train_test.png", mostrar=mostrar
    )


def particionar_datos(
    ruta_archivo: Path = RUTA_DATASET_LIMPIO,
    test_size: float = PROPORCION_PRUEBA,
    random_state: int = SEMILLA_ALEATORIA,
    mostrar_graficas: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide el dataset limpio en entrenamiento (80%) y prueba (20%) y guarda resultados."""
    imprimir_encabezado("FASE 2: PARTICION DE DATOS EN ENTRENAMIENTO Y PRUEBA")
    asegurar_carpetas_resultados()

    # 1. Cargar dataset limpio y verificar estructura real
    datos = cargar_dataset(ruta_archivo, separador=";")
    total_registros, total_columnas = datos.shape
    print(f"Archivo cargado: {ruta_archivo.name}")
    print(f"Dimensiones verificadas del dataset limpio: {datos.shape}")

    if list(datos.columns) != COLUMNAS_INDEPENDIENTES + [COLUMNA_OBJETIVO]:
        raise ValueError(
            "Las columnas del dataset limpio no coinciden con el orden esperado."
        )

    # 2. Separar variables X e y conservando la variable objetivo numérica (regresión)
    X, y = separar_variables(datos)
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("La variable objetivo 'quality' debe ser numerica.")

    # 3. Aplicar train_test_split (80% entrenamiento, 20% prueba)
    imprimir_subtitulo("1. Configuracion y justificacion de la particion")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    print(f"Proporcion configurada: entrenamiento = {(1 - test_size)*100:.0f}%, prueba = {test_size*100:.0f}%")
    print(f"Semilla aleatoria: random_state={random_state}, shuffle=True")
    print(
        "\nJustificacion de particion aleatoria: El dataset no contiene una variable "
        "temporal ni identificadores de grupos o lotes que exijan una particion cronologica o agrupada."
    )
    print(
        f"Justificacion de reproducibilidad: Establecer random_state={random_state} "
        "garantiza obtener exactamente la misma division en futuras ejecuciones."
    )

    # 4. Verificar dimensiones, porcentajes reales e índices disjuntos
    imprimir_subtitulo("2. Verificacion de cantidades, dimensiones e indices")
    registros_train = len(X_train)
    registros_test = len(X_test)
    porcentaje_train = (registros_train / total_registros) * 100
    porcentaje_test = (registros_test / total_registros) * 100

    indices_compartidos = set(X_train.index).intersection(set(X_test.index))
    if indices_compartidos:
        raise ValueError(
            f"Se detectaron {len(indices_compartidos)} indices repetidos entre entrenamiento y prueba."
        )
    if registros_train + registros_test != total_registros:
        raise ValueError("La suma de entrenamiento y prueba no coincide con el total.")
    if not (X_train.index.equals(y_train.index) and X_test.index.equals(y_test.index)):
        raise ValueError("Los indices entre X e y no coinciden tras la particion.")

    print(f"Cantidad total de registros:          {total_registros}")
    print(f"Cantidad de registros entrenamiento:  {registros_train}")
    print(f"Cantidad de registros prueba:         {registros_test}")
    print(f"Porcentaje real de entrenamiento:     {porcentaje_train:.2f}%")
    print(f"Porcentaje real de prueba:            {porcentaje_test:.2f}%")
    print(f"Dimensiones de X_train:               {X_train.shape}")
    print(f"Dimensiones de X_test:                {X_test.shape}")
    print(f"Dimensiones de y_train:               {y_train.shape}")
    print(f"Dimensiones de y_test:                {y_test.shape}")
    print(
        f"Indices compartidos entre train/test: {len(indices_compartidos)} "
        "(sin superposicion verificada)"
    )

    # 5. Guardar archivos en resultados/datos_procesados/ conservando columnas e índices
    imprimir_subtitulo("3. Almacenamiento de datos procesados y tabla resumen")
    X_train.to_csv(RUTA_X_TRAIN, index=True, index_label="indice")
    X_test.to_csv(RUTA_X_TEST, index=True, index_label="indice")
    y_train.to_frame(name=COLUMNA_OBJETIVO).to_csv(
        RUTA_Y_TRAIN, index=True, index_label="indice"
    )
    y_test.to_frame(name=COLUMNA_OBJETIVO).to_csv(
        RUTA_Y_TEST, index=True, index_label="indice"
    )

    print(f"Guardado: {RUTA_X_TRAIN}")
    print(f"Guardado: {RUTA_X_TEST}")
    print(f"Guardado: {RUTA_Y_TRAIN}")
    print(f"Guardado: {RUTA_Y_TEST}")

    # Invalidar archivos escalados/modelos antiguos si se vuelve a particionar
    invalidar_archivos_desactualizados(
        [
            RUTA_X_TRAIN_ESCALADO,
            RUTA_X_TEST_ESCALADO,
            RUTA_ESCALADOR_JOBLIB,
            RUTA_MODELO_JOBLIB,
            RUTA_CONFIG_MODELO,
        ],
        motivo="nueva particion de datos",
    )

    resumen_particion = pd.DataFrame(
        {
            "Conjunto": ["Entrenamiento", "Prueba", "Total"],
            "Registros": [registros_train, registros_test, total_registros],
            "Porcentaje (%)": [
                round(porcentaje_train, 2),
                round(porcentaje_test, 2),
                100.0,
            ],
            "Columnas X": [X_train.shape[1], X_test.shape[1], X.shape[1]],
            "Dimensiones X": [str(X_train.shape), str(X_test.shape), str(X.shape)],
            "Dimensiones y": [str(y_train.shape), str(y_test.shape), str(y.shape)],
            "Semilla aleatoria": [random_state, random_state, random_state],
        }
    )

    ruta_resumen = RUTA_TABLAS / "resumen_particion.csv"
    resumen_particion.to_csv(ruta_resumen, index=False, encoding="utf-8-sig")
    print(f"Tabla resumen guardada en: {ruta_resumen}")

    # 6. Generar gráficas de partición (300 DPI)
    imprimir_subtitulo("4. Generacion de graficas de la particion")
    graficar_particion_datos(registros_train, registros_test, mostrar=mostrar_graficas)
    graficar_distribucion_train_test(y_train, y_test, mostrar=mostrar_graficas)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    try:
        if not existe_dataset_limpio():
            print(
                "Error: Aun no existe 'winequality-red-limpio.csv'. "
                "Ejecute primero 'Limpieza_Datos.py' para generar el dataset limpio."
            )
        else:
            particionar_datos()
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        print(f"Error durante la particion de datos: {error}")
