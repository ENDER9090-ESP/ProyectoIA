"""Estandarización de las variables independientes mediante StandardScaler.

Ajusta StandardScaler exclusivamente sobre el conjunto de entrenamiento (X_train)
para evitar la fuga de datos (data leakage), transforma X_train y X_test sin
alterar la variable objetivo y, verifica media ≈ 0 y desviación estándar ≈ 1
(ddof=0) en entrenamiento, y genera tablas y gráficas comparativas a 300 DPI.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from utilidades import (
    COLUMNAS_INDEPENDIENTES,
    RUTA_CONFIG_MODELO,
    RUTA_ESCALADOR_JOBLIB,
    RUTA_MODELO_JOBLIB,
    RUTA_TABLAS,
    RUTA_X_TEST,
    RUTA_X_TEST_ESCALADO,
    RUTA_X_TRAIN,
    RUTA_X_TRAIN_ESCALADO,
    asegurar_carpeta_modelos,
    asegurar_carpetas_resultados,
    cargar_tabla_procesada,
    datos_particion_actualizados,
    existen_datos_particion,
    guardar_y_mostrar_figura,
    imprimir_encabezado,
    imprimir_subtitulo,
    invalidar_archivos_desactualizados,
)


def graficar_variables_sin_escalar(
    X_train: pd.DataFrame,
    mostrar: bool = True,
) -> Path:
    """Genera diagramas de caja horizontales de X_train antes del escalado."""
    sns.set_theme(style="whitegrid")

    figura, ax = plt.subplots(figsize=(11, 6.5))
    sns.boxplot(data=X_train, orient="h", color="#4299e1", ax=ax)

    ax.set_title(
        "Distribución de las variables de entrenamiento antes del escalado",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Valor en escala original", fontsize=11)
    ax.set_ylabel("Variable fisicoquímica", fontsize=11)
    figura.tight_layout()

    return guardar_y_mostrar_figura(
        figura, "09_variables_sin_escalar.png", mostrar=mostrar
    )


def graficar_variables_escaladas(
    X_train_escalado: pd.DataFrame,
    mostrar: bool = True,
) -> Path:
    """Genera diagramas de caja horizontales de X_train después del escalado."""
    sns.set_theme(style="whitegrid")

    figura, ax = plt.subplots(figsize=(11, 6.5))
    sns.boxplot(data=X_train_escalado, orient="h", color="#2b6cb0", ax=ax)
    ax.axvline(0, color="#c53030", linestyle="--", linewidth=1.2, label="Media = 0")

    ax.set_title(
        "Distribución de las variables de entrenamiento después del escalado (StandardScaler)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Valor estandarizado (z)", fontsize=11)
    ax.set_ylabel("Variable fisicoquímica", fontsize=11)
    ax.legend(loc="lower right")
    figura.tight_layout()

    return guardar_y_mostrar_figura(
        figura, "10_variables_escaladas.png", mostrar=mostrar
    )


def graficar_comparacion_escalado(
    tabla_comparacion: pd.DataFrame,
    mostrar: bool = True,
) -> Path:
    """Compara medias y desviaciones estándar (ddof=0) antes y después en X_train."""
    sns.set_theme(style="whitegrid")
    variables = tabla_comparacion["Variable"].tolist()
    posiciones = np.arange(len(variables))

    figura, ejes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Media antes del escalado
    ax_m_antes = ejes[0, 0]
    medias_antes = tabla_comparacion["Media antes (train)"].values
    ax_m_antes.barh(posiciones, medias_antes, color="#4299e1")
    ax_m_antes.set_yticks(posiciones)
    ax_m_antes.set_yticklabels(variables)
    ax_m_antes.invert_yaxis()
    ax_m_antes.set_title("Media antes del escalado (X_train)", fontsize=11, fontweight="bold")
    ax_m_antes.set_xlabel("Media original", fontsize=10)
    for pos, val in zip(posiciones, medias_antes):
        ax_m_antes.text(val + max(medias_antes) * 0.02, pos, f"{val:.2f}", va="center", fontsize=8.5)
    ax_m_antes.set_xlim(0, max(medias_antes) * 1.18)

    # 2. Media después del escalado
    ax_m_despues = ejes[0, 1]
    medias_despues = tabla_comparacion["Media después (train)"].values
    ax_m_despues.barh(posiciones, medias_despues, color="#2b6cb0")
    ax_m_despues.axvline(0, color="#c53030", linestyle="--", linewidth=1.2)
    ax_m_despues.set_yticks(posiciones)
    ax_m_despues.set_yticklabels(variables)
    ax_m_despues.invert_yaxis()
    ax_m_despues.set_xlim(-1.0, 1.0)
    ax_m_despues.set_title("Media después del escalado (X_train)", fontsize=11, fontweight="bold")
    ax_m_despues.set_xlabel("Media estandarizada (≈ 0)", fontsize=10)
    for pos, val in zip(posiciones, medias_despues):
        ax_m_despues.text(0.06, pos, f"{abs(val):.4f}", va="center", fontsize=8.5, color="#1a202c")

    # 3. Desviación estándar antes del escalado
    ax_d_antes = ejes[1, 0]
    desv_antes = tabla_comparacion["Desv. est. antes (train)"].values
    ax_d_antes.barh(posiciones, desv_antes, color="#ed8936")
    ax_d_antes.set_yticks(posiciones)
    ax_d_antes.set_yticklabels(variables)
    ax_d_antes.invert_yaxis()
    ax_d_antes.set_title(
        "Desviación estándar (ddof=0) antes del escalado", fontsize=11, fontweight="bold"
    )
    ax_d_antes.set_xlabel("Desviación estándar original", fontsize=10)
    for pos, val in zip(posiciones, desv_antes):
        ax_d_antes.text(val + max(desv_antes) * 0.02, pos, f"{val:.2f}", va="center", fontsize=8.5)
    ax_d_antes.set_xlim(0, max(desv_antes) * 1.18)

    # 4. Desviación estándar después del escalado
    ax_d_despues = ejes[1, 1]
    desv_despues = tabla_comparacion["Desv. est. después (train)"].values
    ax_d_despues.barh(posiciones, desv_despues, color="#2f855a")
    ax_d_despues.axvline(1.0, color="#c53030", linestyle="--", linewidth=1.2)
    ax_d_despues.set_yticks(posiciones)
    ax_d_despues.set_yticklabels(variables)
    ax_d_despues.invert_yaxis()
    ax_d_despues.set_xlim(0, 1.3)
    ax_d_despues.set_title(
        "Desviación estándar (ddof=0) después del escalado", fontsize=11, fontweight="bold"
    )
    ax_d_despues.set_xlabel("Desviación estándar estandarizada (= 1)", fontsize=10)
    for pos, val in zip(posiciones, desv_despues):
        ax_d_despues.text(val + 0.03, pos, f"{val:.4f}", va="center", fontsize=8.5)

    figura.suptitle(
        "Comparación de medias y desviaciones estándar en entrenamiento antes y después del escalado",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    figura.tight_layout()

    return guardar_y_mostrar_figura(
        figura, "11_comparacion_escalado.png", mostrar=mostrar
    )


def escalar_datos(
    mostrar_graficas: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Aplica StandardScaler sin fuga de datos, guarda resultados y genera gráficas."""
    imprimir_encabezado("FASE 2: ESCALADO DE VARIABLES CON STANDARDSCALER")
    asegurar_carpetas_resultados()
    asegurar_carpeta_modelos()

    if not existen_datos_particion():
        raise FileNotFoundError(
            "No se encontraron los archivos de particion (X_train.csv, X_test.csv). "
            "Ejecute primero 'Particion_Datos.py'."
        )
    if not datos_particion_actualizados():
        raise ValueError(
            "Los archivos de particion estan desactualizados frente a 'winequality-red-limpio.csv'. "
            "Vuelva a ejecutar 'Particion_Datos.py'."
        )

    # 1. Cargar X_train y X_test conservando columnas e índices originales
    X_train = cargar_tabla_procesada(RUTA_X_TRAIN)
    X_test = cargar_tabla_procesada(RUTA_X_TEST)

    if list(X_train.columns) != COLUMNAS_INDEPENDIENTES or list(X_test.columns) != COLUMNAS_INDEPENDIENTES:
        raise ValueError(
            "Las columnas de X_train o X_test no coinciden con las 11 variables esperadas."
        )
    if set(X_train.index).intersection(set(X_test.index)):
        raise ValueError("Se detecto superposicion de indices entre X_train y X_test.")

    # 2. Justificación académica y regla contra fuga de datos
    imprimir_subtitulo("1. Justificacion y control contra fuga de datos")
    print(
        "Las variables fisicoquimicas tienen magnitudes y unidades muy distintas "
        "(por ejemplo, 'pH' y 'density' toman valores pequenos cercanos a 1-3, "
        "'alcohol' representa porcentaje y 'total sulfur dioxide' alcanza decenas/centenas)."
    )
    print("Formula de estandarizacion aplicada: z = (x - media) / desviacion_estandar")
    print(
        "Nota metodologica: La regresion lineal ordinaria no requiere obligatoriamente "
        "el escalado para generar predicciones, pero se aplica por requisito academico "
        "y para comparar directamente la magnitud de los coeficientes estandarizados."
    )
    print(
        "Prevencion de fuga de datos: StandardScaler se ajusta UNICAMENTE con X_train "
        "(fit_transform en X_train, transform en X_test). Nunca se ajusta sobre X_test "
        "ni sobre el dataset completo, y la variable 'quality' (y) conserva su escala original."
    )

    # 3. Ajuste exclusivo en X_train y transformación de X_train y X_test
    escalador = StandardScaler()
    X_train_escalado = pd.DataFrame(
        escalador.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_escalado = pd.DataFrame(
        escalador.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    # 4. Verificación en consola antes y después del escalado
    imprimir_subtitulo("2. Primeros cinco registros de X_train antes del escalado")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(X_train.head().round(4))

    imprimir_subtitulo("3. Primeros cinco registros de X_train despues del escalado")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(X_train_escalado.head().round(4))

    medias_antes = X_train.mean()
    desv_antes = X_train.std(ddof=0)
    medias_despues = X_train_escalado.mean()
    desv_despues = X_train_escalado.std(ddof=0)

    tabla_comparacion = pd.DataFrame(
        {
            "Variable": X_train.columns,
            "Media antes (train)": medias_antes.values,
            "Desv. est. antes (train)": desv_antes.values,
            "Media después (train)": medias_despues.values,
            "Desv. est. después (train)": desv_despues.values,
        }
    )

    imprimir_subtitulo("4. Medias y desviaciones estandar (ddof=0) en X_train")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(tabla_comparacion.round(6).to_string(index=False))

    # Comprobar media ≈ 0 y desviación estándar poblacional ≈ 1 en variables no constantes
    columnas_no_constantes = desv_antes[desv_antes > 0].index.tolist()
    medias_ok = bool(np.allclose(medias_despues[columnas_no_constantes], 0.0, atol=1e-9))
    desv_ok = bool(np.allclose(desv_despues[columnas_no_constantes], 1.0, atol=1e-9))

    if not (medias_ok and desv_ok):
        raise ValueError(
            "La verificacion del escalado en X_train no arrojo media ≈ 0 y std(ddof=0) ≈ 1."
        )

    imprimir_subtitulo("5. Dimensiones y validacion final del escalado")
    print(f"Dimensiones de X_train_escalado: {X_train_escalado.shape}")
    print(f"Dimensiones de X_test_escalado:  {X_test_escalado.shape}")
    print(
        f"Verificacion en X_train (variables no constantes): "
        f"media ≈ 0 ({medias_ok}), std(ddof=0) ≈ 1 ({desv_ok})."
    )
    print(
        "Nota sobre X_test_escalado: No se exige media 0 ni desviacion 1 en prueba, "
        "ya que fue transformado con los parametros aprendidos exclusivamente en X_train."
    )

    # 5. Guardar conjuntos escalados, parámetros aprendidos y el escalador
    imprimir_subtitulo("6. Almacenamiento de datos escalados, tablas y escalador")
    X_train_escalado.to_csv(RUTA_X_TRAIN_ESCALADO, index=True, index_label="indice")
    X_test_escalado.to_csv(RUTA_X_TEST_ESCALADO, index=True, index_label="indice")
    print(f"Guardado: {RUTA_X_TRAIN_ESCALADO}")
    print(f"Guardado: {RUTA_X_TEST_ESCALADO}")

    joblib.dump(escalador, RUTA_ESCALADOR_JOBLIB)
    print(f"Escalador guardado en: {RUTA_ESCALADOR_JOBLIB}")

    # Invalidar modelo entrenado anterior si se vuelve a ejecutar el escalado
    invalidar_archivos_desactualizados(
        [RUTA_MODELO_JOBLIB, RUTA_CONFIG_MODELO],
        motivo="nuevo escalado de datos",
    )

    ruta_comparacion = RUTA_TABLAS / "comparacion_escalado.csv"
    tabla_comparacion.to_csv(ruta_comparacion, index=False, encoding="utf-8-sig")
    print(f"Tabla de comparacion guardada en: {ruta_comparacion}")

    tabla_parametros = pd.DataFrame(
        {
            "Variable": X_train.columns,
            "Media aprendida (mean_)": escalador.mean_,
            "Desv. est. aprendida (scale_)": escalador.scale_,
            "Varianza aprendida (var_)": escalador.var_,
        }
    )
    ruta_parametros = RUTA_TABLAS / "parametros_escalador.csv"
    tabla_parametros.to_csv(ruta_parametros, index=False, encoding="utf-8-sig")
    print(f"Parametros de StandardScaler guardados en: {ruta_parametros}")

    # 6. Gráficas comparativas usando únicamente datos de entrenamiento (300 DPI)
    imprimir_subtitulo("7. Generacion de graficas comparativas (solo X_train)")
    graficar_variables_sin_escalar(X_train, mostrar=mostrar_graficas)
    graficar_variables_escaladas(X_train_escalado, mostrar=mostrar_graficas)
    graficar_comparacion_escalado(tabla_comparacion, mostrar=mostrar_graficas)

    return X_train_escalado, X_test_escalado, escalador


if __name__ == "__main__":
    try:
        if not existen_datos_particion():
            print(
                "Error: Aun no existen los archivos de particion. "
                "Ejecute primero 'Particion_Datos.py'."
            )
        else:
            escalar_datos()
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        print(f"Error durante el escalado de datos: {error}")
