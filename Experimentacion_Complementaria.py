"""Experimentación complementaria: regularización Ridge con validación cruzada (5-fold).

Evalúa exclusivamente sobre el conjunto de entrenamiento original sin escalar
(X_train, y_train) el desempeño de Ridge para distintos valores de alpha
(0.01, 0.1, 1, 10 y 100) y de LinearRegression como referencia, encapsulando
StandardScaler dentro de un Pipeline de scikit-learn para ajustar el escalado
de forma independiente en cada partición y evitar fuga de datos.

Este experimento es estrictamente complementario: nunca utiliza el conjunto de
prueba reservado ni sustituye o modifica el modelo principal guardado en modelos/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utilidades import (
    COLUMNA_OBJETIVO,
    COLUMNAS_INDEPENDIENTES,
    RUTA_GRAFICA_RIDGE_CV,
    RUTA_TABLA_RIDGE_CV,
    RUTA_X_TRAIN,
    RUTA_Y_TRAIN,
    SEMILLA_ALEATORIA,
    asegurar_carpetas_resultados,
    cargar_tabla_procesada,
    datos_particion_actualizados,
    existen_datos_particion,
    guardar_y_mostrar_figura,
    imprimir_encabezado,
    imprimir_subtitulo,
)

ALPHAS_RIDGE = [0.01, 0.1, 1, 10, 100]
N_PARTICIONES_CV = 5


def _evaluar_pipeline_cv(
    nombre_modelo: str,
    etiqueta_alpha: str,
    estimador,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: KFold,
) -> dict:
    """Evalúa un estimador dentro de Pipeline(StandardScaler -> modelo) con KFold."""
    flujo = Pipeline(
        [
            ("escalador", StandardScaler()),
            ("modelo", estimador),
        ]
    )
    resultados_cv = cross_validate(
        flujo,
        X_train,
        y_train,
        cv=cv,
        scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
    )
    rmses = -np.asarray(resultados_cv["test_rmse"], dtype=float)
    r2s = np.asarray(resultados_cv["test_r2"], dtype=float)

    return {
        "Modelo": nombre_modelo,
        "Alpha": etiqueta_alpha,
        "RMSE_promedio_CV": float(np.mean(rmses)),
        "RMSE_desv_est_CV": float(np.std(rmses, ddof=1)),
        "R2_promedio_CV": float(np.mean(r2s)),
        "R2_desv_est_CV": float(np.std(r2s, ddof=1)),
        "Particiones_CV": int(cv.get_n_splits()),
        "Registros_entrenamiento": int(len(X_train)),
    }


def graficar_comparacion_ridge_cv(
    tabla_resultados: pd.DataFrame,
    mostrar: bool = True,
) -> Path:
    """Genera gráfica de RMSE y R² promedio ± DE en CV 5-fold frente a LinearRegression."""
    sns.set_theme(style="whitegrid")
    ref = tabla_resultados[tabla_resultados["Modelo"] == "LinearRegression"].iloc[0]
    df_ridge = tabla_resultados[tabla_resultados["Modelo"] == "Ridge"].copy()
    alphas = df_ridge["Alpha"].astype(float).values
    etiquetas_x = [str(a) for a in ALPHAS_RIDGE]

    figura, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # Subgráfica 1: RMSE promedio ± DE (menor es mejor)
    rmse_vals = df_ridge["RMSE_promedio_CV"].values
    rmse_errs = df_ridge["RMSE_desv_est_CV"].values
    rmse_min = float(np.min(rmse_vals - rmse_errs))
    rmse_max = float(np.max(rmse_vals + rmse_errs))
    rmse_span = max(rmse_max - rmse_min, 0.02)

    ax_rmse.errorbar(
        alphas,
        rmse_vals,
        yerr=rmse_errs,
        fmt="o-",
        color="#2b6cb0",
        ecolor="#4a5568",
        capsize=4,
        linewidth=1.8,
        markersize=6,
        label="Ridge (promedio ± DE)",
    )
    ax_rmse.axhline(
        float(ref["RMSE_promedio_CV"]),
        color="#c53030",
        linestyle="--",
        linewidth=1.5,
        label=f"LinearRegression ({float(ref['RMSE_promedio_CV']):.4f})",
    )
    ax_rmse.set_xscale("log")
    ax_rmse.set_xticks(alphas)
    ax_rmse.set_xticklabels(etiquetas_x)
    ax_rmse.set_ylim(rmse_min - 0.34 * rmse_span, rmse_max + 0.12 * rmse_span)
    ax_rmse.set_title("RMSE promedio ± DE en CV 5-fold", fontsize=11.5, fontweight="bold")
    ax_rmse.set_xlabel("Parámetro de regularización alpha (escala log)", fontsize=10)
    ax_rmse.set_ylabel("RMSE en validación cruzada", fontsize=10)
    ax_rmse.legend(loc="lower left", fontsize=8.8, frameon=True)

    # Subgráfica 2: R² promedio ± DE (mayor es mejor)
    r2_vals = df_ridge["R2_promedio_CV"].values
    r2_errs = df_ridge["R2_desv_est_CV"].values
    r2_min = float(np.min(r2_vals - r2_errs))
    r2_max = float(np.max(r2_vals + r2_errs))
    r2_span = max(r2_max - r2_min, 0.02)

    ax_r2.errorbar(
        alphas,
        r2_vals,
        yerr=r2_errs,
        fmt="s-",
        color="#2f855a",
        ecolor="#4a5568",
        capsize=4,
        linewidth=1.8,
        markersize=6,
        label="Ridge (promedio ± DE)",
    )
    ax_r2.axhline(
        float(ref["R2_promedio_CV"]),
        color="#c53030",
        linestyle="--",
        linewidth=1.5,
        label=f"LinearRegression ({float(ref['R2_promedio_CV']):.4f})",
    )
    ax_r2.set_xscale("log")
    ax_r2.set_xticks(alphas)
    ax_r2.set_xticklabels(etiquetas_x)
    ax_r2.set_ylim(r2_min - 0.34 * r2_span, r2_max + 0.12 * r2_span)
    ax_r2.set_title("R² promedio ± DE en CV 5-fold", fontsize=11.5, fontweight="bold")
    ax_r2.set_xlabel("Parámetro de regularización alpha (escala log)", fontsize=10)
    ax_r2.set_ylabel("R² en validación cruzada", fontsize=10)
    ax_r2.legend(loc="lower left", fontsize=8.8, frameon=True)

    figura.suptitle(
        "Experimentación complementaria: Ridge vs. LinearRegression (CV 5-fold en entrenamiento)",
        fontsize=12.5,
        fontweight="bold",
    )
    figura.tight_layout(rect=(0, 0, 1, 0.93))

    return guardar_y_mostrar_figura(
        figura, RUTA_GRAFICA_RIDGE_CV.name, mostrar=mostrar
    )


def ejecutar_experimentacion_complementaria(
    mostrar_graficas: bool = True,
) -> pd.DataFrame:
    """Ejecuta validación cruzada 5-fold con Pipeline(StandardScaler) solo en X_train."""
    imprimir_encabezado("EXPERIMENTACION COMPLEMENTARIA: RIDGE CON VALIDACION CRUZADA")
    asegurar_carpetas_resultados()

    if not existen_datos_particion():
        raise FileNotFoundError(
            "No se encontraron los archivos de particion (X_train.csv e y_train.csv). "
            "Ejecute primero 'Particion_Datos.py'."
        )
    if not datos_particion_actualizados():
        raise ValueError(
            "Los archivos de particion estan desactualizados frente al dataset limpio. "
            "Ejecute nuevamente 'Particion_Datos.py'."
        )

    # Cargar exclusivamente entrenamiento original sin escalar (nunca X_test ni y_test)
    X_train = cargar_tabla_procesada(RUTA_X_TRAIN)
    y_train_df = cargar_tabla_procesada(RUTA_Y_TRAIN)
    y_train = y_train_df[COLUMNA_OBJETIVO]

    if list(X_train.columns) != COLUMNAS_INDEPENDIENTES:
        raise ValueError(
            "Las columnas de X_train no coinciden con las 11 variables esperadas."
        )
    if not X_train.index.equals(y_train.index):
        raise ValueError("Los indices de X_train e y_train no coinciden.")

    cv = KFold(
        n_splits=N_PARTICIONES_CV,
        shuffle=True,
        random_state=SEMILLA_ALEATORIA,
    )

    imprimir_subtitulo("1. Configuracion de validacion cruzada (sin fuga de datos)")
    print(f"Conjunto utilizado:       X_train sin escalar ({len(X_train)} x {X_train.shape[1]})")
    print(f"Esquema de particiones:   KFold(n_splits={N_PARTICIONES_CV}, shuffle=True, random_state={SEMILLA_ALEATORIA})")
    print("Flujo por particion:      Pipeline(StandardScaler() -> Modelo)")
    print(f"Valores de alpha Ridge:   {ALPHAS_RIDGE}")
    print("Conjunto de prueba:       NO utilizado (reservado exclusivamente para Fase 4)")

    filas = [
        _evaluar_pipeline_cv(
            "LinearRegression",
            "N/A (OLS)",
            LinearRegression(),
            X_train,
            y_train,
            cv,
        )
    ]
    for alpha in ALPHAS_RIDGE:
        filas.append(
            _evaluar_pipeline_cv(
                "Ridge",
                str(alpha),
                Ridge(alpha=alpha),
                X_train,
                y_train,
                cv,
            )
        )

    tabla_resultados = pd.DataFrame(
        filas,
        columns=[
            "Modelo",
            "Alpha",
            "RMSE_promedio_CV",
            "RMSE_desv_est_CV",
            "R2_promedio_CV",
            "R2_desv_est_CV",
            "Particiones_CV",
            "Registros_entrenamiento",
        ],
    )

    imprimir_subtitulo("2. Resultados de validacion cruzada (5 particiones)")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(tabla_resultados.round(6).to_string(index=False))

    tabla_resultados.to_csv(
        RUTA_TABLA_RIDGE_CV, index=False, encoding="utf-8-sig"
    )
    print(f"\nTabla complementaria guardada en: {RUTA_TABLA_RIDGE_CV}")

    graficar_comparacion_ridge_cv(tabla_resultados, mostrar=mostrar_graficas)
    print(
        "\nNota: Este analisis complementario no modifica ni sustituye al modelo "
        "principal LinearRegression guardado en modelos/."
    )
    return tabla_resultados


if __name__ == "__main__":
    try:
        ejecutar_experimentacion_complementaria(mostrar_graficas=False)
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        print(f"Error durante la experimentacion complementaria: {error}")
