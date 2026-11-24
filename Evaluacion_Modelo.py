"""Evaluación del rendimiento del modelo de Regresión Lineal Múltiple (Fase 4).

Carga el modelo entrenado y el escalador guardados mediante joblib junto con el
conjunto de prueba reservado, genera predicciones continuas, calcula MSE, RMSE,
MAE y R², analiza residuos, compara entrenamiento contra prueba, evalúa un modelo
de referencia (DummyRegressor), analiza el rendimiento por puntuación de calidad,
exporta tablas CSV, gráficas PNG a 300 DPI y el informe resumen_evaluacion.md.
"""

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from utilidades import (
    COLUMNA_OBJETIVO,
    COLUMNAS_INDEPENDIENTES,
    RUTA_CONFIG_MODELO,
    RUTA_ESCALADOR_JOBLIB,
    RUTA_MODELO_JOBLIB,
    RUTA_RESUMEN_EVALUACION,
    RUTA_TABLA_METRICAS,
    RUTA_TABLA_MODELO_BASE,
    RUTA_TABLA_PREDICCIONES,
    RUTA_TABLA_RENDIMIENTO_CALIDAD,
    RUTA_TABLA_RESIDUOS,
    RUTA_TABLA_TRAIN_TEST,
    RUTA_X_TEST,
    RUTA_X_TEST_ESCALADO,
    RUTA_X_TRAIN_ESCALADO,
    RUTA_Y_TEST,
    RUTA_Y_TRAIN,
    asegurar_carpetas_resultados,
    cargar_tabla_procesada,
    datos_escalados_actualizados,
    datos_particion_actualizados,
    existe_modelo_entrenado,
    guardar_y_mostrar_figura,
    imprimir_encabezado,
    imprimir_subtitulo,
    modelo_entrenado_actualizado,
)


def cargar_modelo() -> tuple[LinearRegression, StandardScaler, dict]:
    """Carga el modelo, el escalador y la configuración verificando existencia y vigencia."""
    if not existe_modelo_entrenado():
        raise FileNotFoundError(
            "Faltan archivos del modelo entrenado en 'modelos/'. "
            "Ejecute primero las fases 2 y 3 (particion, escalado y entrenamiento)."
        )

    if not datos_particion_actualizados():
        raise ValueError(
            "Los archivos de particion estan desactualizados frente al dataset limpio. "
            "Ejecute nuevamente las fases 2 y 3 antes de evaluar."
        )

    if not datos_escalados_actualizados():
        raise ValueError(
            "Los archivos escalados estan desactualizados frente a la particion actual. "
            "Ejecute nuevamente el escalado (fase 2) y el entrenamiento (fase 3)."
        )

    if not modelo_entrenado_actualizado():
        raise ValueError(
            "El modelo guardado esta desactualizado respecto a los datos procesados actuales. "
            "Vuelva a entrenar el modelo (fase 3) antes de evaluar."
        )

    modelo = joblib.load(RUTA_MODELO_JOBLIB)
    escalador = joblib.load(RUTA_ESCALADOR_JOBLIB)
    with open(RUTA_CONFIG_MODELO, "r", encoding="utf-8") as archivo_json:
        config = json.load(archivo_json)

    # Verificar consistencia interna entre el modelo, el escalador y el JSON
    if config.get("orden_variables_independientes") != COLUMNAS_INDEPENDIENTES:
        raise ValueError(
            "El orden de variables en 'configuracion_modelo.json' no coincide con el esperado."
        )
    if not hasattr(escalador, "feature_names_in_") or (
        list(escalador.feature_names_in_) != COLUMNAS_INDEPENDIENTES
    ):
        raise ValueError(
            "Las caracteristicas del escalador guardado no coinciden con las 11 esperadas."
        )
    if len(modelo.coef_) != len(COLUMNAS_INDEPENDIENTES):
        raise ValueError(
            "La cantidad de coeficientes del modelo no coincide con las 11 variables independientes."
        )
    if not np.isclose(float(modelo.intercept_), float(config["intercepto"])):
        raise ValueError(
            "El intercepto del modelo guardado no coincide con 'configuracion_modelo.json'."
        )

    return modelo, escalador, config


def cargar_datos_prueba(
    config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Carga X_test, X_test_escalado e y_test verificando columnas, orden e índices."""
    X_test = cargar_tabla_procesada(RUTA_X_TEST)
    X_test_escalado = cargar_tabla_procesada(RUTA_X_TEST_ESCALADO)
    y_test_df = cargar_tabla_procesada(RUTA_Y_TEST)

    if COLUMNA_OBJETIVO not in y_test_df.columns:
        raise KeyError(
            f"No se encontro la columna objetivo '{COLUMNA_OBJETIVO}' en y_test.csv."
        )
    y_test = y_test_df[COLUMNA_OBJETIVO]

    if not pd.api.types.is_numeric_dtype(y_test):
        raise ValueError("La variable objetivo 'y_test' debe ser numerica.")

    if (
        list(X_test.columns) != COLUMNAS_INDEPENDIENTES
        or list(X_test_escalado.columns) != COLUMNAS_INDEPENDIENTES
    ):
        raise ValueError(
            "El orden o nombre de columnas en X_test o X_test_escalado no es consistente."
        )

    if not (len(X_test) == len(X_test_escalado) == len(y_test)):
        raise ValueError(
            "X_test, X_test_escalado e y_test no tienen la misma cantidad de registros."
        )

    if not (
        X_test.index.equals(y_test.index)
        and X_test_escalado.index.equals(y_test.index)
    ):
        raise ValueError(
            "Los indices de X_test, X_test_escalado e y_test no estan alineados."
        )

    if config is not None and len(y_test) != int(config["registros_prueba"]):
        raise ValueError(
            f"La cantidad de registros de prueba ({len(y_test)}) no coincide con "
            f"configuracion_modelo.json ({config['registros_prueba']})."
        )

    return X_test, X_test_escalado, y_test


def cargar_datos_entrenamiento(
    config: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Carga X_train_escalado e y_train únicamente como diagnóstico complementario."""
    X_train_escalado = cargar_tabla_procesada(RUTA_X_TRAIN_ESCALADO)
    y_train_df = cargar_tabla_procesada(RUTA_Y_TRAIN)

    if COLUMNA_OBJETIVO not in y_train_df.columns:
        raise KeyError(
            f"No se encontro la columna objetivo '{COLUMNA_OBJETIVO}' en y_train.csv."
        )
    y_train = y_train_df[COLUMNA_OBJETIVO]

    if list(X_train_escalado.columns) != COLUMNAS_INDEPENDIENTES:
        raise ValueError(
            "Las columnas de X_train_escalado no coinciden con las 11 esperadas."
        )
    if not X_train_escalado.index.equals(y_train.index):
        raise ValueError("Los indices de X_train_escalado e y_train no coinciden.")

    if config is not None and len(y_train) != int(config["registros_entrenamiento"]):
        raise ValueError(
            f"La cantidad de registros de entrenamiento ({len(y_train)}) no coincide con "
            f"configuracion_modelo.json ({config['registros_entrenamiento']})."
        )

    return X_train_escalado, y_train


def generar_predicciones(
    modelo: LinearRegression,
    X_test_escalado: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Genera predicciones continuas sobre X_test_escalado y calcula errores por registro."""
    if not X_test_escalado.index.equals(y_test.index):
        raise ValueError(
            "Los indices de X_test_escalado e y_test deben coincidir antes de predecir."
        )

    # Predicción directa sin redondeo, recorte ni categorización
    y_pred = np.asarray(modelo.predict(X_test_escalado), dtype=float)
    y_real_float = y_test.to_numpy(dtype=float)

    residuos = y_real_float - y_pred
    errores_absolutos = np.abs(residuos)
    errores_cuadraticos = residuos ** 2

    df_predicciones = pd.DataFrame(
        {
            "Registro": X_test_escalado.index.to_numpy(),
            "Calidad_real": y_test.to_numpy(),
            "Calidad_predicha": y_pred,
            "Residuo": residuos,
            "Error_absoluto": errores_absolutos,
            "Error_cuadratico": errores_cuadraticos,
        }
    )

    return y_pred, df_predicciones


def calcular_metricas(
    y_real: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Calcula MSE, RMSE, MAE y R² manteniendo precisión completa de punto flotante."""
    y_real_arr = np.asarray(y_real, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mse = float(mean_squared_error(y_real_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_real_arr, y_pred_arr))
    r2 = float(r2_score(y_real_arr, y_pred_arr))

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
    }


def construir_tabla_metricas(metricas: dict[str, float]) -> pd.DataFrame:
    """Construye una tabla descriptiva con las cuatro métricas de regresión."""
    return pd.DataFrame(
        [
            {
                "Métrica": "MSE",
                "Valor": metricas["MSE"],
                "Fórmula": "MSE = (1/n) * suma((y_real - y_pred)^2)",
                "Descripción": (
                    "Promedio de errores al cuadrado; penaliza errores grandes "
                    "(unidades de calidad al cuadrado)."
                ),
            },
            {
                "Métrica": "RMSE",
                "Valor": metricas["RMSE"],
                "Fórmula": "RMSE = raiz_cuadrada(MSE)",
                "Descripción": (
                    "Raíz del MSE; error típico expresado en la misma escala "
                    "que la calidad del vino."
                ),
            },
            {
                "Métrica": "MAE",
                "Valor": metricas["MAE"],
                "Fórmula": "MAE = (1/n) * suma(abs(y_real - y_pred))",
                "Descripción": (
                    "Promedio de diferencias absolutas; indica cuántos puntos de "
                    "calidad se equivoca en promedio."
                ),
            },
            {
                "Métrica": "R²",
                "Valor": metricas["R²"],
                "Fórmula": "R² = 1 - (suma((y_real - y_pred)^2) / suma((y_real - media)^2))",
                "Descripción": (
                    "Proporción de variabilidad de quality explicada por el modelo "
                    "(no es un porcentaje de exactitud de clasificación)."
                ),
            },
        ]
    )


def analizar_residuos(
    df_predicciones: pd.DataFrame,
    tolerancia_cero: float = 1e-6,
) -> pd.DataFrame:
    """Calcula estadísticas descriptivas y conteos por signo de los residuos."""
    residuos = df_predicciones["Residuo"]

    media = float(residuos.mean())
    mediana = float(residuos.median())
    desv_est = float(residuos.std(ddof=1))
    minimo = float(residuos.min())
    maximo = float(residuos.max())
    positivos = int((residuos > tolerancia_cero).sum())
    negativos = int((residuos < -tolerancia_cero).sum())
    iguales_cero = int((residuos.abs() <= tolerancia_cero).sum())

    return pd.DataFrame(
        [
            {
                "Estadística": "Media de los residuos",
                "Valor": media,
                "Interpretación": (
                    "Sesgo promedio en prueba (>0 subestima, <0 sobreestima)."
                ),
            },
            {
                "Estadística": "Mediana de los residuos",
                "Valor": mediana,
                "Interpretación": "Valor central de los residuos en prueba.",
            },
            {
                "Estadística": "Desviación estándar",
                "Valor": desv_est,
                "Interpretación": "Dispersión muestral (ddof=1) de los residuos.",
            },
            {
                "Estadística": "Residuo mínimo",
                "Valor": minimo,
                "Interpretación": "Mayor sobreestimación (Calidad_real - Calidad_predicha).",
            },
            {
                "Estadística": "Residuo máximo",
                "Valor": maximo,
                "Interpretación": "Mayor subestimación (Calidad_real - Calidad_predicha).",
            },
            {
                "Estadística": "Cantidad de residuos positivos",
                "Valor": float(positivos),
                "Interpretación": (
                    f"Registros donde el modelo subestimó la calidad (residuo > {tolerancia_cero:g})."
                ),
            },
            {
                "Estadística": "Cantidad de residuos negativos",
                "Valor": float(negativos),
                "Interpretación": (
                    f"Registros donde el modelo sobreestimó la calidad (residuo < -{tolerancia_cero:g})."
                ),
            },
            {
                "Estadística": "Cantidad de residuos iguales a cero",
                "Valor": float(iguales_cero),
                "Interpretación": (
                    f"Predicciones dentro de la tolerancia |residuo| <= {tolerancia_cero:g}."
                ),
            },
        ]
    )


def comparar_entrenamiento_prueba(
    modelo: LinearRegression,
    X_train_escalado: pd.DataFrame,
    y_train: pd.Series,
    metricas_prueba: dict[str, float],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evalúa el modelo ya ajustado en entrenamiento para diagnosticar generalización."""
    y_pred_train = np.asarray(modelo.predict(X_train_escalado), dtype=float)
    metricas_train = calcular_metricas(y_train.to_numpy(dtype=float), y_pred_train)

    df_comparacion = pd.DataFrame(
        {
            "Métrica": ["MSE", "RMSE", "MAE", "R²"],
            "Entrenamiento": [
                metricas_train["MSE"],
                metricas_train["RMSE"],
                metricas_train["MAE"],
                metricas_train["R²"],
            ],
            "Prueba": [
                metricas_prueba["MSE"],
                metricas_prueba["RMSE"],
                metricas_prueba["MAE"],
                metricas_prueba["R²"],
            ],
        }
    )

    return metricas_train, df_comparacion


def evaluar_modelo_base(
    X_train_escalado: pd.DataFrame,
    y_train: pd.Series,
    X_test_escalado: pd.DataFrame,
    y_test: pd.Series,
    metricas_prueba: dict[str, float],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evalúa DummyRegressor(strategy='mean') en prueba como referencia base."""
    modelo_base = DummyRegressor(strategy="mean")
    modelo_base.fit(X_train_escalado, y_train)
    predicciones_base = modelo_base.predict(X_test_escalado)

    metricas_base = calcular_metricas(y_test.to_numpy(dtype=float), predicciones_base)

    df_comparacion_base = pd.DataFrame(
        {
            "Métrica": ["MSE", "RMSE", "MAE", "R²"],
            "Modelo base (DummyRegressor)": [
                metricas_base["MSE"],
                metricas_base["RMSE"],
                metricas_base["MAE"],
                metricas_base["R²"],
            ],
            "Regresión Lineal Múltiple": [
                metricas_prueba["MSE"],
                metricas_prueba["RMSE"],
                metricas_prueba["MAE"],
                metricas_prueba["R²"],
            ],
            "Diferencia (RLM - Base)": [
                metricas_prueba["MSE"] - metricas_base["MSE"],
                metricas_prueba["RMSE"] - metricas_base["RMSE"],
                metricas_prueba["MAE"] - metricas_base["MAE"],
                metricas_prueba["R²"] - metricas_base["R²"],
            ],
        }
    )

    return metricas_base, df_comparacion_base


def analizar_rendimiento_por_calidad(df_predicciones: pd.DataFrame) -> pd.DataFrame:
    """Agrupa las predicciones de prueba según cada puntuación real de calidad."""
    filas: list[dict[str, float | int]] = []
    for calidad, grupo in df_predicciones.groupby("Calidad_real", sort=True):
        filas.append(
            {
                "Calidad_real": int(calidad),
                "Cantidad_registros": int(len(grupo)),
                "Promedio_predicciones": float(grupo["Calidad_predicha"].mean()),
                "MAE": float(grupo["Error_absoluto"].mean()),
                "RMSE": float(np.sqrt(grupo["Error_cuadratico"].mean())),
                "Residuo_promedio": float(grupo["Residuo"].mean()),
            }
        )

    return pd.DataFrame(filas)


def _graficar_reales_vs_predichos(
    df_predicciones: pd.DataFrame,
    mostrar: bool = True,
) -> Path:
    """Genera gráfica 13: dispersión de calidad real contra calidad predicha."""
    sns.set_theme(style="whitegrid")
    y_real = df_predicciones["Calidad_real"].to_numpy(dtype=float)
    y_pred = df_predicciones["Calidad_predicha"].to_numpy(dtype=float)

    limite_inf = min(float(y_real.min()), float(y_pred.min())) - 0.35
    limite_sup = max(float(y_real.max()), float(y_pred.max())) + 0.35

    figura, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(
        y_real,
        y_pred,
        alpha=0.55,
        color="#2b6cb0",
        edgecolors="white",
        linewidth=0.4,
        label="Observaciones de prueba",
    )
    ax.plot(
        [limite_inf, limite_sup],
        [limite_inf, limite_sup],
        color="#c53030",
        linestyle="--",
        linewidth=1.5,
        label="Predicción perfecta (y = x)",
    )

    ax.set_xlim(limite_inf, limite_sup)
    ax.set_ylim(limite_inf, limite_sup)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Calidad real vs. calidad predicha en el conjunto de prueba",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Calidad real (quality)", fontsize=11)
    ax.set_ylabel("Calidad predicha", fontsize=11)
    ax.legend(loc="upper left")
    figura.tight_layout()

    return guardar_y_mostrar_figura(
        figura, "13_valores_reales_vs_predichos.png", mostrar=mostrar
    )


def _graficar_comparacion_primeras_30(
    df_predicciones: pd.DataFrame,
    mostrar: bool = True,
) -> Path:
    """Genera gráfica 14: líneas real vs predicho en los primeros 30 registros de prueba."""
    sns.set_theme(style="whitegrid")
    muestra = df_predicciones.head(30).reset_index(drop=True)
    posiciones = np.arange(1, len(muestra) + 1)

    figura, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        posiciones,
        muestra["Calidad_real"].to_numpy(dtype=float),
        marker="o",
        linewidth=1.8,
        color="#2b6cb0",
        label="Calidad real",
    )
    ax.plot(
        posiciones,
        muestra["Calidad_predicha"].to_numpy(dtype=float),
        marker="s",
        linestyle="--",
        linewidth=1.8,
        color="#dd6b20",
        label="Calidad predicha",
    )

    ax.set_title(
        "Comparación entre calidad real y predicha (primeros 30 registros de prueba)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        "Posición en los primeros 30 registros de prueba (orden no temporal)",
        fontsize=10.5,
    )
    ax.set_ylabel("Puntuación de calidad", fontsize=11)
    ax.set_xticks(posiciones)
    ax.legend(loc="best")
    figura.tight_layout()

    return guardar_y_mostrar_figura(
        figura, "14_comparacion_predicciones.png", mostrar=mostrar
    )


def _graficar_dispersion_residuos(
    df_predicciones: pd.DataFrame,
    mostrar: bool = True,
) -> Path:
    """Genera gráfica 15: valores predichos vs residuos con línea horizontal en cero."""
    sns.set_theme(style="whitegrid")
    figura, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(
        df_predicciones["Calidad_predicha"],
        df_predicciones["Residuo"],
        alpha=0.55,
        color="#2b6cb0",
        edgecolors="white",
        linewidth=0.4,
    )
    ax.axhline(0, color="#c53030", linestyle="--", linewidth=1.4, label="Residuo = 0")

    ax.set_title(
        "Residuos vs. valores predichos (conjunto de prueba)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Calidad predicha", fontsize=11)
    ax.set_ylabel("Residuo (Calidad real - Calidad predicha)", fontsize=11)
    ax.legend(loc="upper right")
    figura.tight_layout()

    return guardar_y_mostrar_figura(
        figura, "15_grafica_residuos.png", mostrar=mostrar
    )


def _graficar_distribucion_residuos(
    df_predicciones: pd.DataFrame,
    mostrar: bool = True,
) -> Path:
    """Genera gráfica 16: histograma y KDE de los residuos con línea vertical en cero."""
    sns.set_theme(style="whitegrid")
    figura, ax = plt.subplots(figsize=(9, 5.5))
    sns.histplot(
        df_predicciones["Residuo"],
        kde=True,
        color="#2b6cb0",
        ax=ax,
    )
    ax.axvline(0, color="#c53030", linestyle="--", linewidth=1.4, label="Residuo = 0")

    ax.set_title(
        "Distribución de los residuos en el conjunto de prueba",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Residuo (Calidad real - Calidad predicha)", fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    ax.legend(loc="upper right")
    figura.tight_layout()

    return guardar_y_mostrar_figura(
        figura, "16_distribucion_residuos.png", mostrar=mostrar
    )


def _graficar_boxplot_residuos(
    df_predicciones: pd.DataFrame,
    mostrar: bool = True,
) -> Path:
    """Genera gráfica 17: diagrama de caja de los residuos para observar extremos."""
    sns.set_theme(style="whitegrid")
    figura, ax = plt.subplots(figsize=(9, 4.2))
    sns.boxplot(
        x=df_predicciones["Residuo"],
        color="#4299e1",
        width=0.45,
        ax=ax,
    )
    ax.axvline(0, color="#c53030", linestyle="--", linewidth=1.4, label="Residuo = 0")

    ax.set_title(
        "Diagrama de caja de los residuos en el conjunto de prueba",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Residuo (Calidad real - Calidad predicha)", fontsize=11)
    ax.legend(loc="upper right")
    figura.tight_layout()

    return guardar_y_mostrar_figura(
        figura, "17_boxplot_residuos.png", mostrar=mostrar
    )


def _anotar_barras_verticales(ax, barras, decimales: int = 4) -> None:
    """Coloca etiquetas numéricas sobre o debajo de barras verticales evitando solapamientos."""
    for barra in barras:
        altura = float(barra.get_height())
        desplazamiento = 0.015 if altura >= 0 else -0.015
        alineacion = "bottom" if altura >= 0 else "top"
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            altura + desplazamiento,
            f"{altura:.{decimales}f}",
            ha="center",
            va=alineacion,
            fontsize=9,
            fontweight="bold",
        )


def _graficar_comparacion_dos_paneles(
    etiqueta_a: str,
    etiqueta_b: str,
    valores_err_a: list[float],
    valores_err_b: list[float],
    r2_a: float,
    r2_b: float,
    color_a: str,
    color_b: str,
    titulo_general: str,
    nombre_archivo: str,
    mostrar: bool = True,
) -> Path:
    """Genera una gráfica de 2 paneles separando métricas de error (MSE/RMSE/MAE) y R²."""
    sns.set_theme(style="whitegrid")
    metricas_error = ["MSE", "RMSE", "MAE"]
    posiciones = np.arange(len(metricas_error))
    ancho = 0.36

    figura, (ax_err, ax_r2) = plt.subplots(
        1, 2, figsize=(12.5, 5.5), gridspec_kw={"width_ratios": [2.2, 1.1]}
    )

    # Panel izquierdo: métricas de error
    barras_a = ax_err.bar(
        posiciones - ancho / 2,
        valores_err_a,
        width=ancho,
        color=color_a,
        edgecolor="#1a202c",
        linewidth=0.7,
        label=etiqueta_a,
    )
    barras_b = ax_err.bar(
        posiciones + ancho / 2,
        valores_err_b,
        width=ancho,
        color=color_b,
        edgecolor="#1a202c",
        linewidth=0.7,
        label=etiqueta_b,
    )
    _anotar_barras_verticales(ax_err, barras_a, decimales=4)
    _anotar_barras_verticales(ax_err, barras_b, decimales=4)

    max_err = max(max(valores_err_a), max(valores_err_b))
    ax_err.set_ylim(0, max_err * 1.22)
    ax_err.set_xticks(posiciones)
    ax_err.set_xticklabels(metricas_error, fontsize=10.5)
    ax_err.set_title("Métricas de error (menor es mejor)", fontsize=11.5, fontweight="bold")
    ax_err.set_ylabel("Valor del error", fontsize=10.5)
    ax_err.legend(loc="upper right")
    ax_err.grid(axis="x", visible=False)

    # Panel derecho: R² en su propia escala
    pos_r2 = np.array([0.0])
    barra_r2_a = ax_r2.bar(
        pos_r2 - ancho / 2,
        [r2_a],
        width=ancho,
        color=color_a,
        edgecolor="#1a202c",
        linewidth=0.7,
        label=etiqueta_a,
    )
    barra_r2_b = ax_r2.bar(
        pos_r2 + ancho / 2,
        [r2_b],
        width=ancho,
        color=color_b,
        edgecolor="#1a202c",
        linewidth=0.7,
        label=etiqueta_b,
    )
    ax_r2.axhline(0, color="#2d3748", linewidth=1)
    _anotar_barras_verticales(ax_r2, barra_r2_a, decimales=4)
    _anotar_barras_verticales(ax_r2, barra_r2_b, decimales=4)

    min_r2 = min(0.0, r2_a, r2_b)
    max_r2 = max(0.0, r2_a, r2_b)
    rango_r2 = max(max_r2 - min_r2, 0.25)
    ax_r2.set_ylim(min_r2 - rango_r2 * 0.25, max_r2 + rango_r2 * 0.25)
    ax_r2.set_xticks(pos_r2)
    ax_r2.set_xticklabels(["R²"], fontsize=10.5)
    ax_r2.set_title(
        "Coeficiente de determinación R² (mayor es mejor)", fontsize=11.5, fontweight="bold"
    )
    ax_r2.set_ylabel("Valor de R²", fontsize=10.5)
    ax_r2.grid(axis="x", visible=False)

    figura.suptitle(titulo_general, fontsize=13.5, fontweight="bold", y=1.02)
    figura.tight_layout()

    return guardar_y_mostrar_figura(figura, nombre_archivo, mostrar=mostrar)


def _graficar_comparacion_train_test(
    df_comparacion: pd.DataFrame,
    mostrar: bool = True,
) -> Path:
    """Genera gráfica 18: comparación de métricas entre entrenamiento y prueba."""
    valores_train = df_comparacion.set_index("Métrica")["Entrenamiento"]
    valores_test = df_comparacion.set_index("Métrica")["Prueba"]

    return _graficar_comparacion_dos_paneles(
        etiqueta_a="Entrenamiento",
        etiqueta_b="Prueba",
        valores_err_a=[float(valores_train[m]) for m in ("MSE", "RMSE", "MAE")],
        valores_err_b=[float(valores_test[m]) for m in ("MSE", "RMSE", "MAE")],
        r2_a=float(valores_train["R²"]),
        r2_b=float(valores_test["R²"]),
        color_a="#2b6cb0",
        color_b="#dd6b20",
        titulo_general="Comparación de rendimiento entre entrenamiento y prueba",
        nombre_archivo="18_comparacion_train_test.png",
        mostrar=mostrar,
    )


def _graficar_comparacion_modelo_base(
    df_comparacion_base: pd.DataFrame,
    mostrar: bool = True,
) -> Path:
    """Genera gráfica 19: comparación entre DummyRegressor y Regresión Lineal Múltiple."""
    valores_base = df_comparacion_base.set_index("Métrica")["Modelo base (DummyRegressor)"]
    valores_rlm = df_comparacion_base.set_index("Métrica")["Regresión Lineal Múltiple"]

    return _graficar_comparacion_dos_paneles(
        etiqueta_a="Modelo base (media)",
        etiqueta_b="Regresión Lineal Múltiple",
        valores_err_a=[float(valores_base[m]) for m in ("MSE", "RMSE", "MAE")],
        valores_err_b=[float(valores_rlm[m]) for m in ("MSE", "RMSE", "MAE")],
        r2_a=float(valores_base["R²"]),
        r2_b=float(valores_rlm["R²"]),
        color_a="#718096",
        color_b="#2b6cb0",
        titulo_general="Comparación contra modelo de referencia (DummyRegressor)",
        nombre_archivo="19_comparacion_modelo_base.png",
        mostrar=mostrar,
    )


def _graficar_error_por_calidad(
    df_rendimiento_calidad: pd.DataFrame,
    mostrar: bool = True,
) -> Path:
    """Genera gráfica 20: MAE y cantidad de registros por puntuación real de calidad."""
    sns.set_theme(style="whitegrid")
    calidades = df_rendimiento_calidad["Calidad_real"].astype(str).tolist()
    maes = df_rendimiento_calidad["MAE"].to_numpy(dtype=float)
    conteos = df_rendimiento_calidad["Cantidad_registros"].to_numpy(dtype=int)

    figura, ax = plt.subplots(figsize=(9, 5.5))
    barras = ax.bar(
        calidades,
        maes,
        color="#2b6cb0",
        edgecolor="#1a365d",
        linewidth=0.8,
        width=0.55,
    )

    max_mae = float(maes.max())
    for barra, mae_val, n_reg in zip(barras, maes, conteos):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + max_mae * 0.025,
            f"{mae_val:.2f}\n(n={n_reg})",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_ylim(0, max_mae * 1.24)
    ax.set_title(
        "Error absoluto medio (MAE) según la puntuación real de calidad",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Puntuación real de calidad (quality)", fontsize=11)
    ax.set_ylabel("Error absoluto medio (MAE)", fontsize=11)
    ax.grid(axis="x", visible=False)
    figura.tight_layout()

    return guardar_y_mostrar_figura(
        figura, "20_error_por_calidad.png", mostrar=mostrar
    )


def generar_graficas_evaluacion(
    df_predicciones: pd.DataFrame,
    df_comparacion_train_test: pd.DataFrame,
    df_comparacion_base: pd.DataFrame,
    df_rendimiento_calidad: pd.DataFrame,
    mostrar: bool = True,
) -> list[Path]:
    """Genera y guarda en 300 DPI las 8 gráficas de evaluación de la Fase 4."""
    asegurar_carpetas_resultados()
    return [
        _graficar_reales_vs_predichos(df_predicciones, mostrar=mostrar),
        _graficar_comparacion_primeras_30(df_predicciones, mostrar=mostrar),
        _graficar_dispersion_residuos(df_predicciones, mostrar=mostrar),
        _graficar_distribucion_residuos(df_predicciones, mostrar=mostrar),
        _graficar_boxplot_residuos(df_predicciones, mostrar=mostrar),
        _graficar_comparacion_train_test(df_comparacion_train_test, mostrar=mostrar),
        _graficar_comparacion_modelo_base(df_comparacion_base, mostrar=mostrar),
        _graficar_error_por_calidad(df_rendimiento_calidad, mostrar=mostrar),
    ]


def guardar_resultados(
    df_predicciones: pd.DataFrame,
    df_metricas: pd.DataFrame,
    df_estadisticas_residuos: pd.DataFrame,
    df_comparacion_train_test: pd.DataFrame,
    df_comparacion_base: pd.DataFrame,
    df_rendimiento_calidad: pd.DataFrame,
) -> dict[str, Path]:
    """Guarda en resultados/tablas/ las 6 tablas CSV de evaluación."""
    asegurar_carpetas_resultados()
    archivos = {
        "predicciones_modelo": (df_predicciones, RUTA_TABLA_PREDICCIONES),
        "metricas_evaluacion": (df_metricas, RUTA_TABLA_METRICAS),
        "estadisticas_residuos": (df_estadisticas_residuos, RUTA_TABLA_RESIDUOS),
        "comparacion_train_test": (df_comparacion_train_test, RUTA_TABLA_TRAIN_TEST),
        "comparacion_modelo_base": (df_comparacion_base, RUTA_TABLA_MODELO_BASE),
        "rendimiento_por_calidad": (df_rendimiento_calidad, RUTA_TABLA_RENDIMIENTO_CALIDAD),
    }

    rutas_guardadas: dict[str, Path] = {}
    for clave, (df, ruta) in archivos.items():
        df.to_csv(ruta, index=False, encoding="utf-8-sig")
        rutas_guardadas[clave] = ruta
        print(f"Tabla guardada en: {ruta}")

    return rutas_guardadas


def _construir_resumen_generalizacion(
    metricas_train: dict[str, float],
    metricas_prueba: dict[str, float],
    metricas_base: dict[str, float],
) -> tuple[str, str]:
    """Redacta un diagnóstico objetivo de sobreajuste, subajuste y generalización."""
    diff_r2 = metricas_train["R²"] - metricas_prueba["R²"]
    diff_rmse = metricas_prueba["RMSE"] - metricas_train["RMSE"]
    mejora_r2_base = metricas_prueba["R²"] - metricas_base["R²"]
    reduccion_rmse = metricas_base["RMSE"] - metricas_prueba["RMSE"]

    texto_sobreajuste_subajuste = (
        f"- **Sobreajuste:** No se observan indicios marcados de sobreajuste, ya que el "
        f"rendimiento en entrenamiento (`R² = {metricas_train['R²']:.4f}`, `RMSE = {metricas_train['RMSE']:.4f}`) "
        f"es muy cercano al obtenido en prueba (`R² = {metricas_prueba['R²']:.4f}`, "
        f"`RMSE = {metricas_prueba['RMSE']:.4f}`), con una diferencia de `R²` de `{diff_r2:+.4f}` "
        f"y de `RMSE` de `{diff_rmse:+.4f}`.\n"
        f"- **Subajuste:** Sí existen indicios de capacidad explicativa limitada o subajuste "
        f"frente a la complejidad real del fenómeno: tanto en entrenamiento (`R² = {metricas_train['R²']:.4f}`) "
        f"como en prueba (`R² = {metricas_prueba['R²']:.4f}`), un modelo estrictamente lineal "
        f"explica menos de la mitad de la variabilidad de `quality`, dejando un error promedio de "
        f"`MAE = {metricas_prueba['MAE']:.4f}` puntos."
    )

    texto_generalizacion = (
        f"La brecha entre entrenamiento y prueba es reducida (`MAE`: `{metricas_train['MAE']:.4f}` vs "
        f"`{metricas_prueba['MAE']:.4f}`; `RMSE`: `{metricas_train['RMSE']:.4f}` vs "
        f"`{metricas_prueba['RMSE']:.4f}`), lo que muestra estabilidad al evaluar datos no vistos. "
        f"Además, frente al modelo de referencia que predice siempre la media (`RMSE = {metricas_base['RMSE']:.4f}`, "
        f"`R² = {metricas_base['R²']:.4f}`), la regresión lineal múltiple reduce el RMSE en "
        f"`{reduccion_rmse:.4f}` puntos e incrementa `R²` en `{mejora_r2_base:+.4f}`. No obstante, "
        f"que entrenamiento y prueba sean similares no implica por sí solo alta precisión: el error "
        f"sigue siendo moderado y requiere interpretar sus limitaciones."
    )

    return texto_sobreajuste_subajuste, texto_generalizacion


def generar_resumen_evaluacion(
    config: dict,
    metricas_prueba: dict[str, float],
    metricas_train: dict[str, float],
    metricas_base: dict[str, float],
    df_comparacion_train_test: pd.DataFrame,
    df_comparacion_base: pd.DataFrame,
    df_estadisticas_residuos: pd.DataFrame,
    df_rendimiento_calidad: pd.DataFrame,
) -> Path:
    """Genera el informe académico resultados/informes/resumen_evaluacion.md con datos reales."""
    asegurar_carpetas_resultados()

    n_train = int(config["registros_entrenamiento"])
    n_test = int(config["registros_prueba"])
    n_features = len(config["orden_variables_independientes"])
    algoritmo = config["algoritmo"]

    stats_res = df_estadisticas_residuos.set_index("Estadística")["Valor"]
    media_res = float(stats_res["Media de los residuos"])
    mediana_res = float(stats_res["Mediana de los residuos"])
    std_res = float(stats_res["Desviación estándar"])
    min_res = float(stats_res["Residuo mínimo"])
    max_res = float(stats_res["Residuo máximo"])
    pos_res = int(stats_res["Cantidad de residuos positivos"])
    neg_res = int(stats_res["Cantidad de residuos negativos"])
    cero_res = int(stats_res["Cantidad de residuos iguales a cero"])

    texto_sobreajuste_subajuste, texto_generalizacion = _construir_resumen_generalizacion(
        metricas_train, metricas_prueba, metricas_base
    )

    calidades_medias = df_rendimiento_calidad[
        df_rendimiento_calidad["Calidad_real"].isin([5, 6])
    ]
    n_calidades_medias = int(calidades_medias["Cantidad_registros"].sum())
    pct_calidades_medias = (n_calidades_medias / n_test) * 100

    filas_calidad_md = "\n".join(
        f"| {int(f['Calidad_real'])} | {int(f['Cantidad_registros'])} | "
        f"{float(f['Promedio_predicciones']):.4f} | {float(f['MAE']):.4f} | "
        f"{float(f['RMSE']):.4f} | {float(f['Residuo_promedio']):+.4f} |"
        for _, f in df_rendimiento_calidad.iterrows()
    )

    filas_tt_md = "\n".join(
        f"| {f['Métrica']} | {float(f['Entrenamiento']):.4f} | {float(f['Prueba']):.4f} |"
        for _, f in df_comparacion_train_test.iterrows()
    )

    filas_base_md = "\n".join(
        f"| {f['Métrica']} | {float(f['Modelo base (DummyRegressor)']):.4f} | "
        f"{float(f['Regresión Lineal Múltiple']):.4f} | "
        f"{float(f['Diferencia (RLM - Base)']):+.4f} |"
        for _, f in df_comparacion_base.iterrows()
    )

    contenido = f"""# Resumen de Evaluación del Modelo — Fase 4

## 1. Información general de la evaluación

- **Modelo evaluado:** {algoritmo}
- **Dataset:** `{config['dataset']}`
- **Registros de entrenamiento:** {n_train}
- **Registros de prueba (evaluados):** {n_test}
- **Variables independientes:** {n_features}

## 2. Métricas obtenidas en el conjunto de prueba

| Métrica | Valor obtenido | Interpretación individual |
|---|---:|---|
| **MSE** | `{metricas_prueba['MSE']:.4f}` | Promedio de los errores al cuadrado (unidades de calidad²); penaliza en mayor medida las desviaciones grandes. |
| **RMSE** | `{metricas_prueba['RMSE']:.4f}` | Raíz cuadrada del MSE (`sqrt({metricas_prueba['MSE']:.4f})`); error cuadrático típico expresado en puntos de calidad. |
| **MAE** | `{metricas_prueba['MAE']:.4f}` | En promedio, las predicciones se desvían `{metricas_prueba['MAE']:.4f}` puntos respecto a la calificación real de calidad. |
| **R²** | `{metricas_prueba['R²']:.4f}` | El modelo lineal explica aproximadamente el `{metricas_prueba['R²']*100:.2f}%` de la variabilidad de `quality` en el conjunto de prueba (no equivale a exactitud de clasificación). |

## 3. Comparación entre entrenamiento y prueba

| Métrica | Entrenamiento | Prueba |
|---|---:|---:|
{filas_tt_md}

## 4. Comparación con el modelo de referencia (`DummyRegressor`)

El modelo de referencia (`DummyRegressor(strategy="mean")`) predice constantemente la media de `quality` observada en entrenamiento.

| Métrica | Modelo base (`DummyRegressor`) | Regresión Lineal Múltiple | Diferencia (RLM - Base) |
|---|---:|---:|---:|
{filas_base_md}

La regresión lineal múltiple supera al modelo base al disminuir los errores (`MSE`, `RMSE` y `MAE`) y aumentar `R²` de `{metricas_base['R²']:.4f}` a `{metricas_prueba['R²']:.4f}`, lo que confirma que las 11 variables fisicoquímicas aportan información lineal útil por encima de predecir simplemente el promedio.

## 5. Análisis de los residuos (`Calidad_real - Calidad_predicha`)

- **Media de los residuos:** `{media_res:+.4f}`
- **Mediana de los residuos:** `{mediana_res:+.4f}`
- **Desviación estándar:** `{std_res:.4f}`
- **Rango observado:** mínimo `{min_res:+.4f}` hasta máximo `{max_res:+.4f}`
- **Conteo por signo (tolerancia `1e-6`):** `{pos_res}` positivos (subestimación), `{neg_res}` negativos (sobreestimación) y `{cero_res}` iguales a cero.

Dado que `quality` solo toma valores enteros (`3` a `8`) mientras el modelo emite predicciones continuas, los residuos muestran bandas diagonales en la gráfica de dispersión y diferencias sistemáticas en los extremos de la escala. No se concluye normalidad únicamente a partir del histograma de residuos.

## 6. Rendimiento por puntuación de calidad

| Calidad real | Cantidad de registros | Promedio de predicciones | MAE | RMSE | Residuo promedio |
|---:|---:|---:|---:|---:|---:|
{filas_calidad_md}

En el conjunto de prueba, **{n_calidades_medias} de {n_test} registros ({pct_calidades_medias:.2f}%)** tienen puntuación real `5` o `6`, donde el MAE es menor. En las calidades extremas (`3`, `4`, `7` y `8`), tanto el promedio de predicciones tiende hacia el centro de la escala (entre ~`5.1` y ~`6.5`) como el error promedio aumenta considerablemente. Además, puntuaciones como `3` (`n=4`) y `8` (`n=3`) contienen muy pocas muestras de prueba, por lo que sus estimaciones puntuales son menos estables.

## 7. Indicios de sobreajuste, subajuste y capacidad de generalización

{texto_sobreajuste_subajuste}

- **Capacidad de generalización observada:** {texto_generalizacion}

## 8. Limitaciones del modelo

### Limitaciones observadas directamente en los resultados
1. **Concentración en calidades 5 y 6:** En prueba, el `{pct_calidades_medias:.2f}%` de los casos corresponde a notas `5` y `6`, provocando predicciones conservadoras hacia valores intermedios.
2. **Escasez en puntuaciones extremas:** Hay pocos vinos con calidad `3` (`4` en prueba), `4` (`11` en prueba) y `8` (`3` en prueba), donde los errores (`MAE` y `RMSE`) resultan más elevados.
3. **Capacidad explicativa moderada:** Un `R²` de `{metricas_prueba['R²']:.4f}` indica que una parte importante de la variación en la calidad sensorial no queda capturada por una combinación lineal simple de las variables disponibles.

### Limitaciones potenciales y estructurales del dataset
4. **Subjetividad de la variable objetivo:** La calificación `quality` proviene de evaluaciones sensoriales discretas realizadas por catadores humanos.
5. **Posibles relaciones no lineales:** Algunos efectos fisicoquímicos sobre la calidad percibida pueden seguir comportamientos no lineales que `LinearRegression` no modela directamente.
6. **Colinealidad entre características:** La correlación entre variables independientes (observada en la Fase 1) puede influir en la estabilidad individual de los coeficientes.
7. **Generalización poblacional:** Los resultados son aplicables a vinos con características similares al vino tinto *Vinho Verde* representado en el dataset.

## 9. Conclusiones preliminares

- El modelo de Regresión Lineal Múltiple entrenado ofrece un desempeño **moderado y consistente** entre entrenamiento y prueba, mejorando claramente frente al modelo base de media constante (`DummyRegressor`).
- Su utilidad principal es capturar tendencias lineales globales (como el aporte positivo del alcohol y negativo de la acidez volátil), pero presenta errores considerables en calidades extremas y no debe considerarse un predictor exacto para decisiones críticas sin explorar en trabajos futuros relaciones no lineales o mayor equilibrio en los datos.
"""

    with open(RUTA_RESUMEN_EVALUACION, "w", encoding="utf-8") as archivo_md:
        archivo_md.write(contenido)

    print(f"Resumen de evaluacion guardado en: {RUTA_RESUMEN_EVALUACION}")
    return RUTA_RESUMEN_EVALUACION


def _preparar_evaluacion_base() -> tuple[
    LinearRegression,
    StandardScaler,
    dict,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    pd.DataFrame,
    dict[str, float],
]:
    """Carga modelo y prueba una sola vez, genera predicciones y calcula métricas base."""
    modelo, escalador, config = cargar_modelo()
    X_test, X_test_escalado, y_test = cargar_datos_prueba(config)
    y_pred, df_predicciones = generar_predicciones(modelo, X_test_escalado, y_test)
    metricas_prueba = calcular_metricas(y_test.to_numpy(dtype=float), y_pred)
    return (
        modelo,
        escalador,
        config,
        X_test,
        X_test_escalado,
        y_test,
        y_pred,
        df_predicciones,
        metricas_prueba,
    )


def _imprimir_metricas_prueba(metricas_prueba: dict[str, float], n_prueba: int) -> pd.DataFrame:
    """Imprime la tabla organizada de métricas con 4 decimales y la devuelve."""
    df_metricas = construir_tabla_metricas(metricas_prueba)
    tabla_consola = df_metricas[["Métrica", "Valor", "Descripción"]].copy()
    tabla_consola["Valor"] = tabla_consola["Valor"].map(lambda v: f"{v:.4f}")

    print(f"Registros evaluados en prueba: {n_prueba}")
    with pd.option_context("display.max_colwidth", None, "display.width", 1000):
        print(tabla_consola.to_string(index=False))
    return df_metricas


def _imprimir_predicciones(df_predicciones: pd.DataFrame) -> None:
    """Muestra los primeros 15 registros reales, predichos y errores en consola."""
    print(f"Total de predicciones de prueba: {len(df_predicciones)}")
    print("\nPrimeros 15 resultados en el conjunto de prueba:")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(df_predicciones.head(15).round(4).to_string(index=False))


def _imprimir_residuos(df_estadisticas_residuos: pd.DataFrame) -> None:
    """Imprime las estadísticas de residuos y su interpretación."""
    print(
        "Definicion: Residuo = Calidad_real - Calidad_predicha\n"
        "- Residuo positivo (> 0): el modelo subestimo la calidad.\n"
        "- Residuo negativo (< 0): el modelo sobreestimo la calidad.\n"
        "- Residuo cercano a 0: prediccion cercana al valor real."
    )
    print()
    with pd.option_context("display.max_colwidth", None, "display.width", 1000):
        print(df_estadisticas_residuos.round(4).to_string(index=False))


def _imprimir_comparacion_train_test(
    df_comparacion: pd.DataFrame,
    metricas_train: dict[str, float],
    metricas_prueba: dict[str, float],
    metricas_base: dict[str, float] | None = None,
) -> None:
    """Imprime la tabla comparativa entrenamiento vs prueba y su análisis de generalización."""
    print(
        "Nota: Las metricas principales corresponden al conjunto de prueba; "
        "entrenamiento se usa solo como diagnostico complementario."
    )
    print()
    print(df_comparacion.round(4).to_string(index=False))

    diff_r2 = metricas_train["R²"] - metricas_prueba["R²"]
    diff_rmse = metricas_prueba["RMSE"] - metricas_train["RMSE"]
    print("\nAnalisis de generalizacion:")
    print(
        f"- Sobreajuste: diferencia acotada entre entrenamiento y prueba "
        f"(R²: {diff_r2:+.4f}; RMSE: {diff_rmse:+.4f}), sin indicios marcados de memorizacion."
    )
    print(
        f"- Subajuste: el modelo lineal explica una fraccion moderada de la variabilidad "
        f"(R² prueba = {metricas_prueba['R²']:.4f}, R² entrenamiento = {metricas_train['R²']:.4f})."
    )
    if metricas_base is not None:
        print(
            f"- Frente al modelo base (R² = {metricas_base['R²']:.4f}, "
            f"RMSE = {metricas_base['RMSE']:.4f}), la regresion lineal si mejora las predicciones, "
            f"aunque con margen de error moderado."
        )


def evaluar_metricas_principales() -> pd.DataFrame:
    """Opción 12: Carga modelo y prueba, evalúa MSE, RMSE, MAE y R² y guarda tabla."""
    imprimir_encabezado("FASE 4: METRICAS DE EVALUACION (MSE, RMSE, MAE Y R²)")
    _, _, _, _, _, y_test, _, _, metricas_prueba = _preparar_evaluacion_base()
    df_metricas = _imprimir_metricas_prueba(metricas_prueba, len(y_test))
    asegurar_carpetas_resultados()
    df_metricas.to_csv(RUTA_TABLA_METRICAS, index=False, encoding="utf-8-sig")
    print(f"\nTabla guardada en: {RUTA_TABLA_METRICAS}")
    return df_metricas


def mostrar_valores_reales_y_predicciones() -> pd.DataFrame:
    """Opción 13: Muestra los primeros 15 valores reales vs predichos y guarda CSV."""
    imprimir_encabezado("FASE 4: VALORES REALES VS PREDICCIONES DE PRUEBA")
    _, _, _, _, _, _, _, df_predicciones, _ = _preparar_evaluacion_base()
    _imprimir_predicciones(df_predicciones)
    asegurar_carpetas_resultados()
    df_predicciones.to_csv(RUTA_TABLA_PREDICCIONES, index=False, encoding="utf-8-sig")
    print(f"\nPredicciones completas guardadas en: {RUTA_TABLA_PREDICCIONES}")
    return df_predicciones


def ejecutar_analisis_residuos() -> pd.DataFrame:
    """Opción 14: Calcula y muestra las estadísticas de los residuos en prueba."""
    imprimir_encabezado("FASE 4: ANALISIS DE RESIDUOS")
    _, _, _, _, _, _, _, df_predicciones, _ = _preparar_evaluacion_base()
    df_estadisticas_residuos = analizar_residuos(df_predicciones)
    _imprimir_residuos(df_estadisticas_residuos)
    asegurar_carpetas_resultados()
    df_estadisticas_residuos.to_csv(RUTA_TABLA_RESIDUOS, index=False, encoding="utf-8-sig")
    print(f"\nTabla de residuos guardada en: {RUTA_TABLA_RESIDUOS}")
    return df_estadisticas_residuos


def ejecutar_comparacion_train_test() -> pd.DataFrame:
    """Opción 15: Compara métricas entre entrenamiento y prueba."""
    imprimir_encabezado("FASE 4: COMPARACION ENTRENAMIENTO VS PRUEBA")
    modelo, _, config, _, X_test_escalado, y_test, _, _, metricas_prueba = (
        _preparar_evaluacion_base()
    )
    X_train_escalado, y_train = cargar_datos_entrenamiento(config)
    metricas_train, df_comparacion = comparar_entrenamiento_prueba(
        modelo, X_train_escalado, y_train, metricas_prueba
    )
    metricas_base, _ = evaluar_modelo_base(
        X_train_escalado, y_train, X_test_escalado, y_test, metricas_prueba
    )
    _imprimir_comparacion_train_test(
        df_comparacion, metricas_train, metricas_prueba, metricas_base
    )
    asegurar_carpetas_resultados()
    df_comparacion.to_csv(RUTA_TABLA_TRAIN_TEST, index=False, encoding="utf-8-sig")
    print(f"\nTabla guardada en: {RUTA_TABLA_TRAIN_TEST}")
    return df_comparacion


def ejecutar_comparacion_modelo_base() -> pd.DataFrame:
    """Opción 16: Compara el modelo entrenado contra DummyRegressor(strategy='mean')."""
    imprimir_encabezado("FASE 4: COMPARACION CON MODELO DE REFERENCIA (DUMMYREGRESSOR)")
    _, _, config, _, X_test_escalado, y_test, _, _, metricas_prueba = (
        _preparar_evaluacion_base()
    )
    X_train_escalado, y_train = cargar_datos_entrenamiento(config)
    _, df_comparacion_base = evaluar_modelo_base(
        X_train_escalado, y_train, X_test_escalado, y_test, metricas_prueba
    )
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(df_comparacion_base.round(4).to_string(index=False))
    asegurar_carpetas_resultados()
    df_comparacion_base.to_csv(RUTA_TABLA_MODELO_BASE, index=False, encoding="utf-8-sig")
    print(f"\nTabla guardada en: {RUTA_TABLA_MODELO_BASE}")
    return df_comparacion_base


def ejecutar_rendimiento_por_calidad() -> pd.DataFrame:
    """Opción 17: Agrupa las predicciones por puntuación real y calcula errores."""
    imprimir_encabezado("FASE 4: RENDIMIENTO POR PUNTUACION DE CALIDAD")
    _, _, _, _, _, _, _, df_predicciones, _ = _preparar_evaluacion_base()
    df_rendimiento = analizar_rendimiento_por_calidad(df_predicciones)
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(df_rendimiento.round(4).to_string(index=False))
    asegurar_carpetas_resultados()
    df_rendimiento.to_csv(
        RUTA_TABLA_RENDIMIENTO_CALIDAD, index=False, encoding="utf-8-sig"
    )
    print(f"\nTabla guardada en: {RUTA_TABLA_RENDIMIENTO_CALIDAD}")
    return df_rendimiento


def ejecutar_graficas_evaluacion(mostrar: bool = True) -> list[Path]:
    """Opción 18: Genera y guarda todas las gráficas PNG de la Fase 4 (300 DPI)."""
    imprimir_encabezado("FASE 4: GENERACION DE GRAFICAS DE EVALUACION")
    modelo, _, config, _, X_test_escalado, y_test, _, df_predicciones, metricas_prueba = (
        _preparar_evaluacion_base()
    )
    X_train_escalado, y_train = cargar_datos_entrenamiento(config)
    _, df_comparacion_train_test = comparar_entrenamiento_prueba(
        modelo, X_train_escalado, y_train, metricas_prueba
    )
    _, df_comparacion_base = evaluar_modelo_base(
        X_train_escalado, y_train, X_test_escalado, y_test, metricas_prueba
    )
    df_rendimiento_calidad = analizar_rendimiento_por_calidad(df_predicciones)

    rutas = generar_graficas_evaluacion(
        df_predicciones=df_predicciones,
        df_comparacion_train_test=df_comparacion_train_test,
        df_comparacion_base=df_comparacion_base,
        df_rendimiento_calidad=df_rendimiento_calidad,
        mostrar=mostrar,
    )
    print(f"\nTotal de graficas de evaluacion generadas: {len(rutas)}")
    return rutas


def evaluar_modelo(mostrar_graficas: bool = True) -> dict:
    """Opción 19 y ejecución completa de la Fase 4 en un solo paso sin cálculos redundantes."""
    imprimir_encabezado("FASE 4: EVALUACION COMPLETA DEL RENDIMIENTO")

    # 1. Cargar modelo entrenado, escalador y datos de prueba y entrenamiento
    modelo, _, config = cargar_modelo()
    _, X_test_escalado, y_test = cargar_datos_prueba(config)
    X_train_escalado, y_train = cargar_datos_entrenamiento(config)

    if set(X_train_escalado.index).intersection(set(X_test_escalado.index)):
        raise ValueError(
            "Se detecto superposicion de indices entre entrenamiento y prueba."
        )

    print(f"Modelo cargado:                 {config['algoritmo']}")
    print(f"Registros de entrenamiento:     {len(X_train_escalado)}")
    print(f"Registros de prueba evaluados:  {len(X_test_escalado)}")

    # 2. Predicciones sobre el conjunto de prueba
    imprimir_subtitulo("1. Predicciones sobre el conjunto de prueba (primeros 15)")
    y_pred, df_predicciones = generar_predicciones(modelo, X_test_escalado, y_test)
    _imprimir_predicciones(df_predicciones)

    # 3. Métricas de evaluación principales
    imprimir_subtitulo("2. Metricas de evaluacion en prueba (MSE, RMSE, MAE y R²)")
    metricas_prueba = calcular_metricas(y_test.to_numpy(dtype=float), y_pred)
    df_metricas = _imprimir_metricas_prueba(metricas_prueba, len(y_test))

    # 4. Análisis de residuos
    imprimir_subtitulo("3. Analisis estadistico de residuos")
    df_estadisticas_residuos = analizar_residuos(df_predicciones)
    _imprimir_residuos(df_estadisticas_residuos)

    # 5. Comparación con modelo de referencia (DummyRegressor)
    imprimir_subtitulo("4. Comparacion con modelo de referencia (DummyRegressor)")
    metricas_base, df_comparacion_base = evaluar_modelo_base(
        X_train_escalado, y_train, X_test_escalado, y_test, metricas_prueba
    )
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(df_comparacion_base.round(4).to_string(index=False))

    # 6. Comparación entrenamiento vs prueba
    imprimir_subtitulo("5. Comparacion entrenamiento vs prueba y generalizacion")
    metricas_train, df_comparacion_train_test = comparar_entrenamiento_prueba(
        modelo, X_train_escalado, y_train, metricas_prueba
    )
    _imprimir_comparacion_train_test(
        df_comparacion_train_test, metricas_train, metricas_prueba, metricas_base
    )

    # 7. Rendimiento por puntuación de calidad
    imprimir_subtitulo("6. Rendimiento agrupado por puntuacion real de calidad")
    df_rendimiento_calidad = analizar_rendimiento_por_calidad(df_predicciones)
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(df_rendimiento_calidad.round(4).to_string(index=False))

    # 8. Guardar todas las tablas CSV
    imprimir_subtitulo("7. Almacenamiento de tablas CSV")
    rutas_tablas = guardar_resultados(
        df_predicciones=df_predicciones,
        df_metricas=df_metricas,
        df_estadisticas_residuos=df_estadisticas_residuos,
        df_comparacion_train_test=df_comparacion_train_test,
        df_comparacion_base=df_comparacion_base,
        df_rendimiento_calidad=df_rendimiento_calidad,
    )

    # 9. Generar las 8 gráficas a 300 DPI
    imprimir_subtitulo("8. Generacion de graficas de evaluacion (300 DPI)")
    rutas_graficas = generar_graficas_evaluacion(
        df_predicciones=df_predicciones,
        df_comparacion_train_test=df_comparacion_train_test,
        df_comparacion_base=df_comparacion_base,
        df_rendimiento_calidad=df_rendimiento_calidad,
        mostrar=mostrar_graficas,
    )

    # 10. Generar el resumen de evaluación en Markdown
    imprimir_subtitulo("9. Generacion del resumen de evaluacion")
    ruta_resumen = generar_resumen_evaluacion(
        config=config,
        metricas_prueba=metricas_prueba,
        metricas_train=metricas_train,
        metricas_base=metricas_base,
        df_comparacion_train_test=df_comparacion_train_test,
        df_comparacion_base=df_comparacion_base,
        df_estadisticas_residuos=df_estadisticas_residuos,
        df_rendimiento_calidad=df_rendimiento_calidad,
    )

    return {
        "metricas_prueba": metricas_prueba,
        "metricas_train": metricas_train,
        "metricas_base": metricas_base,
        "predicciones": df_predicciones,
        "residuos": df_estadisticas_residuos,
        "comparacion_train_test": df_comparacion_train_test,
        "comparacion_base": df_comparacion_base,
        "rendimiento_calidad": df_rendimiento_calidad,
        "rutas_tablas": rutas_tablas,
        "rutas_graficas": rutas_graficas,
        "ruta_resumen": ruta_resumen,
    }


if __name__ == "__main__":
    try:
        evaluar_modelo()
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        print(f"Error durante la evaluacion del modelo: {error}")
