"""Entrenamiento del modelo de Regresión Lineal Múltiple (Fase 3).

Entrena LinearRegression de scikit-learn exclusivamente con X_train_escalado e
y_train mediante mínimos cuadrados ordinarios (OLS), muestra la ecuación y los
coeficientes obtenidos, genera la gráfica de coeficientes y almacena el modelo,
el escalador y la configuración JSON verificando su recarga con joblib.
"""

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

from utilidades import (
    COLUMNA_OBJETIVO,
    COLUMNAS_INDEPENDIENTES,
    PROPORCION_ENTRENAMIENTO,
    PROPORCION_PRUEBA,
    RUTA_CONFIG_MODELO,
    RUTA_DATASET_LIMPIO,
    RUTA_ESCALADOR_JOBLIB,
    RUTA_MODELO_JOBLIB,
    RUTA_TABLA_COEFICIENTES,
    RUTA_X_TEST_ESCALADO,
    RUTA_X_TRAIN_ESCALADO,
    RUTA_Y_TEST,
    RUTA_Y_TRAIN,
    SEMILLA_ALEATORIA,
    asegurar_carpeta_modelos,
    asegurar_carpetas_resultados,
    cargar_tabla_procesada,
    datos_escalados_actualizados,
    existen_datos_escalados,
    existe_modelo_entrenado,
    guardar_y_mostrar_figura,
    imprimir_encabezado,
    imprimir_subtitulo,
    modelo_entrenado_actualizado,
)


def construir_ecuacion_regresion(
    intercepto: float,
    variables: list[str],
    coeficientes: np.ndarray,
) -> str:
    """Construye la representación textual de la ecuación de regresión lineal múltiple."""
    terminos = [f"{intercepto:.4f}"]
    for variable, coef in zip(variables, coeficientes):
        signo = "+" if coef >= 0 else "-"
        terminos.append(f"{signo} {abs(float(coef)):.4f}*({variable})")
    return "y_pred = " + " ".join(terminos)


def construir_tabla_coeficientes(
    variables: list[str],
    coeficientes: np.ndarray,
) -> pd.DataFrame:
    """Organiza los coeficientes en un DataFrame con valor absoluto y signo."""
    valores = [float(c) for c in coeficientes]
    return pd.DataFrame(
        {
            "Variable": variables,
            "Coeficiente": valores,
            "Valor absoluto del coeficiente": [abs(v) for v in valores],
            "Signo del coeficiente": [
                "Positivo" if v >= 0 else "Negativo" for v in valores
            ],
        }
    )


def imprimir_notas_interpretacion() -> None:
    """Imprime las pautas académicas para interpretar correctamente el modelo."""
    imprimir_subtitulo("Notas metodologicas de interpretacion")
    print(
        "1. Coeficiente positivo: indica una relacion positiva estimada con la calidad,\n"
        "   manteniendo constantes las demas variables."
    )
    print(
        "2. Coeficiente negativo: indica una relacion negativa estimada con la calidad,\n"
        "   manteniendo constantes las demas variables."
    )
    print(
        "3. Escala estandarizada: como las caracteristicas X fueron estandarizadas,\n"
        "   cada coeficiente representa el cambio estimado en 'quality' ante un aumento\n"
        "   de una desviacion estandar en esa variable."
    )
    print(
        "4. Intercepto (b0): representa la puntuacion de calidad estimada cuando todas\n"
        "   las caracteristicas escaladas valen 0 (es decir, en la media de entrenamiento)."
    )
    print(
        "5. Causalidad: los coeficientes estimados muestran asociacion lineal y NO\n"
        "   demuestran causalidad directa."
    )
    print(
        "6. Colinealidad: la correlacion entre variables independientes (como acidez fija,\n"
        "   pH y densidad) puede afectar la magnitud y signo individual de los coeficientes."
    )


def graficar_coeficientes_regresion(
    tabla_coeficientes: pd.DataFrame,
    mostrar: bool = True,
) -> Path:
    """Genera una gráfica de barras horizontales con los coeficientes del modelo."""
    sns.set_theme(style="whitegrid")
    datos_ordenados = tabla_coeficientes.sort_values(
        by="Coeficiente", ascending=True
    ).reset_index(drop=True)

    variables = datos_ordenados["Variable"].tolist()
    valores = datos_ordenados["Coeficiente"].values
    colores = ["#2b6cb0" if v >= 0 else "#c53030" for v in valores]

    figura, ax = plt.subplots(figsize=(10, 6.5))
    barras = ax.barh(variables, valores, color=colores, height=0.65)
    ax.axvline(0, color="#2d3748", linewidth=1.1)

    min_val = float(min(valores))
    max_val = float(max(valores))
    rango = max(max_val - min_val, 0.2)
    margen = rango * 0.22

    for barra, valor in zip(barras, valores):
        desplazamiento = rango * 0.02 if valor >= 0 else -rango * 0.02
        alineacion = "left" if valor >= 0 else "right"
        ax.text(
            valor + desplazamiento,
            barra.get_y() + barra.get_height() / 2,
            f"{valor:+.4f}",
            va="center",
            ha=alineacion,
            fontsize=9.5,
            fontweight="bold",
        )

    leyenda = [
        Patch(facecolor="#2b6cb0", label="Coeficiente positivo"),
        Patch(facecolor="#c53030", label="Coeficiente negativo"),
    ]
    ax.legend(handles=leyenda, loc="lower right")
    ax.set_xlim(min_val - margen, max_val + margen)
    ax.set_title(
        "Coeficientes estandarizados del modelo de Regresión Lineal Múltiple",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Valor del coeficiente estandarizado", fontsize=11)
    ax.set_ylabel("Variable fisicoquímica", fontsize=11)
    ax.grid(axis="y", visible=False)
    figura.tight_layout()

    return guardar_y_mostrar_figura(
        figura, "12_coeficientes_regresion.png", mostrar=mostrar
    )


def entrenar_modelo(
    mostrar_graficas: bool = True,
) -> tuple[LinearRegression, pd.DataFrame]:
    """Entrena LinearRegression solo con datos de entrenamiento y guarda el modelo."""
    imprimir_encabezado("FASE 3: ENTRENAMIENTO DE REGRESION LINEAL MULTIPLE")
    asegurar_carpetas_resultados()
    asegurar_carpeta_modelos()

    if not existen_datos_escalados():
        raise FileNotFoundError(
            "No se encontraron los datos escalados o el escalador. "
            "Ejecute primero 'Escalado_Datos.py'."
        )
    if not datos_escalados_actualizados():
        raise ValueError(
            "Los datos escalados estan desactualizados frente a la particion actual. "
            "Vuelva a ejecutar 'Escalado_Datos.py'."
        )
    if not (RUTA_Y_TRAIN.is_file() and RUTA_Y_TEST.is_file()):
        raise FileNotFoundError(
            "No se encontraron los archivos y_train.csv o y_test.csv. "
            "Ejecute primero 'Particion_Datos.py'."
        )

    # 1. Cargar exclusivamente datos de entrenamiento para entrenar el modelo
    X_train_escalado = cargar_tabla_procesada(RUTA_X_TRAIN_ESCALADO)
    y_train_df = cargar_tabla_procesada(RUTA_Y_TRAIN)
    y_train = y_train_df[COLUMNA_OBJETIVO]

    # X_test_escalado solo se consulta para verificar integridad y registrar su tamaño en JSON;
    # nunca se utiliza para entrenar, ajustar ni evaluar en esta fase.
    X_test_escalado = cargar_tabla_procesada(RUTA_X_TEST_ESCALADO)

    if list(X_train_escalado.columns) != COLUMNAS_INDEPENDIENTES:
        raise ValueError(
            "El orden o nombre de las columnas en X_train_escalado no coincide con el esperado."
        )
    if not X_train_escalado.index.equals(y_train.index):
        raise ValueError(
            "Los indices de X_train_escalado e y_train no coinciden."
        )
    if set(X_train_escalado.index).intersection(set(X_test_escalado.index)):
        raise ValueError(
            "Se detecto superposicion de indices entre entrenamiento y prueba."
        )

    # 2. Fundamento matemático del algoritmo
    imprimir_subtitulo("1. Fundamento matematico (Minimos Cuadrados Ordinarios)")
    print("Ecuacion general: y_pred = b0 + b1*x1 + b2*x2 + ... + b11*x11")
    print(
        "El algoritmo LinearRegression utiliza el metodo de minimos cuadrados "
        "ordinarios (OLS) para calcular el intercepto (b0) y los coeficientes (b1..b11)\n"
        "que minimizan la suma de los errores cuadrados sobre el conjunto de entrenamiento."
    )

    # 3. Creación y entrenamiento exclusivo con (X_train_escalado, y_train)
    modelo = LinearRegression()
    modelo.fit(X_train_escalado, y_train)

    variables = list(X_train_escalado.columns)
    tabla_original = construir_tabla_coeficientes(variables, modelo.coef_)
    tabla_ordenada = tabla_original.sort_values(
        by="Valor absoluto del coeficiente", ascending=False
    ).reset_index(drop=True)

    ecuacion = construir_ecuacion_regresion(
        float(modelo.intercept_), variables, modelo.coef_
    )

    # 4. Mostrar información del modelo entrenado
    imprimir_subtitulo("2. Informacion del modelo entrenado")
    print("Algoritmo utilizado:           Regresion Lineal Multiple (LinearRegression)")
    print(f"Registros de entrenamiento:    {len(X_train_escalado)}")
    print(f"Cantidad de caracteristicas:   {X_train_escalado.shape[1]}")
    print(f"Intercepto (modelo.intercept_): {float(modelo.intercept_):.6f}")

    print("\nCoeficientes en orden original de las variables (modelo.coef_):")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(tabla_original.round(6).to_string(index=False))

    print("\nCoeficientes ordenados por valor absoluto (mayor a menor impacto):")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(tabla_ordenada.round(6).to_string(index=False))

    imprimir_subtitulo("3. Ecuacion de regresion lineal multiple obtenida")
    print(ecuacion)

    imprimir_notas_interpretacion()

    print("\nConfirmacion: El entrenamiento del modelo finalizo correctamente.")

    # 5. Guardar tabla de coeficientes y gráfica a 300 DPI
    imprimir_subtitulo("4. Almacenamiento de tabla, grafica y artefactos del modelo")
    tabla_ordenada.to_csv(
        RUTA_TABLA_COEFICIENTES, index=False, encoding="utf-8-sig"
    )
    print(f"Tabla de coeficientes guardada en: {RUTA_TABLA_COEFICIENTES}")

    graficar_coeficientes_regresion(tabla_original, mostrar=mostrar_graficas)

    # 6. Guardar modelo entrenado y configuración JSON
    joblib.dump(modelo, RUTA_MODELO_JOBLIB)
    print(f"Modelo guardado en: {RUTA_MODELO_JOBLIB}")

    registros_train = int(len(X_train_escalado))
    registros_test = int(len(X_test_escalado))
    total_registros = registros_train + registros_test

    configuracion = {
        "algoritmo": "Regresión Lineal Múltiple (LinearRegression)",
        "dataset": RUTA_DATASET_LIMPIO.name,
        "variable_objetivo": COLUMNA_OBJETIVO,
        "orden_variables_independientes": variables,
        "registros_entrenamiento": registros_train,
        "registros_prueba": registros_test,
        "semilla_aleatoria": SEMILLA_ALEATORIA,
        "porcentaje_entrenamiento": round(PROPORCION_ENTRENAMIENTO * 100, 2),
        "porcentaje_prueba": round(PROPORCION_PRUEBA * 100, 2),
        "porcentaje_entrenamiento_real": round(
            (registros_train / total_registros) * 100, 2
        ),
        "porcentaje_prueba_real": round(
            (registros_test / total_registros) * 100, 2
        ),
        "metodo_escalado": "StandardScaler",
        "intercepto": float(modelo.intercept_),
    }

    with open(RUTA_CONFIG_MODELO, "w", encoding="utf-8") as archivo_json:
        json.dump(configuracion, archivo_json, indent=4, ensure_ascii=False)
    print(f"Configuracion del modelo guardada en: {RUTA_CONFIG_MODELO}")

    # 7. Comprobación de recarga mediante joblib.load()
    modelo_recargado = joblib.load(RUTA_MODELO_JOBLIB)
    escalador_recargado = joblib.load(RUTA_ESCALADOR_JOBLIB)

    coincide_coef = bool(np.allclose(modelo_recargado.coef_, modelo.coef_))
    coincide_intercept = bool(
        np.isclose(modelo_recargado.intercept_, modelo.intercept_)
    )
    coincide_orden = (
        list(escalador_recargado.feature_names_in_) == COLUMNAS_INDEPENDIENTES
    )

    if not (coincide_coef and coincide_intercept and coincide_orden):
        raise ValueError(
            "Error al verificar la recarga del modelo o escalador mediante joblib."
        )

    print(
        "Verificacion joblib: modelo y escalador recargados correctamente "
        "conservando parametros y orden de caracteristicas."
    )

    return modelo, tabla_ordenada


def mostrar_coeficientes_modelo() -> pd.DataFrame:
    """Carga el modelo guardado y muestra en consola el intercepto, coeficientes y ecuación."""
    imprimir_encabezado("COEFICIENTES DEL MODELO ENTRENADO")

    if not existe_modelo_entrenado():
        raise FileNotFoundError(
            "No se encontro el modelo entrenado. Ejecute primero el entrenamiento (opcion 10)."
        )
    if not modelo_entrenado_actualizado():
        raise ValueError(
            "El modelo guardado esta desactualizado respecto a los datos escalados. "
            "Vuelva a entrenar el modelo (opcion 10)."
        )

    modelo = joblib.load(RUTA_MODELO_JOBLIB)
    with open(RUTA_CONFIG_MODELO, "r", encoding="utf-8") as archivo_json:
        config = json.load(archivo_json)

    variables = config["orden_variables_independientes"]
    tabla_original = construir_tabla_coeficientes(variables, modelo.coef_)
    tabla_ordenada = tabla_original.sort_values(
        by="Valor absoluto del coeficiente", ascending=False
    ).reset_index(drop=True)
    ecuacion = construir_ecuacion_regresion(
        float(modelo.intercept_), variables, modelo.coef_
    )

    print(f"Algoritmo:                  {config['algoritmo']}")
    print(f"Registros de entrenamiento: {config['registros_entrenamiento']}")
    print(f"Intercepto (b0):            {float(modelo.intercept_):.6f}")

    imprimir_subtitulo("Coeficientes ordenados por valor absoluto")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(tabla_ordenada.round(6).to_string(index=False))

    imprimir_subtitulo("Ecuacion de regresion lineal multiple")
    print(ecuacion)

    imprimir_notas_interpretacion()
    return tabla_ordenada


if __name__ == "__main__":
    try:
        if not existen_datos_escalados():
            print(
                "Error: Aun no existen los archivos escalados. "
                "Ejecute primero 'Escalado_Datos.py'."
            )
        else:
            entrenar_modelo()
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        print(f"Error durante el entrenamiento del modelo: {error}")
