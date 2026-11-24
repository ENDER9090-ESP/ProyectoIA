"""Análisis estadístico y visualización del dataset limpio Wine Quality (Red Wine).

Calcula estadísticas descriptivas, distribución de calidad, matriz de correlación,
correlaciones con 'quality' y detección de valores atípicos mediante IQR, además
de generar y guardar tablas CSV y gráficas PNG a 300 DPI en la carpeta resultados/.
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns

from utilidades import (
    COLUMNA_OBJETIVO,
    RUTA_DATASET_LIMPIO,
    RUTA_GRAFICAS,
    RUTA_TABLAS,
    asegurar_carpetas_resultados,
    cargar_dataset,
    existe_dataset_limpio,
    imprimir_encabezado,
    imprimir_subtitulo,
)


def _configurar_estilo_graficas() -> None:
    """Configura un estilo visual limpio y legible para todas las gráficas."""
    sns.set_theme(style="whitegrid")


def _guardar_y_mostrar_figura(
    figura: plt.Figure,
    nombre_archivo: str,
    mostrar: bool = True,
) -> Path:
    """Guarda la figura en PNG a 300 DPI, la muestra si hay backend gráfico y la cierra."""
    asegurar_carpetas_resultados()
    ruta_salida = RUTA_GRAFICAS / nombre_archivo
    figura.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    print(f"Grafica guardada (300 DPI): {ruta_salida}")

    if mostrar and matplotlib.get_backend().lower() != "agg":
        plt.show()

    plt.close(figura)
    return ruta_salida


def _contar_atipicos_serie(serie: pd.Series) -> tuple[int, float, float, float, float, float]:
    """Calcula Q1, Q3, IQR, límites y cantidad de posibles atípicos según método IQR."""
    q1 = float(serie.quantile(0.25))
    q3 = float(serie.quantile(0.75))
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    mascara_atipicos = (serie < limite_inferior) | (serie > limite_superior)
    cantidad = int(mascara_atipicos.sum())
    return cantidad, q1, q3, iqr, limite_inferior, limite_superior


def calcular_estadisticas_descriptivas(datos: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadísticas descriptivas con describe().T y guarda la tabla CSV."""
    estadisticas = datos.describe().T.rename(
        columns={
            "count": "Cantidad",
            "mean": "Media",
            "std": "Desviación estándar",
            "min": "Mínimo",
            "25%": "Q1 (25%)",
            "50%": "Mediana (50%)",
            "75%": "Q3 (75%)",
            "max": "Máximo",
        }
    )

    asegurar_carpetas_resultados()
    ruta_csv = RUTA_TABLAS / "estadisticas_descriptivas.csv"
    estadisticas.to_csv(ruta_csv, index=True, index_label="Variable", encoding="utf-8-sig")
    return estadisticas


def calcular_distribucion_calidad(datos: pd.DataFrame) -> pd.DataFrame:
    """Calcula la cantidad y porcentaje de vinos por puntuación de calidad."""
    conteos = datos[COLUMNA_OBJETIVO].value_counts().sort_index()
    total = len(datos)

    distribucion = pd.DataFrame(
        {
            "Calidad": conteos.index,
            "Cantidad": conteos.values,
            "Porcentaje (%)": ((conteos.values / total) * 100).round(2),
        }
    )

    asegurar_carpetas_resultados()
    ruta_csv = RUTA_TABLAS / "distribucion_calidad.csv"
    distribucion.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    return distribucion


def calcular_matriz_correlacion(datos: pd.DataFrame) -> pd.DataFrame:
    """Calcula la matriz de correlación de Pearson y la guarda en CSV."""
    matriz = datos.corr(numeric_only=True)

    asegurar_carpetas_resultados()
    ruta_csv = RUTA_TABLAS / "matriz_correlacion.csv"
    matriz.to_csv(ruta_csv, index=True, index_label="Variable", encoding="utf-8-sig")
    return matriz


def calcular_correlaciones_objetivo(matriz_correlacion: pd.DataFrame) -> pd.DataFrame:
    """Extrae y ordena la correlación de cada variable independiente con quality."""
    correlaciones = (
        matriz_correlacion[COLUMNA_OBJETIVO]
        .drop(labels=[COLUMNA_OBJETIVO])
        .sort_values(ascending=False)
    )

    tabla_correlaciones = pd.DataFrame(
        {
            "Variable": correlaciones.index,
            "Correlación con quality": correlaciones.values,
        }
    )

    asegurar_carpetas_resultados()
    ruta_csv = RUTA_TABLAS / "correlaciones_con_quality.csv"
    tabla_correlaciones.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    return tabla_correlaciones


def detectar_valores_atipicos_iqr(datos: pd.DataFrame) -> pd.DataFrame:
    """Detecta posibles valores atípicos con el método IQR sin eliminar registros."""
    variables_independientes = datos.drop(columns=[COLUMNA_OBJETIVO])
    total_registros = len(datos)
    filas_resumen = []

    for columna in variables_independientes.columns:
        cantidad, q1, q3, iqr, lim_inf, lim_sup = _contar_atipicos_serie(
            variables_independientes[columna]
        )
        porcentaje = round((cantidad / total_registros) * 100, 2)
        filas_resumen.append(
            {
                "Variable": columna,
                "Q1": round(q1, 4),
                "Q3": round(q3, 4),
                "IQR": round(iqr, 4),
                "Límite inferior": round(lim_inf, 4),
                "Límite superior": round(lim_sup, 4),
                "Valores atípicos": cantidad,
                "Porcentaje (%)": porcentaje,
            }
        )

    tabla_atipicos = pd.DataFrame(filas_resumen)

    asegurar_carpetas_resultados()
    ruta_csv = RUTA_TABLAS / "valores_atipicos_iqr.csv"
    tabla_atipicos.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    return tabla_atipicos


def realizar_analisis_estadistico(
    ruta_archivo: Path = RUTA_DATASET_LIMPIO,
) -> dict[str, pd.DataFrame]:
    """Ejecuta todos los análisis estadísticos sobre el dataset limpio y guarda las tablas."""
    imprimir_encabezado("ANALISIS ESTADISTICO DEL DATASET LIMPIO")
    asegurar_carpetas_resultados()

    datos = cargar_dataset(ruta_archivo)
    print(f"Archivo cargado: {ruta_archivo.name} -> Dimensiones: {datos.shape}")

    # 1. Estadísticas descriptivas
    imprimir_subtitulo("1. Estadisticas descriptivas (describe().T)")
    estadisticas = calcular_estadisticas_descriptivas(datos)
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(estadisticas.round(4))
    print(f"Tabla guardada en: {RUTA_TABLAS / 'estadisticas_descriptivas.csv'}")

    # 2. Distribución de calidad
    imprimir_subtitulo("2. Distribucion de la calidad del vino")
    distribucion = calcular_distribucion_calidad(datos)
    print(distribucion.to_string(index=False))
    print(f"Tabla guardada en: {RUTA_TABLAS / 'distribucion_calidad.csv'}")

    # 3. Matriz de correlación de Pearson
    imprimir_subtitulo("3. Matriz de correlacion de Pearson")
    matriz = calcular_matriz_correlacion(datos)
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(matriz.round(2))
    print(f"Tabla guardada en: {RUTA_TABLAS / 'matriz_correlacion.csv'}")

    # 4. Correlación de cada variable independiente con quality
    imprimir_subtitulo("4. Correlacion de variables independientes con 'quality'")
    correlaciones_obj = calcular_correlaciones_objetivo(matriz)
    print(correlaciones_obj.round(4).to_string(index=False))
    print(
        "\nNota metodologica: No se eliminan variables basandose exclusivamente en "
        "su correlacion lineal individual, ya que pueden aportar informacion en conjunto."
    )
    print(f"Tabla guardada en: {RUTA_TABLAS / 'correlaciones_con_quality.csv'}")

    # 5. Identificación de valores atípicos (IQR)
    imprimir_subtitulo("5. Identificacion de posibles valores atipicos (metodo IQR)")
    tabla_atipicos = detectar_valores_atipicos_iqr(datos)
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(tabla_atipicos.to_string(index=False))
    print(
        "\nNota metodologica: Todos los registros con valores atipicos se conservan "
        "intactos porque pueden corresponder a mediciones fisicoquimicas reales."
    )
    print(f"Tabla guardada en: {RUTA_TABLAS / 'valores_atipicos_iqr.csv'}")

    return {
        "estadisticas_descriptivas": estadisticas,
        "distribucion_calidad": distribucion,
        "matriz_correlacion": matriz,
        "correlaciones_objetivo": correlaciones_obj,
        "valores_atipicos": tabla_atipicos,
    }


def graficar_distribucion_calidad(datos: pd.DataFrame, mostrar: bool = True) -> Path:
    """Genera y guarda la gráfica de barras de distribución de la calidad del vino."""
    _configurar_estilo_graficas()
    conteos = datos[COLUMNA_OBJETIVO].value_counts().sort_index()

    figura, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(
        conteos.index.astype(str),
        conteos.values,
        color="#2b6cb0",
        edgecolor="#1a365d",
        linewidth=0.8,
    )

    for barra in barras:
        altura = int(barra.get_height())
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            altura + 8,
            f"{altura}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(
        "Distribución de la puntuación de calidad del vino tinto (dataset limpio)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Puntuación de calidad (quality)", fontsize=11)
    ax.set_ylabel("Cantidad de vinos", fontsize=11)
    ax.set_ylim(0, max(conteos.values) * 1.12)
    ax.grid(axis="x", visible=False)
    figura.tight_layout()

    return _guardar_y_mostrar_figura(figura, "01_distribucion_calidad.png", mostrar=mostrar)


def graficar_matriz_correlacion(datos: pd.DataFrame, mostrar: bool = True) -> Path:
    """Genera y guarda el mapa de calor de correlaciones de Pearson."""
    _configurar_estilo_graficas()
    matriz = datos.corr(numeric_only=True)

    figura, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        matriz,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Correlación de Pearson", "shrink": 0.85},
        ax=ax,
    )

    ax.set_title(
        "Matriz de correlación de Pearson entre variables fisicoquímicas y calidad",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    figura.tight_layout()

    return _guardar_y_mostrar_figura(figura, "02_matriz_correlacion.png", mostrar=mostrar)


def graficar_correlacion_objetivo(datos: pd.DataFrame, mostrar: bool = True) -> Path:
    """Genera y guarda barras horizontales con la correlación respecto a quality."""
    _configurar_estilo_graficas()
    correlaciones = (
        datos.corr(numeric_only=True)[COLUMNA_OBJETIVO]
        .drop(labels=[COLUMNA_OBJETIVO])
        .sort_values(ascending=True)
    )

    colores = ["#2b6cb0" if valor >= 0 else "#c53030" for valor in correlaciones.values]

    figura, ax = plt.subplots(figsize=(9, 6))
    barras = ax.barh(correlaciones.index, correlaciones.values, color=colores, height=0.65)
    ax.axvline(0, color="#2d3748", linewidth=1)

    for barra, valor in zip(barras, correlaciones.values):
        desplazamiento = 0.015 if valor >= 0 else -0.015
        alineacion = "left" if valor >= 0 else "right"
        ax.text(
            valor + desplazamiento,
            barra.get_y() + barra.get_height() / 2,
            f"{valor:+.2f}",
            va="center",
            ha=alineacion,
            fontsize=9.5,
            fontweight="bold",
        )

    leyenda = [
        Patch(facecolor="#2b6cb0", label="Correlación positiva"),
        Patch(facecolor="#c53030", label="Correlación negativa"),
    ]
    ax.legend(handles=leyenda, loc="lower right")
    ax.set_xlim(-0.55, 0.65)
    ax.set_title(
        "Correlación de las variables independientes con la calidad (quality)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Coeficiente de correlación de Pearson", fontsize=11)
    ax.set_ylabel("Variable fisicoquímica", fontsize=11)
    ax.grid(axis="y", visible=False)
    figura.tight_layout()

    return _guardar_y_mostrar_figura(figura, "03_correlacion_con_quality.png", mostrar=mostrar)


def graficar_boxplots_atipicos(datos: pd.DataFrame, mostrar: bool = True) -> Path:
    """Genera diagramas de caja para las 11 variables independientes e indica conteo IQR."""
    _configurar_estilo_graficas()
    variables = datos.drop(columns=[COLUMNA_OBJETIVO]).columns.tolist()

    figura, ejes = plt.subplots(3, 4, figsize=(16, 11))
    ejes_planos = ejes.flatten()

    for indice, columna in enumerate(variables):
        ax = ejes_planos[indice]
        conteo_atipicos, *_ = _contar_atipicos_serie(datos[columna])
        sns.boxplot(y=datos[columna], ax=ax, color="#4299e1", width=0.45)
        ax.set_title(f"{columna}\n(Atípicos IQR: {conteo_atipicos})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Valor", fontsize=9)
        ax.set_xlabel("")

    # Ocultar el último panel sobrante de la cuadrícula 3x4 (11 variables)
    ejes_planos[-1].axis("off")

    figura.suptitle(
        "Diagramas de caja (boxplots) de las 11 variables independientes",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    figura.tight_layout()

    return _guardar_y_mostrar_figura(figura, "04_boxplots_variables.png", mostrar=mostrar)


def graficar_histogramas(datos: pd.DataFrame, mostrar: bool = True) -> Path:
    """Genera histogramas de distribución para las 11 variables independientes."""
    _configurar_estilo_graficas()
    variables = datos.drop(columns=[COLUMNA_OBJETIVO]).columns.tolist()

    figura, ejes = plt.subplots(3, 4, figsize=(16, 11))
    ejes_planos = ejes.flatten()

    for indice, columna in enumerate(variables):
        ax = ejes_planos[indice]
        sns.histplot(datos[columna], kde=True, ax=ax, color="#2b6cb0")
        ax.set_title(columna, fontsize=10, fontweight="bold")
        ax.set_xlabel("Valor", fontsize=9)
        ax.set_ylabel("Frecuencia", fontsize=9)

    # Ocultar el último panel sobrante de la cuadrícula 3x4
    ejes_planos[-1].axis("off")

    figura.suptitle(
        "Histogramas de distribución de las 11 variables independientes",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    figura.tight_layout()

    return _guardar_y_mostrar_figura(figura, "05_histogramas_variables.png", mostrar=mostrar)


def graficar_dispersion_calidad(datos: pd.DataFrame, mostrar: bool = True) -> Path:
    """Genera gráficas de dispersión entre variables fisicoquímicas clave y calidad."""
    _configurar_estilo_graficas()

    pares_variables = [
        ("alcohol", "Alcohol vs. Calidad del vino", "Alcohol (% vol.) [alcohol]"),
        ("volatile acidity", "Acidez volátil vs. Calidad del vino", "Acidez volátil (g/dm³) [volatile acidity]"),
        ("sulphates", "Sulfatos vs. Calidad del vino", "Sulfatos (g/dm³) [sulphates]"),
        ("citric acid", "Ácido cítrico vs. Calidad del vino", "Ácido cítrico (g/dm³) [citric acid]"),
    ]

    figura, ejes = plt.subplots(2, 2, figsize=(12, 9))
    ejes_planos = ejes.flatten()

    for ax, (columna, titulo, etiqueta_x) in zip(ejes_planos, pares_variables):
        ax.scatter(
            datos[columna],
            datos[COLUMNA_OBJETIVO],
            alpha=0.5,
            color="#2b6cb0",
            edgecolors="white",
            linewidth=0.3,
        )
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.set_xlabel(etiqueta_x, fontsize=10)
        ax.set_ylabel("Calidad del vino (quality)", fontsize=10)

    figura.suptitle(
        "Relación entre variables fisicoquímicas seleccionadas y la calidad del vino",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    figura.tight_layout()

    return _guardar_y_mostrar_figura(figura, "06_dispersion_calidad.png", mostrar=mostrar)


def generar_graficas(
    ruta_archivo: Path = RUTA_DATASET_LIMPIO,
    mostrar: bool = True,
) -> list[Path]:
    """Carga el dataset limpio y genera todas las gráficas PNG a 300 DPI."""
    imprimir_encabezado("GENERACION DE GRAFICAS DEL DATASET LIMPIO")
    asegurar_carpetas_resultados()

    datos = cargar_dataset(ruta_archivo)
    print(f"Archivo cargado: {ruta_archivo.name} -> Dimensiones: {datos.shape}\n")

    rutas_generadas = [
        graficar_distribucion_calidad(datos, mostrar=mostrar),
        graficar_matriz_correlacion(datos, mostrar=mostrar),
        graficar_correlacion_objetivo(datos, mostrar=mostrar),
        graficar_boxplots_atipicos(datos, mostrar=mostrar),
        graficar_histogramas(datos, mostrar=mostrar),
        graficar_dispersion_calidad(datos, mostrar=mostrar),
    ]

    print(f"\nTotal de graficas generadas: {len(rutas_generadas)}")
    return rutas_generadas


if __name__ == "__main__":
    try:
        if not existe_dataset_limpio():
            print(
                "Error: Aun no existe 'winequality-red-limpio.csv'. "
                "Ejecute primero 'Limpieza_Datos.py' para generar el dataset limpio."
            )
        else:
            realizar_analisis_estadistico(RUTA_DATASET_LIMPIO)
            generar_graficas(RUTA_DATASET_LIMPIO)
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        print(f"Error durante el analisis y visualizacion: {error}")
