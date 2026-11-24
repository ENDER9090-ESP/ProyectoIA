"""Utilidades compartidas para el proyecto de calidad del vino tinto.

Centraliza rutas relativas del proyecto, constantes de partición, formato
de consola, carga segura de archivos CSV, guardado de gráficas y control
de vigencia entre fases para evitar trabajar con archivos desactualizados.
"""

from pathlib import Path
import pandas as pd

# Ruta base del proyecto (carpeta donde reside este archivo)
RUTA_BASE = Path(__file__).resolve().parent

# Rutas de los archivos de datos originales y limpios (Fase 1)
RUTA_DATASET_ORIGINAL = RUTA_BASE / "winequality-red.csv"
RUTA_DATASET_LIMPIO = RUTA_BASE / "winequality-red-limpio.csv"

# Carpetas para almacenar resultados
RUTA_RESULTADOS = RUTA_BASE / "resultados"
RUTA_GRAFICAS = RUTA_RESULTADOS / "graficas"
RUTA_TABLAS = RUTA_RESULTADOS / "tablas"
RUTA_DATOS_PROCESADOS = RUTA_RESULTADOS / "datos_procesados"
RUTA_INFORMES = RUTA_RESULTADOS / "informes"

# Carpeta para almacenar el modelo entrenado, el escalador y la configuración
RUTA_MODELOS = RUTA_BASE / "modelos"

# Archivos de partición y escalado (Fase 2)
RUTA_X_TRAIN = RUTA_DATOS_PROCESADOS / "X_train.csv"
RUTA_X_TEST = RUTA_DATOS_PROCESADOS / "X_test.csv"
RUTA_Y_TRAIN = RUTA_DATOS_PROCESADOS / "y_train.csv"
RUTA_Y_TEST = RUTA_DATOS_PROCESADOS / "y_test.csv"
RUTA_X_TRAIN_ESCALADO = RUTA_DATOS_PROCESADOS / "X_train_escalado.csv"
RUTA_X_TEST_ESCALADO = RUTA_DATOS_PROCESADOS / "X_test_escalado.csv"

# Archivos del modelo entrenado y artefactos (Fase 3)
RUTA_MODELO_JOBLIB = RUTA_MODELOS / "regresion_lineal.joblib"
RUTA_ESCALADOR_JOBLIB = RUTA_MODELOS / "standard_scaler.joblib"
RUTA_CONFIG_MODELO = RUTA_MODELOS / "configuracion_modelo.json"
RUTA_TABLA_COEFICIENTES = RUTA_TABLAS / "coeficientes_regresion.csv"

# Archivos de evaluación e informe (Fase 4)
RUTA_RESUMEN_EVALUACION = RUTA_INFORMES / "resumen_evaluacion.md"
RUTA_TABLA_PREDICCIONES = RUTA_TABLAS / "predicciones_modelo.csv"
RUTA_TABLA_METRICAS = RUTA_TABLAS / "metricas_evaluacion.csv"
RUTA_TABLA_RESIDUOS = RUTA_TABLAS / "estadisticas_residuos.csv"
RUTA_TABLA_TRAIN_TEST = RUTA_TABLAS / "comparacion_train_test.csv"
RUTA_TABLA_MODELO_BASE = RUTA_TABLAS / "comparacion_modelo_base.csv"
RUTA_TABLA_RENDIMIENTO_CALIDAD = RUTA_TABLAS / "rendimiento_por_calidad.csv"

# Archivos de documentación técnica en PDF y Markdown (Fase 5)
RUTA_DOCUMENTACION = RUTA_BASE / "documentacion"
RUTA_REPORTE_PDF = RUTA_DOCUMENTACION / "Reporte_Tecnico_ProyectoIA.pdf"
RUTA_REPORTE_PDF_CORREGIDO = RUTA_DOCUMENTACION / "Reporte_Tecnico_ProyectoIA_Corregido.pdf"
RUTA_REPORTE_MD = RUTA_DOCUMENTACION / "Reporte_Tecnico_ProyectoIA.md"
RUTA_CONFIG_REPORTE = RUTA_DOCUMENTACION / "configuracion_reporte.json"
RUTA_LOGOS_INSTITUCIONALES = RUTA_BASE / "logos_institucionales.png"

# Archivos de experimentación complementaria (validación cruzada con Ridge)
RUTA_TABLA_RIDGE_CV = RUTA_TABLAS / "experimentacion_ridge_cv.csv"
RUTA_GRAFICA_RIDGE_CV = RUTA_GRAFICAS / "21_comparacion_ridge_cv.png"

# Configuración estándar de partición para reproducibilidad
SEMILLA_ALEATORIA = 42
PROPORCION_PRUEBA = 0.20
PROPORCION_ENTRENAMIENTO = 0.80

# Variables independientes (X) y variable dependiente (y) conservando los nombres originales
COLUMNAS_INDEPENDIENTES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
COLUMNA_OBJETIVO = "quality"


def imprimir_encabezado(titulo: str) -> None:
    """Imprime un encabezado principal legible en consola usando caracteres ASCII."""
    linea = "=" * 68
    print(f"\n{linea}")
    print(f" {titulo}")
    print(linea)


def imprimir_subtitulo(titulo: str) -> None:
    """Imprime un separador de sección secundario en consola."""
    linea = "-" * 68
    print(f"\n{titulo}")
    print(linea)


def asegurar_carpetas_resultados() -> None:
    """Crea las carpetas de resultados (gráficas, tablas, datos procesados e informes)."""
    RUTA_GRAFICAS.mkdir(parents=True, exist_ok=True)
    RUTA_TABLAS.mkdir(parents=True, exist_ok=True)
    RUTA_DATOS_PROCESADOS.mkdir(parents=True, exist_ok=True)
    RUTA_INFORMES.mkdir(parents=True, exist_ok=True)


def asegurar_carpeta_modelos() -> None:
    """Crea la carpeta modelos/ si no existe."""
    RUTA_MODELOS.mkdir(parents=True, exist_ok=True)


def asegurar_carpeta_documentacion() -> None:
    """Crea la carpeta documentacion/ si no existe."""
    RUTA_DOCUMENTACION.mkdir(parents=True, exist_ok=True)


def guardar_y_mostrar_figura(
    figura,
    nombre_archivo: str,
    mostrar: bool = True,
) -> Path:
    """Guarda una figura en PNG a 300 DPI, la muestra si aplica y la cierra."""
    import matplotlib
    import matplotlib.pyplot as plt

    asegurar_carpetas_resultados()
    ruta_salida = RUTA_GRAFICAS / nombre_archivo
    figura.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    print(f"Grafica guardada (300 DPI): {ruta_salida}")

    if mostrar and matplotlib.get_backend().lower() != "agg":
        plt.show()

    plt.close(figura)
    return ruta_salida


def existe_dataset_limpio() -> bool:
    """Indica si el archivo winequality-red-limpio.csv ya fue generado."""
    return RUTA_DATASET_LIMPIO.is_file()


def cargar_dataset(ruta_archivo: Path, separador: str = ";") -> pd.DataFrame:
    """Carga un dataset CSV verificando su existencia y manejando errores de lectura."""
    ruta = Path(ruta_archivo)

    if not ruta.is_file():
        raise FileNotFoundError(
            f"No se encontro el archivo de datos: '{ruta}'."
        )

    try:
        datos = pd.read_csv(ruta, sep=separador)
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"El archivo '{ruta.name}' esta vacio o no contiene columnas validas."
        ) from error
    except pd.errors.ParserError as error:
        raise ValueError(
            f"Error al interpretar el formato CSV del archivo '{ruta.name}': {error}"
        ) from error
    except OSError as error:
        raise OSError(
            f"No fue posible leer el archivo '{ruta.name}': {error}"
        ) from error

    if datos.empty:
        raise ValueError(f"El dataset '{ruta.name}' no contiene registros.")

    return datos


def cargar_tabla_procesada(ruta_archivo: Path) -> pd.DataFrame:
    """Carga un CSV procesado (coma como separador) preservando el índice original."""
    ruta = Path(ruta_archivo)

    if not ruta.is_file():
        raise FileNotFoundError(
            f"No se encontro el archivo procesado: '{ruta}'."
        )

    try:
        datos = pd.read_csv(ruta)
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"El archivo '{ruta.name}' esta vacio o no contiene columnas validas."
        ) from error
    except pd.errors.ParserError as error:
        raise ValueError(
            f"Error al interpretar el archivo procesado '{ruta.name}': {error}"
        ) from error
    except OSError as error:
        raise OSError(
            f"No fue posible leer el archivo '{ruta.name}': {error}"
        ) from error

    if datos.empty:
        raise ValueError(f"El archivo '{ruta.name}' no contiene registros.")

    # Restablecer el índice original guardado para verificar ausencia de superposición
    if "indice" in datos.columns:
        datos = datos.set_index("indice")

    return datos


def _archivos_actualizados(destinos: list[Path], origenes: list[Path]) -> bool:
    """Verifica que los archivos destino existan y no sean más antiguos que sus orígenes."""
    if not all(ruta.is_file() for ruta in destinos):
        return False
    if not all(ruta.is_file() for ruta in origenes):
        return False

    marca_destino_mas_antigua = min(ruta.stat().st_mtime for ruta in destinos)
    marca_origen_mas_reciente = max(ruta.stat().st_mtime for ruta in origenes)
    return marca_destino_mas_antigua + 1e-3 >= marca_origen_mas_reciente


def existen_datos_particion() -> bool:
    """Indica si ya existen los cuatro archivos CSV de partición train/test."""
    return all(
        ruta.is_file()
        for ruta in (RUTA_X_TRAIN, RUTA_X_TEST, RUTA_Y_TRAIN, RUTA_Y_TEST)
    )


def datos_particion_actualizados() -> bool:
    """Comprueba que la partición exista y esté actualizada frente al dataset limpio."""
    return _archivos_actualizados(
        [RUTA_X_TRAIN, RUTA_X_TEST, RUTA_Y_TRAIN, RUTA_Y_TEST],
        [RUTA_DATASET_LIMPIO],
    )


def existen_datos_escalados() -> bool:
    """Indica si ya existen los conjuntos escalados y el escalador guardado."""
    return all(
        ruta.is_file()
        for ruta in (
            RUTA_X_TRAIN_ESCALADO,
            RUTA_X_TEST_ESCALADO,
            RUTA_ESCALADOR_JOBLIB,
        )
    )


def datos_escalados_actualizados() -> bool:
    """Comprueba que los datos escalados estén vigentes respecto a la partición actual."""
    return _archivos_actualizados(
        [RUTA_X_TRAIN_ESCALADO, RUTA_X_TEST_ESCALADO, RUTA_ESCALADOR_JOBLIB],
        [RUTA_X_TRAIN, RUTA_X_TEST],
    )


def existe_modelo_entrenado() -> bool:
    """Indica si ya existen el modelo, el escalador y la configuración guardados."""
    return all(
        ruta.is_file()
        for ruta in (
            RUTA_MODELO_JOBLIB,
            RUTA_ESCALADOR_JOBLIB,
            RUTA_CONFIG_MODELO,
        )
    )


def modelo_entrenado_actualizado() -> bool:
    """Comprueba que el modelo entrenado esté vigente respecto a los datos escalados."""
    return _archivos_actualizados(
        [RUTA_MODELO_JOBLIB, RUTA_CONFIG_MODELO],
        [
            RUTA_X_TRAIN_ESCALADO,
            RUTA_X_TEST_ESCALADO,
            RUTA_Y_TRAIN,
            RUTA_Y_TEST,
            RUTA_ESCALADOR_JOBLIB,
        ],
    )


def invalidar_archivos_desactualizados(rutas: list[Path], motivo: str) -> list[str]:
    """Elimina artefactos posteriores desactualizados (nunca archivos de la Fase 1)."""
    eliminados: list[str] = []
    for ruta in rutas:
        if ruta.is_file():
            ruta.unlink()
            eliminados.append(ruta.name)

    if eliminados:
        print(
            f"\nAviso de coherencia ({motivo}): Se invalidaron artefactos "
            f"posteriores desactualizados: {', '.join(eliminados)}."
        )

    return eliminados
