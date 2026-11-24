"""Generador automático del Reporte Técnico Académico en PDF y Markdown (Fase 5).

Construye el documento formal institucional de la Universidad Autónoma de Tamaulipas
(Facultad de Ingeniería Tampico) utilizando ReportLab Platypus a partir de los resultados
reales almacenados en las fases 1 a 4 y la experimentación complementaria (Ridge CV),
sin volver a entrenar el modelo principal ni modificar el dataset ni los artefactos
originales. Genera la versión corregida en documentacion/Reporte_Tecnico_ProyectoIA_Corregido.pdf
conservando la versión original como respaldo.
"""

import html
import importlib.metadata
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import re
import pandas as pd
from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from utilidades import (
    COLUMNA_OBJETIVO,
    RUTA_BASE,
    RUTA_CONFIG_MODELO,
    RUTA_CONFIG_REPORTE,
    RUTA_DATASET_LIMPIO,
    RUTA_GRAFICAS,
    RUTA_LOGOS_INSTITUCIONALES,
    RUTA_REPORTE_MD,
    RUTA_REPORTE_PDF_CORREGIDO,
    RUTA_RESUMEN_EVALUACION,
    RUTA_TABLAS,
    RUTA_TABLA_RIDGE_CV,
    RUTA_X_TEST,
    RUTA_X_TRAIN,
    RUTA_Y_TEST,
    RUTA_Y_TRAIN,
    asegurar_carpeta_documentacion,
    cargar_dataset,
    cargar_tabla_procesada,
    imprimir_encabezado,
)

# Paleta sobria académica
COLOR_PRIMARIO = colors.HexColor("#1A2B49")
COLOR_SECUNDARIO = colors.HexColor("#243B61")
COLOR_TEXTO = colors.HexColor("#1F2937")
COLOR_GRIS_TEXTO = colors.HexColor("#4B5563")
COLOR_FONDO_SUAVE = colors.HexColor("#F7F9FC")
COLOR_FONDO_CODIGO = colors.HexColor("#F4F6F8")
COLOR_BORDE = colors.HexColor("#CBD5E0")

# Listas exactas de insumos requeridos (Fases 1 a 4 + experimentación complementaria)
TABLAS_REQUERIDAS = [
    "comparacion_limpieza.csv",
    "estadisticas_descriptivas.csv",
    "distribucion_calidad.csv",
    "matriz_correlacion.csv",
    "correlaciones_con_quality.csv",
    "valores_atipicos_iqr.csv",
    "resumen_particion.csv",
    "comparacion_escalado.csv",
    "parametros_escalador.csv",
    "coeficientes_regresion.csv",
    "predicciones_modelo.csv",
    "metricas_evaluacion.csv",
    "estadisticas_residuos.csv",
    "comparacion_train_test.csv",
    "comparacion_modelo_base.csv",
    "rendimiento_por_calidad.csv",
    "experimentacion_ridge_cv.csv",
]

GRAFICAS_REQUERIDAS = [
    "01_distribucion_calidad.png",
    "02_matriz_correlacion.png",
    "03_correlacion_con_quality.png",
    "04_boxplots_variables.png",
    "05_histogramas_variables.png",
    "06_dispersion_calidad.png",
    "07_particion_datos.png",
    "08_distribucion_train_test.png",
    "09_variables_sin_escalar.png",
    "10_variables_escaladas.png",
    "11_comparacion_escalado.png",
    "12_coeficientes_regresion.png",
    "13_valores_reales_vs_predichos.png",
    "14_comparacion_predicciones.png",
    "15_grafica_residuos.png",
    "16_distribucion_residuos.png",
    "17_boxplot_residuos.png",
    "18_comparacion_train_test.png",
    "19_comparacion_modelo_base.png",
    "20_error_por_calidad.png",
    "21_comparacion_ridge_cv.png",
]

CONFIG_REPORTE_POR_DEFECTO = {
    "universidad": "Universidad Autónoma de Tamaulipas (UAT)",
    "facultad": "Facultad de Ingeniería Tampico (FIT)",
    "carrera": "Ingeniería en Sistemas Computacionales",
    "materia": "INTELIGENCIA ARTIFICIAL",
    "tipo_trabajo": "PROYECTO FINAL",
    "titulo": (
        "Desarrollo y evaluación de un modelo de aprendizaje supervisado para la "
        "predicción de la calidad del vino tinto mediante regresión lineal múltiple"
    ),
    "integrantes": [
        "Francisco Mogollón José Antonio",
        "Gómez Sánchez Haziel Emiliano",
        "Hernández González Brian Satihs",
    ],
    "profesor": "Martínez Ponce Mario Alberto",
    "grupo": "",
    "lugar": "Tampico, Tamaulipas, México",
    "fecha": "23 de noviembre de 2026",
    "logo_institucional": "logos_institucionales.png",
}

# Claves bibliográficas APA 7 ordenadas alfabéticamente y verificadas
REFERENCIAS_APA7 = [
    {
        "clave": "Cortez et al., 2009a",
        "patron": r"Cortez et al\.,?\s*\(?2009a",
        "texto": (
            "Cortez, P., Cerdeira, A., Almeida, F., Matos, T., &amp; Reis, J. (2009a). "
            "Modeling wine preferences by data mining from physicochemical properties. "
            "<i>Decision Support Systems</i>, <i>47</i>(4), 547–553. "
            "https://doi.org/10.1016/j.dss.2009.05.016"
        ),
        "md": (
            "Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009a). "
            "Modeling wine preferences by data mining from physicochemical properties. "
            "*Decision Support Systems*, *47*(4), 547–553. "
            "https://doi.org/10.1016/j.dss.2009.05.016"
        ),
    },
    {
        "clave": "Cortez et al., 2009b",
        "patron": r"Cortez et al\.,?\s*\(?2009b",
        "texto": (
            "Cortez, P., Cerdeira, A., Almeida, F., Matos, T., &amp; Reis, J. (2009b). "
            "<i>Wine Quality</i> [Conjunto de datos]. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C56S3T"
        ),
        "md": (
            "Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009b). "
            "*Wine Quality* [Conjunto de datos]. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C56S3T"
        ),
    },
    {
        "clave": "Harris et al., 2020",
        "patron": r"Harris et al\.,?\s*\(?2020",
        "texto": (
            "Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., "
            "Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., "
            "Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., "
            "Brett, M., Haldane, A., Fernández del Río, J., Wiebe, M., Peterson, P., ... "
            "Oliphant, T. E. (2020). Array programming with NumPy. <i>Nature</i>, "
            "<i>585</i>(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2"
        ),
        "md": (
            "Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., "
            "Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., "
            "Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., "
            "Brett, M., Haldane, A., Fernández del Río, J., Wiebe, M., Peterson, P., ... "
            "Oliphant, T. E. (2020). Array programming with NumPy. *Nature*, "
            "*585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2"
        ),
    },
    {
        "clave": "Hunter, 2007",
        "patron": r"Hunter,?\s*\(?2007",
        "texto": (
            "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. "
            "<i>Computing in Science &amp; Engineering</i>, <i>9</i>(3), 90–95. "
            "https://doi.org/10.1109/MCSE.2007.55"
        ),
        "md": (
            "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. "
            "*Computing in Science & Engineering*, *9*(3), 90–95. "
            "https://doi.org/10.1109/MCSE.2007.55"
        ),
    },
    {
        "clave": "James et al., 2023",
        "patron": r"James et al\.,?\s*\(?2023",
        "texto": (
            "James, G., Witten, D., Hastie, T., Tibshirani, R., &amp; Taylor, J. (2023). "
            "<i>An introduction to statistical learning: With applications in Python</i>. "
            "Springer. https://doi.org/10.1007/978-3-031-38747-0"
        ),
        "md": (
            "James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). "
            "*An introduction to statistical learning: With applications in Python*. "
            "Springer. https://doi.org/10.1007/978-3-031-38747-0"
        ),
    },
    {
        "clave": "McKinney, 2010",
        "patron": r"McKinney,?\s*\(?2010",
        "texto": (
            "McKinney, W. (2010). Data structures for statistical computing in Python. "
            "En S. van der Walt &amp; J. Millman (Eds.), <i>Proceedings of the 9th Python "
            "in Science Conference</i> (pp. 56–61). "
            "https://doi.org/10.25080/Majora-92bf1922-00a"
        ),
        "md": (
            "McKinney, W. (2010). Data structures for statistical computing in Python. "
            "En S. van der Walt & J. Millman (Eds.), *Proceedings of the 9th Python "
            "in Science Conference* (pp. 56–61). "
            "https://doi.org/10.25080/Majora-92bf1922-00a"
        ),
    },
    {
        "clave": "pandas development team, 2026",
        "patron": r"pandas development team,?\s*\(?2026",
        "texto": (
            "pandas development team. (2026). <i>pandas documentation</i> "
            "(Versión 3.1.1) [Manual técnico]. https://pandas.pydata.org/docs/"
        ),
        "md": (
            "pandas development team. (2026). *pandas documentation* "
            "(Versión 3.1.1) [Manual técnico]. https://pandas.pydata.org/docs/"
        ),
    },
    {
        "clave": "Pedregosa et al., 2011",
        "patron": r"Pedregosa et al\.,?\s*\(?2011",
        "texto": (
            "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., "
            "Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., "
            "Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., "
            "&amp; Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. "
            "<i>Journal of Machine Learning Research</i>, <i>12</i>(85), 2825–2830. "
            "https://jmlr.org/papers/v12/pedregosa11a.html"
        ),
        "md": (
            "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., "
            "Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., "
            "Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., "
            "& Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. "
            "*Journal of Machine Learning Research*, *12*(85), 2825–2830. "
            "https://jmlr.org/papers/v12/pedregosa11a.html"
        ),
    },
    {
        "clave": "Russell & Norvig, 2021",
        "patron": r"Russell\s*(?:&amp;|&|y)\s*Norvig,?\s*\(?2021",
        "texto": (
            "Russell, S., &amp; Norvig, P. (2021). <i>Artificial intelligence: "
            "A modern approach</i> (4.ª ed.). Pearson."
        ),
        "md": (
            "Russell, S., & Norvig, P. (2021). *Artificial intelligence: "
            "A modern approach* (4.ª ed.). Pearson."
        ),
    },
    {
        "clave": "scikit-learn developers, 2026",
        "patron": r"scikit-learn developers,?\s*\(?2026",
        "texto": (
            "scikit-learn developers. (2026). <i>scikit-learn user guide</i> "
            "(Versión 1.9.1) [Manual técnico]. "
            "https://scikit-learn.org/stable/user_guide.html"
        ),
        "md": (
            "scikit-learn developers. (2026). *scikit-learn user guide* "
            "(Versión 1.9.1) [Manual técnico]. "
            "https://scikit-learn.org/stable/user_guide.html"
        ),
    },
    {
        "clave": "Waskom, 2021",
        "patron": r"Waskom,?\s*\(?2021",
        "texto": (
            "Waskom, M. L. (2021). seaborn: Statistical data visualization. "
            "<i>Journal of Open Source Software</i>, <i>6</i>(60), 3021. "
            "https://doi.org/10.21105/joss.03021"
        ),
        "md": (
            "Waskom, M. L. (2021). seaborn: Statistical data visualization. "
            "*Journal of Open Source Software*, *6*(60), 3021. "
            "https://doi.org/10.21105/joss.03021"
        ),
    },
]


def _cargar_o_crear_config_reporte(ruta_config: Path = RUTA_CONFIG_REPORTE) -> dict:
    """Carga configuracion_reporte.json o la crea con los datos institucionales base."""
    asegurar_carpeta_documentacion()
    if not ruta_config.is_file():
        with open(ruta_config, "w", encoding="utf-8") as archivo:
            json.dump(CONFIG_REPORTE_POR_DEFECTO, archivo, indent=4, ensure_ascii=False)
        return dict(CONFIG_REPORTE_POR_DEFECTO)

    with open(ruta_config, "r", encoding="utf-8") as archivo:
        config = json.load(archivo)
    return config


def verificar_insumos_reporte(
    ruta_config_modelo: Path = RUTA_CONFIG_MODELO,
    ruta_tablas: Path = RUTA_TABLAS,
    ruta_graficas: Path = RUTA_GRAFICAS,
    ruta_resumen: Path = RUTA_RESUMEN_EVALUACION,
    ruta_requirements: Path = RUTA_BASE / "requirements.txt",
    ruta_dataset_limpio: Path = RUTA_DATASET_LIMPIO,
    rutas_particion: tuple[Path, ...] = (
        RUTA_X_TRAIN,
        RUTA_X_TEST,
        RUTA_Y_TRAIN,
        RUTA_Y_TEST,
    ),
) -> None:
    """Comprueba que existan todas las tablas, gráficas, particiones y archivos necesarios."""
    faltantes: list[str] = []

    if not ruta_dataset_limpio.is_file():
        faltantes.append(f"{ruta_dataset_limpio} (Fase 1: Limpieza de datos)")

    for ruta_part in rutas_particion:
        if not ruta_part.is_file():
            faltantes.append(f"{ruta_part} (Fase 2: Partición de los datos)")

    if not ruta_config_modelo.is_file():
        faltantes.append(f"{ruta_config_modelo} (Fase 3: Entrenamiento del modelo)")

    for nombre_tabla in TABLAS_REQUERIDAS:
        ruta_archivo = ruta_tablas / nombre_tabla
        if not ruta_archivo.is_file():
            fase_origen = (
                "Experimentación complementaria (Ridge CV)"
                if nombre_tabla == RUTA_TABLA_RIDGE_CV.name
                else "Tablas de fases 1 a 4"
            )
            faltantes.append(f"{ruta_archivo} ({fase_origen})")

    for nombre_grafica in GRAFICAS_REQUERIDAS:
        ruta_archivo = ruta_graficas / nombre_grafica
        if not ruta_archivo.is_file():
            fase_origen = (
                "Experimentación complementaria (Ridge CV)"
                if nombre_grafica == "21_comparacion_ridge_cv.png"
                else "Gráficas de fases 1 a 4"
            )
            faltantes.append(f"{ruta_archivo} ({fase_origen})")

    if not ruta_resumen.is_file():
        faltantes.append(f"{ruta_resumen} (Fase 4: Evaluación del rendimiento)")

    if not ruta_requirements.is_file():
        faltantes.append(f"{ruta_requirements} (Dependencias del proyecto)")

    if faltantes:
        listado = "\n  - ".join(faltantes)
        raise FileNotFoundError(
            "No es posible generar el reporte técnico PDF porque faltan insumos "
            f"necesarios:\n  - {listado}\n"
            "Ejecute primero las fases anteriores (opciones 1 a 20 o 22) antes de "
            "generar la documentación."
        )


def _leer_versiones_requirements(ruta_requirements: Path) -> dict[str, str]:
    """Lee requirements.txt y devuelve un diccionario {paquete: version}."""
    versiones: dict[str, str] = {}
    with open(ruta_requirements, "r", encoding="utf-8") as archivo:
        for linea in archivo:
            limpia = linea.strip()
            if not limpia or limpia.startswith("#"):
                continue
            if "==" in limpia:
                nombre, version = limpia.split("==", maxsplit=1)
                versiones[nombre.strip().lower()] = version.strip()
    return versiones


def _obtener_versiones_entorno(versiones_req: dict[str, str]) -> dict[str, str]:
    """Consulta las versiones realmente instaladas en el entorno activo."""
    versiones_reales = {"python": platform.python_version()}
    paquetes = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "joblib",
        "reportlab",
        "pillow",
    ]
    for paquete in paquetes:
        try:
            instalada = importlib.metadata.version(paquete)
        except importlib.metadata.PackageNotFoundError:
            instalada = versiones_req.get(paquete, "No instalada")
        versiones_reales[paquete] = instalada

    discrepancias = []
    for paquete in paquetes:
        req = versiones_req.get(paquete)
        inst = versiones_reales.get(paquete)
        if req and inst and req != inst:
            discrepancias.append(f"{paquete} (requirements={req}, instalado={inst})")
    if discrepancias:
        raise ValueError(
            "Inconsistencia entre requirements.txt y las versiones instaladas: "
            + ", ".join(discrepancias)
        )
    return versiones_reales


def _verificar_vectores_x_compartidos(
    df_limpio: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict[str, int]:
    """Comprueba si existen vectores de características X idénticos entre train y test."""
    cols_x = list(X_train.columns)
    duplicados_x_limpio = int(df_limpio.duplicated(subset=cols_x, keep=False).sum())
    grupos_distinta_calidad = int(
        (df_limpio.groupby(cols_x)[COLUMNA_OBJETIVO].nunique() > 1).sum()
    )

    claves_train = set(X_train.apply(tuple, axis=1))
    claves_test = set(X_test.apply(tuple, axis=1))
    compartidos = claves_train.intersection(claves_test)

    return {
        "duplicados_x_limpio": duplicados_x_limpio,
        "grupos_distinta_calidad": grupos_distinta_calidad,
        "compartidos_train_test": len(compartidos),
        "indices_compartidos": len(set(X_train.index).intersection(set(X_test.index))),
    }


def _calcular_atipicos_iqr_extremos(
    df_limpio: pd.DataFrame,
    df_iqr: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Calcula para cada variable cuántos atípicos caen debajo o encima de los límites IQR."""
    detalle: dict[str, dict[str, float]] = {}
    for _, fila in df_iqr.iterrows():
        var = str(fila["Variable"])
        lim_inf = float(fila["Límite inferior"])
        lim_sup = float(fila["Límite superior"])
        serie = df_limpio[var]
        detalle[var] = {
            "inferior": int((serie < lim_inf).sum()),
            "superior": int((serie > lim_sup).sum()),
            "total": int(fila["Valores atípicos"]),
            "porcentaje": float(fila["Porcentaje (%)"]),
        }
    return detalle


def cargar_insumos_proyecto() -> dict:
    """Carga dinámicamente todas las métricas, tablas y configuraciones del proyecto."""
    config_reporte = _cargar_o_crear_config_reporte()
    with open(RUTA_CONFIG_MODELO, "r", encoding="utf-8") as archivo:
        config_modelo = json.load(archivo)

    versiones_req = _leer_versiones_requirements(RUTA_BASE / "requirements.txt")
    versiones = _obtener_versiones_entorno(versiones_req)

    with open(RUTA_RESUMEN_EVALUACION, "r", encoding="utf-8") as archivo:
        resumen_md = archivo.read()

    tablas: dict[str, pd.DataFrame] = {}
    for nombre_tabla in TABLAS_REQUERIDAS:
        clave = nombre_tabla.replace(".csv", "")
        if nombre_tabla == RUTA_TABLA_RIDGE_CV.name:
            tablas[clave] = pd.read_csv(
                RUTA_TABLAS / nombre_tabla, dtype={"Alpha": str}
            )
        else:
            tablas[clave] = pd.read_csv(RUTA_TABLAS / nombre_tabla)

    df_limpio = cargar_dataset(RUTA_DATASET_LIMPIO, separador=";")
    X_train = cargar_tabla_procesada(RUTA_X_TRAIN)
    X_test = cargar_tabla_procesada(RUTA_X_TEST)
    revision_x = _verificar_vectores_x_compartidos(df_limpio, X_train, X_test)
    iqr_extremos = _calcular_atipicos_iqr_extremos(
        df_limpio, tablas["valores_atipicos_iqr"]
    )

    # Construir la ecuación real del modelo distinguiendo z(variable) estandarizada
    coef_por_var = dict(
        zip(
            tablas["coeficientes_regresion"]["Variable"],
            tablas["coeficientes_regresion"]["Coeficiente"],
        )
    )
    orden_vars = config_modelo["orden_variables_independientes"]
    intercepto = float(config_modelo["intercepto"])
    terminos = [f"{intercepto:.4f}"]
    for var in orden_vars:
        coef = float(coef_por_var[var])
        signo = "+" if coef >= 0 else "−"
        terminos.append(f"{signo} {abs(coef):.4f}·z({var})")
    ecuacion_real = "ŷ = " + " ".join(terminos)

    metricas_map = dict(
        zip(
            tablas["metricas_evaluacion"]["Métrica"],
            tablas["metricas_evaluacion"]["Valor"],
        )
    )
    tt_map = tablas["comparacion_train_test"].set_index("Métrica")
    base_map = tablas["comparacion_modelo_base"].set_index("Métrica")
    res_map = dict(
        zip(
            tablas["estadisticas_residuos"]["Estadística"],
            tablas["estadisticas_residuos"]["Valor"],
        )
    )

    ruta_logo_cfg = RUTA_BASE / config_reporte.get(
        "logo_institucional", "logos_institucionales.png"
    )
    ruta_logo = (
        ruta_logo_cfg
        if ruta_logo_cfg.is_file()
        else (
            RUTA_LOGOS_INSTITUCIONALES
            if RUTA_LOGOS_INSTITUCIONALES.is_file()
            else None
        )
    )

    return {
        "config_reporte": config_reporte,
        "config_modelo": config_modelo,
        "versiones": versiones,
        "resumen_eval_md": resumen_md,
        "tablas": tablas,
        "revision_x": revision_x,
        "iqr_extremos": iqr_extremos,
        "ecuacion_real": ecuacion_real,
        "metricas": {k: float(v) for k, v in metricas_map.items()},
        "train_test": tt_map,
        "modelo_base": base_map,
        "residuos_stats": {k: float(v) for k, v in res_map.items()},
        "ruta_logo": ruta_logo,
    }


def registrar_fuentes_academicas() -> tuple[str, str]:
    """Registra Times New Roman y Courier New usando WINDIR; con fallback estándar."""
    ruta_fuentes = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    times_archivos = {
        "normal": ruta_fuentes / "times.ttf",
        "bold": ruta_fuentes / "timesbd.ttf",
        "italic": ruta_fuentes / "timesi.ttf",
        "boldItalic": ruta_fuentes / "timesbi.ttf",
    }
    if all(p.is_file() for p in times_archivos.values()):
        pdfmetrics.registerFont(TTFont("TimesNewRoman", str(times_archivos["normal"])))
        pdfmetrics.registerFont(
            TTFont("TimesNewRoman-Bold", str(times_archivos["bold"]))
        )
        pdfmetrics.registerFont(
            TTFont("TimesNewRoman-Italic", str(times_archivos["italic"]))
        )
        pdfmetrics.registerFont(
            TTFont("TimesNewRoman-BoldItalic", str(times_archivos["boldItalic"]))
        )
        pdfmetrics.registerFontFamily(
            "TimesNewRoman",
            normal="TimesNewRoman",
            bold="TimesNewRoman-Bold",
            italic="TimesNewRoman-Italic",
            boldItalic="TimesNewRoman-BoldItalic",
        )
        fuente_serif = "TimesNewRoman"
    else:
        fuente_serif = "Times-Roman"

    cour_archivos = {
        "normal": ruta_fuentes / "cour.ttf",
        "bold": ruta_fuentes / "courbd.ttf",
        "italic": ruta_fuentes / "couri.ttf",
        "boldItalic": ruta_fuentes / "courbi.ttf",
    }
    if all(p.is_file() for p in cour_archivos.values()):
        pdfmetrics.registerFont(TTFont("CourierNew", str(cour_archivos["normal"])))
        pdfmetrics.registerFont(TTFont("CourierNew-Bold", str(cour_archivos["bold"])))
        pdfmetrics.registerFont(
            TTFont("CourierNew-Italic", str(cour_archivos["italic"]))
        )
        pdfmetrics.registerFont(
            TTFont("CourierNew-BoldItalic", str(cour_archivos["boldItalic"]))
        )
        pdfmetrics.registerFontFamily(
            "CourierNew",
            normal="CourierNew",
            bold="CourierNew-Bold",
            italic="CourierNew-Italic",
            boldItalic="CourierNew-BoldItalic",
        )
        fuente_mono = "CourierNew"
    else:
        fuente_mono = "Courier"

    return fuente_serif, fuente_mono


def _cargar_imagen_reportlab(
    ruta_imagen: Path,
    ancho_max_cm: float,
    alto_max_cm: float,
) -> Image:
    """Carga una imagen conservando su proporción exacta y evitando artefactos de paleta."""
    with PILImage.open(ruta_imagen) as img:
        ancho_px, alto_px = img.size
        relacion = ancho_px / alto_px
        ancho_pt = ancho_max_cm * cm
        alto_pt = ancho_pt / relacion
        if alto_pt > alto_max_cm * cm:
            alto_pt = alto_max_cm * cm
            ancho_pt = alto_pt * relacion

        # Si es PNG en modo paleta/alfa, componer en blanco en memoria sin alterar el disco
        if img.mode in ("P", "PA", "RGBA", "LA"):
            rgba = img.convert("RGBA")
            fondo = PILImage.new("RGB", rgba.size, (255, 255, 255))
            fondo.paste(rgba, mask=rgba.split()[3])
            buffer = BytesIO()
            fondo.save(buffer, format="PNG")
            buffer.seek(0)
            return Image(buffer, width=ancho_pt, height=alto_pt)

    return Image(str(ruta_imagen), width=ancho_pt, height=alto_pt)


class ReporteDocTemplate(BaseDocTemplate):
    """Plantilla Platypus con portada limpia, páginas verticales/horizontales e índices reales."""

    def __init__(
        self,
        filename: str,
        fuente_serif: str,
        config_reporte: dict,
        **kwargs,
    ) -> None:
        super().__init__(filename, **kwargs)
        self.fuente_serif = fuente_serif
        self.config_reporte = config_reporte
        self.paginas_secciones: dict[str, int] = {}
        self.paginas_tablas: dict[str, int] = {}
        self.paginas_figuras: dict[str, int] = {}
        self.conteo_flowables_pagina: dict[int, int] = {}
        self._notificados: set[tuple[str, str]] = set()
        self.total_paginas = 0

        margen = 2.5 * cm
        ancho_util = self.pagesize[0] - 2 * margen
        alto_util = self.pagesize[1] - 2 * margen

        frame_portada = Frame(
            margen,
            margen,
            ancho_util,
            alto_util,
            id="frame_portada",
            topPadding=0,
            bottomPadding=0,
            leftPadding=0,
            rightPadding=0,
        )
        frame_contenido = Frame(
            margen,
            margen,
            ancho_util,
            alto_util,
            id="frame_contenido",
            topPadding=6,
            bottomPadding=6,
            leftPadding=0,
            rightPadding=0,
        )

        # Página horizontal (Carta landscape) para Anexo C
        tam_horiz = landscape(letter)
        margen_h = 2.0 * cm
        margen_v = 2.4 * cm
        frame_horizontal = Frame(
            margen_h,
            margen_v,
            tam_horiz[0] - 2 * margen_h,
            tam_horiz[1] - 2 * margen_v,
            id="frame_horizontal",
            topPadding=4,
            bottomPadding=4,
            leftPadding=0,
            rightPadding=0,
        )

        self.addPageTemplates(
            [
                PageTemplate(id="Portada", frames=[frame_portada], pagesize=letter),
                PageTemplate(
                    id="Contenido",
                    frames=[frame_contenido],
                    pagesize=letter,
                    onPage=self._dibujar_encabezado_y_pie,
                ),
                PageTemplate(
                    id="ContenidoHorizontal",
                    frames=[frame_horizontal],
                    pagesize=tam_horiz,
                    onPage=self._dibujar_encabezado_y_pie,
                ),
            ]
        )

    def handle_documentBegin(self) -> None:
        super().handle_documentBegin()
        self.paginas_secciones = {}
        self.paginas_tablas = {}
        self.paginas_figuras = {}
        self.conteo_flowables_pagina = {}
        self._notificados = set()
        self.total_paginas = 0

    def handle_documentEnd(self) -> None:
        self.total_paginas = max(
            self.total_paginas,
            self.page,
            max(self.conteo_flowables_pagina.keys(), default=0),
        )
        super().handle_documentEnd()

    def _dibujar_encabezado_y_pie(self, canvas, doc) -> None:
        """Dibuja encabezado y pie discretos adaptándose a orientación vertical u horizontal."""
        self.total_paginas = max(self.total_paginas, doc.page)
        canvas.saveState()
        ancho_pagina, alto_pagina = canvas._pagesize
        margen_x = 2.0 * cm if ancho_pagina > alto_pagina else 2.5 * cm

        fuente_normal = self.fuente_serif
        fuente_italica = (
            f"{self.fuente_serif}-Italic"
            if self.fuente_serif == "TimesNewRoman"
            else "Times-Italic"
        )

        y_header = alto_pagina - 1.60 * cm
        canvas.setFont(fuente_italica, 8.5)
        canvas.setFillColor(COLOR_GRIS_TEXTO)
        canvas.drawString(
            margen_x, y_header, "UAT — Facultad de Ingeniería Tampico (FIT)"
        )
        canvas.drawRightString(
            ancho_pagina - margen_x,
            y_header,
            "Reporte Técnico — Calidad del Vino Tinto",
        )
        canvas.setStrokeColor(COLOR_BORDE)
        canvas.setLineWidth(0.5)
        canvas.line(margen_x, y_header - 4, ancho_pagina - margen_x, y_header - 4)

        y_footer = 1.55 * cm
        canvas.line(margen_x, y_footer + 11, ancho_pagina - margen_x, y_footer + 11)
        canvas.setFont(fuente_normal, 9)
        canvas.setFillColor(COLOR_GRIS_TEXTO)
        canvas.drawString(
            margen_x,
            y_footer,
            f"{self.config_reporte['carrera']}  |  {self.config_reporte['materia'].title()}",
        )
        canvas.drawRightString(
            ancho_pagina - margen_x, y_footer, f"Página {doc.page}"
        )

        canvas.restoreState()

    def _registrar_elemento(self, elemento) -> None:
        """Notifica al TOC, índice de tablas o índice de figuras evitando duplicados."""
        info_toc = getattr(elemento, "_toc_info", None)
        if info_toc is not None:
            nivel, texto, clave = info_toc
            if ("TOC", clave) not in self._notificados:
                self._notificados.add(("TOC", clave))
                self.canv.bookmarkPage(clave)
                self.notify("TOCEntry", (nivel, texto, self.page, clave))
                self.paginas_secciones[texto] = self.page

        info_tot = getattr(elemento, "_tot_info", None)
        if info_tot is not None:
            nivel, texto, clave = info_tot
            if ("TOT", clave) not in self._notificados:
                self._notificados.add(("TOT", clave))
                self.canv.bookmarkPage(clave)
                self.notify("TOTEntry", (nivel, texto, self.page, clave))
                self.paginas_tablas[texto] = self.page

        info_tof = getattr(elemento, "_tof_info", None)
        if info_tof is not None:
            nivel, texto, clave = info_tof
            if ("TOF", clave) not in self._notificados:
                self._notificados.add(("TOF", clave))
                self.canv.bookmarkPage(clave)
                self.notify("TOFEntry", (nivel, texto, self.page, clave))
                self.paginas_figuras[texto] = self.page

    def afterFlowable(self, flowable) -> None:
        self.conteo_flowables_pagina[self.page] = (
            self.conteo_flowables_pagina.get(self.page, 0) + 1
        )
        self._registrar_elemento(flowable)
        for hijo in getattr(flowable, "_content", []):
            self._registrar_elemento(hijo)


def crear_estilos(fuente_serif: str, fuente_mono: str) -> dict[str, ParagraphStyle]:
    """Define todos los estilos tipográficos académicos con control de viudas/huérfanas."""
    base = getSampleStyleSheet()
    fuente_bold = (
        f"{fuente_serif}-Bold" if fuente_serif == "TimesNewRoman" else "Times-Bold"
    )
    fuente_italica = (
        f"{fuente_serif}-Italic" if fuente_serif == "TimesNewRoman" else "Times-Italic"
    )

    estilos = {
        "Cuerpo": ParagraphStyle(
            "Cuerpo",
            parent=base["Normal"],
            fontName=fuente_serif,
            fontSize=12,
            leading=18,  # Interlineado 1.5
            alignment=TA_JUSTIFY,
            textColor=COLOR_TEXTO,
            spaceAfter=6,
            allowWidows=0,
            allowOrphans=0,
        ),
        "CuerpoConFormula": ParagraphStyle(
            "CuerpoConFormula",
            parent=base["Normal"],
            fontName=fuente_serif,
            fontSize=12,
            leading=18,
            alignment=TA_JUSTIFY,
            textColor=COLOR_TEXTO,
            spaceAfter=4,
            allowWidows=0,
            allowOrphans=0,
            keepWithNext=True,
        ),
        "Lista": ParagraphStyle(
            "Lista",
            parent=base["Normal"],
            fontName=fuente_serif,
            fontSize=12,
            leading=18,
            alignment=TA_JUSTIFY,
            textColor=COLOR_TEXTO,
            leftIndent=16,
            firstLineIndent=-10,
            spaceAfter=4,
            allowWidows=0,
            allowOrphans=0,
        ),
        "TituloPrincipal": ParagraphStyle(
            "TituloPrincipal",
            parent=base["Heading1"],
            fontName=fuente_bold,
            fontSize=16,
            leading=21,
            alignment=TA_LEFT,
            textColor=COLOR_PRIMARIO,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "Subtitulo": ParagraphStyle(
            "Subtitulo",
            parent=base["Heading2"],
            fontName=fuente_bold,
            fontSize=14,
            leading=18,
            alignment=TA_LEFT,
            textColor=COLOR_SECUNDARIO,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "SubtituloMenor": ParagraphStyle(
            "SubtituloMenor",
            parent=base["Heading3"],
            fontName=fuente_bold,
            fontSize=12.5,
            leading=16,
            alignment=TA_LEFT,
            textColor=COLOR_TEXTO,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "Formula": ParagraphStyle(
            "Formula",
            parent=base["Normal"],
            fontName=fuente_serif,
            fontSize=11.5,
            leading=15.5,
            alignment=TA_CENTER,
            textColor=COLOR_PRIMARIO,
        ),
        "Codigo": ParagraphStyle(
            "Codigo",
            parent=base["Code"],
            fontName=fuente_mono,
            fontSize=8.5,
            leading=11.5,
            alignment=TA_LEFT,
            textColor=COLOR_TEXTO,
        ),
        "TituloTabla": ParagraphStyle(
            "TituloTabla",
            parent=base["Normal"],
            fontName=fuente_bold,
            fontSize=10.5,
            leading=13.5,
            alignment=TA_LEFT,
            textColor=COLOR_PRIMARIO,
            spaceBefore=8,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "NotaTabla": ParagraphStyle(
            "NotaTabla",
            parent=base["Normal"],
            fontName=fuente_italica,
            fontSize=9.5,
            leading=12.5,
            alignment=TA_LEFT,
            textColor=COLOR_GRIS_TEXTO,
            spaceBefore=3,
            spaceAfter=8,
            allowWidows=0,
            allowOrphans=0,
        ),
        "CeldaEncabezado": ParagraphStyle(
            "CeldaEncabezado",
            parent=base["Normal"],
            fontName=fuente_bold,
            fontSize=9,
            leading=11.5,
            alignment=TA_CENTER,
            textColor=colors.white,
        ),
        "CeldaTexto": ParagraphStyle(
            "CeldaTexto",
            parent=base["Normal"],
            fontName=fuente_serif,
            fontSize=9,
            leading=12,
            alignment=TA_LEFT,
            textColor=COLOR_TEXTO,
        ),
        "CeldaNumero": ParagraphStyle(
            "CeldaNumero",
            parent=base["Normal"],
            fontName=fuente_serif,
            fontSize=9,
            leading=12,
            alignment=TA_RIGHT,
            textColor=COLOR_TEXTO,
        ),
        "PieFigura": ParagraphStyle(
            "PieFigura",
            parent=base["Normal"],
            fontName=fuente_bold,
            fontSize=10.5,
            leading=13.5,
            alignment=TA_CENTER,
            textColor=COLOR_PRIMARIO,
            spaceBefore=3,
            spaceAfter=1,
        ),
        "FuenteFigura": ParagraphStyle(
            "FuenteFigura",
            parent=base["Normal"],
            fontName=fuente_italica,
            fontSize=9.5,
            leading=12.5,
            alignment=TA_CENTER,
            textColor=COLOR_GRIS_TEXTO,
            spaceAfter=8,
        ),
        "ReferenciaAPA": ParagraphStyle(
            "ReferenciaAPA",
            parent=base["Normal"],
            fontName=fuente_serif,
            fontSize=12,
            leading=17,
            alignment=TA_LEFT,  # Alineación izquierda para evitar huecos excesivos en URLs/DOIs
            textColor=COLOR_TEXTO,
            leftIndent=1.27 * cm,
            firstLineIndent=-1.27 * cm,
            spaceAfter=6,
            allowWidows=0,
            allowOrphans=0,
        ),
        "TOCNivel0": ParagraphStyle(
            "TOCNivel0",
            parent=base["Normal"],
            fontName=fuente_bold,
            fontSize=11,
            leading=14.5,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=3,
            textColor=COLOR_PRIMARIO,
        ),
        "TOCNivel1": ParagraphStyle(
            "TOCNivel1",
            parent=base["Normal"],
            fontName=fuente_serif,
            fontSize=10.5,
            leading=13.5,
            leftIndent=16,
            firstLineIndent=0,
            textColor=COLOR_TEXTO,
        ),
        "TOCElemento": ParagraphStyle(
            "TOCElemento",
            parent=base["Normal"],
            fontName=fuente_serif,
            fontSize=10.2,
            leading=13.0,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=1.0,
            textColor=COLOR_TEXTO,
        ),
    }
    return estilos


def _crear_heading_toc(
    texto: str,
    nivel: int,
    clave: str,
    estilo: ParagraphStyle,
) -> Paragraph:
    """Crea un encabezado Paragraph con metadatos para el índice general."""
    p = Paragraph(texto, estilo)
    p._toc_info = (nivel, texto, clave)
    return p


def _crear_formula(texto_html: str, estilos: dict[str, ParagraphStyle]) -> Table:
    """Envuelve una fórmula matemática en un recuadro discreto y centrado."""
    celda = Paragraph(texto_html, estilos["Formula"])
    tabla = Table([[celda]], colWidths=[15.8 * cm])
    tabla.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), COLOR_FONDO_SUAVE),
                ("BOX", (0, 0), (-1, -1), 0.6, COLOR_BORDE),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    return tabla


def _crear_bloque_codigo(codigo: str, estilos: dict[str, ParagraphStyle]) -> Table:
    """Renderiza un fragmento corto de código Python con fondo gris claro."""
    lineas_html = (
        codigo.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
        .replace(" ", "&nbsp;")
    )
    celda = Paragraph(lineas_html, estilos["Codigo"])
    tabla = Table([[celda]], colWidths=[16.2 * cm])
    tabla.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), COLOR_FONDO_CODIGO),
                ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDE),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return tabla


def _crear_tabla_academica(
    numero_tabla: int,
    titulo: str,
    encabezados: list[str],
    filas: list[list[str]],
    anchos_cm: list[float],
    alineaciones: list[str],
    estilos: dict[str, ParagraphStyle],
    nota_fuente: str = "",
    tamano_fuente: float = 9.0,
    padding_h: float = 4.0,
) -> list:
    """Construye título, tabla estilizada con encabezados repetidos y nota/fuente."""
    estilo_enc = ParagraphStyle(
        f"EncT{numero_tabla}",
        parent=estilos["CeldaEncabezado"],
        fontSize=tamano_fuente,
        leading=tamano_fuente + 2.5,
    )
    estilo_izq = ParagraphStyle(
        f"IzqT{numero_tabla}",
        parent=estilos["CeldaTexto"],
        fontSize=tamano_fuente,
        leading=tamano_fuente + 2.5,
    )
    estilo_der = ParagraphStyle(
        f"DerT{numero_tabla}",
        parent=estilos["CeldaNumero"],
        fontSize=tamano_fuente,
        leading=tamano_fuente + 2.5,
    )
    estilo_cen = ParagraphStyle(
        f"CenT{numero_tabla}",
        parent=estilos["CeldaTexto"],
        fontSize=tamano_fuente,
        leading=tamano_fuente + 2.5,
        alignment=TA_CENTER,
    )

    titulo_completo = f"Tabla {numero_tabla}. {titulo}"
    p_titulo = Paragraph(titulo_completo, estilos["TituloTabla"])
    p_titulo._tot_info = (0, titulo_completo, f"tabla_{numero_tabla}")

    datos_celdas = [[Paragraph(str(h), estilo_enc) for h in encabezados]]
    for fila in filas:
        fila_p = []
        for valor, alin in zip(fila, alineaciones):
            if alin == "R":
                fila_p.append(Paragraph(str(valor), estilo_der))
            elif alin == "C":
                fila_p.append(Paragraph(str(valor), estilo_cen))
            else:
                fila_p.append(Paragraph(str(valor), estilo_izq))
        datos_celdas.append(fila_p)

    tabla = Table(
        datos_celdas,
        colWidths=[w * cm for w in anchos_cm],
        repeatRows=1,
    )
    tabla.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARIO),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_FONDO_SUAVE]),
                ("GRID", (0, 0), (-1, -1), 0.4, COLOR_BORDE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
                ("LEFTPADDING", (0, 0), (-1, -1), padding_h),
                ("RIGHTPADDING", (0, 0), (-1, -1), padding_h),
            ]
        )
    )

    elementos: list = [p_titulo, tabla]
    if nota_fuente:
        elementos.append(Paragraph(nota_fuente, estilos["NotaTabla"]))
    else:
        elementos.append(Spacer(1, 6))
    return elementos


def _crear_figura_academica(
    numero_figura: int,
    ruta_imagen: Path,
    titulo: str,
    fuente: str,
    estilos: dict[str, ParagraphStyle],
    ancho_max_cm: float = 15.6,
    alto_max_cm: float = 9.0,
) -> KeepTogether:
    """Empaqueta una imagen proporcional con su número, título descriptivo y fuente."""
    img = _cargar_imagen_reportlab(ruta_imagen, ancho_max_cm, alto_max_cm)
    img.hAlign = "CENTER"
    titulo_completo = f"Figura {numero_figura}. {titulo}"
    p_pie = Paragraph(titulo_completo, estilos["PieFigura"])
    p_pie._tof_info = (0, titulo_completo, f"figura_{numero_figura}")
    p_fuente = Paragraph(f"Fuente: {fuente}", estilos["FuenteFigura"])
    return KeepTogether([Spacer(1, 2), img, p_pie, p_fuente])


def construir_portada(
    config: dict,
    ruta_logo: Path | None,
    fuente_serif: str,
) -> list:
    """Construye la portada institucional formal ocupando exactamente la página 1."""
    fuente_bold = (
        f"{fuente_serif}-Bold" if fuente_serif == "TimesNewRoman" else "Times-Bold"
    )
    estilo_uni = ParagraphStyle(
        "PortadaUni",
        fontName=fuente_bold,
        fontSize=15,
        leading=19,
        alignment=TA_CENTER,
        textColor=COLOR_PRIMARIO,
    )
    estilo_fac = ParagraphStyle(
        "PortadaFac",
        fontName=fuente_bold,
        fontSize=13.5,
        leading=17,
        alignment=TA_CENTER,
        textColor=COLOR_SECUNDARIO,
    )
    estilo_centro = ParagraphStyle(
        "PortadaCentro",
        fontName=fuente_serif,
        fontSize=12,
        leading=16.5,
        alignment=TA_CENTER,
        textColor=COLOR_TEXTO,
    )
    estilo_centro_bold = ParagraphStyle(
        "PortadaCentroBold",
        fontName=fuente_bold,
        fontSize=12.5,
        leading=17,
        alignment=TA_CENTER,
        textColor=COLOR_PRIMARIO,
    )
    estilo_titulo = ParagraphStyle(
        "PortadaTitulo",
        fontName=fuente_bold,
        fontSize=16.5,
        leading=22,
        alignment=TA_CENTER,
        textColor=COLOR_PRIMARIO,
    )

    story: list = []

    if ruta_logo is not None and ruta_logo.is_file():
        logo_img = _cargar_imagen_reportlab(ruta_logo, ancho_max_cm=16.0, alto_max_cm=2.2)
        logo_img.hAlign = "CENTER"
        story.append(logo_img)
        story.append(Spacer(1, 0.75 * cm))
    else:
        story.append(Spacer(1, 2.5 * cm))

    story.append(Paragraph(config["universidad"].upper(), estilo_uni))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(config["facultad"], estilo_fac))
    story.append(Spacer(1, 0.35 * cm))

    linea = Table([[""]], colWidths=[14.5 * cm], rowHeights=[1])
    linea.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 1.0, COLOR_PRIMARIO)]))
    story.append(linea)
    story.append(Spacer(1, 0.45 * cm))

    story.append(Paragraph(f"<b>Carrera:</b> {config['carrera']}", estilo_centro))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(f"<b>Materia:</b> {config['materia']}", estilo_centro))
    if config.get("grupo", "").strip():
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph(f"<b>Grupo:</b> {config['grupo'].strip()}", estilo_centro))

    story.append(Spacer(1, 0.9 * cm))
    story.append(Paragraph(config["tipo_trabajo"], estilo_centro_bold))
    story.append(Spacer(1, 0.45 * cm))
    story.append(Paragraph(config["titulo"], estilo_titulo))
    story.append(Spacer(1, 1.15 * cm))

    story.append(Paragraph("<b>Integrantes</b>", estilo_centro))
    story.append(Spacer(1, 0.15 * cm))
    for integrante in config["integrantes"]:
        story.append(Paragraph(integrante, estilo_centro))

    story.append(Spacer(1, 0.7 * cm))
    story.append(Paragraph("<b>Profesor</b>", estilo_centro))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(config["profesor"], estilo_centro))

    story.append(Spacer(1, 1.3 * cm))
    story.append(Paragraph(config["lugar"], estilo_centro))
    story.append(Paragraph(config["fecha"], estilo_centro))

    story.append(NextPageTemplate("Contenido"))
    story.append(PageBreak())
    return story


def crear_diagrama_flujo_sistema(fuente_serif: str) -> Drawing:
    """Construye un diagrama de flujo vectorial nativo de la arquitectura del proyecto."""
    fuente_bold = (
        f"{fuente_serif}-Bold" if fuente_serif == "TimesNewRoman" else "Times-Bold"
    )
    dibujo = Drawing(460, 295)

    cajas = [
        (110, 256, 240, 30, "Entrada: winequality-red.csv (1,599 × 12)", True),
        (20, 200, 200, 34, "Fase 1: Exploración, limpieza y EDA", False),
        (240, 200, 200, 34, "Salida: winequality-red-limpio.csv (1,359 × 12)", True),
        (20, 142, 200, 34, "Fase 2: Partición 80/20 (random_state=42)", False),
        (240, 142, 200, 34, "Fase 2: StandardScaler (solo en X_train)", False),
        (240, 84, 200, 34, "Fase 3: LinearRegression + joblib", False),
        (20, 84, 200, 34, "Fase 4: Evaluación en prueba | CV Ridge (X_train)", False),
        (90, 18, 280, 38, "Fase 5: Reporte Técnico Corregido PDF y Markdown", True),
    ]

    for x, y, w, h, texto, destacado in cajas:
        fondo = COLOR_PRIMARIO if destacado else COLOR_FONDO_SUAVE
        borde = COLOR_PRIMARIO
        color_txt = colors.white if destacado else COLOR_PRIMARIO
        dibujo.add(
            Rect(x, y, w, h, rx=4, ry=4, fillColor=fondo, strokeColor=borde, strokeWidth=1)
        )
        dibujo.add(
            String(
                x + w / 2,
                y + h / 2 - 3.2,
                texto,
                fontName=fuente_bold if destacado else fuente_serif,
                fontSize=8.0,
                fillColor=color_txt,
                textAnchor="middle",
            )
        )

    def _flecha(x1: float, y1: float, x2: float, y2: float) -> None:
        dibujo.add(Line(x1, y1, x2, y2, strokeColor=COLOR_SECUNDARIO, strokeWidth=1.1))
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) >= abs(dy):
            dir_x = 1 if dx > 0 else -1
            pts = [x2, y2, x2 - 5 * dir_x, y2 + 3, x2 - 5 * dir_x, y2 - 3]
        else:
            dir_y = 1 if dy > 0 else -1
            pts = [x2, y2, x2 - 3, y2 - 5 * dir_y, x2 + 3, y2 - 5 * dir_y]
        dibujo.add(Polygon(pts, fillColor=COLOR_SECUNDARIO, strokeColor=COLOR_SECUNDARIO))

    _flecha(180, 256, 120, 234)
    _flecha(220, 217, 240, 217)
    _flecha(340, 200, 120, 176)
    _flecha(220, 159, 240, 159)
    _flecha(340, 142, 340, 118)
    _flecha(240, 101, 220, 101)
    _flecha(120, 84, 200, 56)

    dibujo.hAlign = "CENTER"
    return dibujo


def _construir_texto_resumen(insumos: dict) -> tuple[str, str]:
    """Genera el texto del Resumen (150-250 palabras) reflejando las correcciones técnicas."""
    m = insumos["metricas"]
    mb = insumos["modelo_base"]
    tab_limp = insumos["tablas"]["comparacion_limpieza"]
    n_orig = int(tab_limp.loc[tab_limp["Característica"] == "Registros", "Antes"].iloc[0])
    n_dup = int(tab_limp.loc[tab_limp["Característica"] == "Duplicados", "Antes"].iloc[0])
    cfg_m = insumos["config_modelo"]
    n_train = int(cfg_m["registros_entrenamiento"])
    n_test = int(cfg_m["registros_prueba"])
    n_limpio = n_train + n_test
    df_cv = insumos["tablas"]["experimentacion_ridge_cv"]
    rmse_cv_ols = float(
        df_cv.loc[df_cv["Modelo"] == "LinearRegression", "RMSE_promedio_CV"].iloc[0]
    )

    texto = (
        f"Este reporte documenta el desarrollo, entrenamiento y evaluación de un modelo "
        f"de aprendizaje supervisado de regresión para estimar la calidad del vino tinto a "
        f"partir de variables fisicoquímicas. Se empleó el conjunto Wine Quality (Red Wine) del "
        f"repositorio UCI, integrado por {n_orig:,} registros y 12 columnas numéricas. Tras "
        f"comprobar cero valores nulos y retirar {n_dup} filas duplicadas para prevenir fuga "
        f"entre particiones, se conservaron {n_limpio:,} observaciones con 11 predictores y la "
        f"puntuación numérica discreta ordinal quality tratada como respuesta cuantitativa. Se "
        f"aplicó una partición aleatoria 80/20 ({n_train:,} entrenamiento y {n_test} prueba) y "
        f"estandarización mediante StandardScaler ajustado exclusivamente en entrenamiento, "
        f"entrenándose un modelo de Regresión Lineal Múltiple por mínimos cuadrados ordinarios. "
        f"En el conjunto de prueba se obtuvo MSE = {m['MSE']:.4f}, RMSE = {m['RMSE']:.4f}, "
        f"MAE = {m['MAE']:.4f} y R² = {m['R²']:.4f}, mejorando frente al predictor de media "
        f"constante (RMSE = {float(mb.loc['RMSE', 'Modelo base (DummyRegressor)']):.4f}) "
        f"sin brecha pronunciada respecto a entrenamiento en esta partición. Adicionalmente, "
        f"una validación cruzada 5-fold exclusiva sobre entrenamiento mostró desempeño similar "
        f"entre LinearRegression (RMSE = {rmse_cv_ols:.4f}) y Ridge (alpha de 0.01 a 100), por "
        f"lo que se conservó LinearRegression para muestras afines al conjunto estudiado."
    )
    palabras_clave = (
        "Inteligencia Artificial, Machine Learning, Regresión Lineal Múltiple, "
        "Wine Quality, StandardScaler, Ridge, Validación Cruzada."
    )
    return texto, palabras_clave


def validar_citas_y_referencias(texto_completo: str) -> None:
    """Verifica correspondencia bidireccional entre citas del texto y REFERENCIAS_APA7."""
    texto_plano = html.unescape(re.sub(r"<[^>]+>", "", texto_completo))

    # 1. Toda referencia declarada debe estar citada en el cuerpo
    faltantes = [
        ref["clave"]
        for ref in REFERENCIAS_APA7
        if not re.search(ref["patron"], texto_plano)
    ]
    if faltantes:
        raise ValueError(
            "Las siguientes referencias bibliográficas no fueron citadas en el cuerpo "
            f"del documento: {', '.join(faltantes)}"
        )

    # 2. Toda cita año-autor (entre paréntesis o narrativa) debe existir en REFERENCIAS_APA7
    coincidencias = re.findall(
        r"([A-Za-zÁÉÍÓÚáéíóúñÑ0-9.,&\s-]{0,55}?)\(([^()]*?\b(?:19|20)\d{2}[a-b]?\b[^()]*)\)",
        texto_plano,
    )
    no_registradas = []
    for prefijo, grupo_citas in coincidencias:
        for fragmento in grupo_citas.split(";"):
            frag = fragmento.strip()
            anios = re.findall(r"\b(?:19|20)\d{2}[a-b]?\b", frag)
            if not anios:
                continue
            tiene_autor = bool(re.search(r"[A-Za-zÁÉÍÓÚáéíóúñÑ]{2,}", frag))
            if tiene_autor:
                candidato = frag
            else:
                autor = prefijo.strip().split(",")[-1].strip()
                candidato = f"{autor} {frag}".strip()
            if not any(re.search(ref["patron"], candidato) for ref in REFERENCIAS_APA7):
                no_registradas.append(candidato)

    if no_registradas:
        raise ValueError(
            "Se detectaron citas en el cuerpo que no constan en REFERENCIAS_APA7: "
            + ", ".join(no_registradas)
        )


def _generar_contenido_compartido(insumos: dict) -> dict:
    """Prepara textos y estructuras compartidas entre el PDF y el Markdown editable."""
    resumen_texto, palabras_clave = _construir_texto_resumen(insumos)
    cantidad_palabras_resumen = len(resumen_texto.split())
    if not (150 <= cantidad_palabras_resumen <= 250):
        raise ValueError(
            f"El Resumen debe tener entre 150 y 250 palabras (actual: {cantidad_palabras_resumen})."
        )
    return {
        "resumen": resumen_texto,
        "palabras_clave": palabras_clave,
        "palabras_resumen": cantidad_palabras_resumen,
    }


def _construir_story_pdf(
    insumos: dict,
    estilos: dict[str, ParagraphStyle],
    fuente_serif: str,
) -> tuple[list, int, int, str]:
    """Construye la lista completa de flowables Platypus para el reporte PDF corregido."""
    cfg_r = insumos["config_reporte"]
    cfg_m = insumos["config_modelo"]
    tablas = insumos["tablas"]
    m = insumos["metricas"]
    tt = insumos["train_test"]
    mb = insumos["modelo_base"]
    res_stats = insumos["residuos_stats"]
    ver = insumos["versiones"]
    rev_x = insumos["revision_x"]
    iqr_ext = insumos["iqr_extremos"]
    textos_base = _generar_contenido_compartido(insumos)

    # Valores dinámicos verificados desde los CSV reales
    tab_limp = tablas["comparacion_limpieza"]
    n_orig = int(tab_limp.loc[tab_limp["Característica"] == "Registros", "Antes"].iloc[0])
    n_cols = int(tab_limp.loc[tab_limp["Característica"] == "Columnas", "Antes"].iloc[0])
    n_nulos = int(tab_limp.loc[tab_limp["Característica"] == "Valores faltantes", "Antes"].iloc[0])
    n_dup = int(tab_limp.loc[tab_limp["Característica"] == "Duplicados", "Antes"].iloc[0])

    n_train = int(cfg_m["registros_entrenamiento"])
    n_test = int(cfg_m["registros_prueba"])
    n_limpio = n_train + n_test
    pct_train = float(cfg_m["porcentaje_entrenamiento_real"])
    pct_test = float(cfg_m["porcentaje_prueba_real"])
    intercepto = float(cfg_m["intercepto"])

    df_dist = tablas["distribucion_calidad"].set_index("Calidad")
    n_cal5_limp = int(df_dist.loc[5, "Cantidad"])
    n_cal6_limp = int(df_dist.loc[6, "Cantidad"])
    n_56_limp = n_cal5_limp + n_cal6_limp
    pct_5_limp = float(df_dist.loc[5, "Porcentaje (%)"])
    pct_6_limp = float(df_dist.loc[6, "Porcentaje (%)"])
    pct_56_limp = (n_56_limp / n_limpio) * 100
    pct_3_limp = float(df_dist.loc[3, "Porcentaje (%)"])
    pct_8_limp = float(df_dist.loc[8, "Porcentaje (%)"])

    df_pred_all = tablas["predicciones_modelo"]
    pred_min = float(df_pred_all["Calidad_predicha"].min())
    pred_max = float(df_pred_all["Calidad_predicha"].max())

    df_cal_idx = tablas["rendimiento_por_calidad"].set_index("Calidad_real")
    n_56_test = int(df_cal_idx.loc[5, "Cantidad_registros"]) + int(
        df_cal_idx.loc[6, "Cantidad_registros"]
    )
    pct_56_test = (n_56_test / n_test) * 100

    df_corr_q = tablas["correlaciones_con_quality"].set_index("Variable")
    df_coef_idx = tablas["coeficientes_regresion"].set_index("Variable")
    df_esc_idx = tablas["comparacion_escalado"].set_index("Variable")

    df_cv = tablas["experimentacion_ridge_cv"]
    cv_ols = df_cv[df_cv["Modelo"] == "LinearRegression"].iloc[0]
    cv_ridge = df_cv[df_cv["Modelo"] == "Ridge"].copy()
    mejor_ridge = cv_ridge.loc[cv_ridge["RMSE_promedio_CV"].idxmin()]

    num_tabla = 0
    num_figura = 0

    def sig_tabla() -> int:
        nonlocal num_tabla
        num_tabla += 1
        return num_tabla

    def sig_figura() -> int:
        nonlocal num_figura
        num_figura += 1
        return num_figura

    story: list = []
    texto_para_citas: list[str] = []

    def add_p(html: str, estilo_key: str = "Cuerpo") -> None:
        story.append(Paragraph(html, estilos[estilo_key]))
        texto_para_citas.append(html)

    def add_p_formula(html_intro: str, formula_html: str) -> None:
        """Mantiene unido el párrafo introductorio con su fórmula (y con su subtítulo)."""
        p_intro = Paragraph(html_intro, estilos["CuerpoConFormula"])
        texto_para_citas.append(html_intro)
        story.append(p_intro)
        story.append(_crear_formula(formula_html, estilos))
        story.append(Spacer(1, 6))

    def add_h1(titulo: str, clave: str) -> None:
        story.append(_crear_heading_toc(titulo, 0, clave, estilos["TituloPrincipal"]))

    def add_h2(titulo: str, clave: str) -> None:
        story.append(_crear_heading_toc(titulo, 1, clave, estilos["Subtitulo"]))

    # ------------------------------------------------------------------
    # PORTADA (PÁGINA 1 EXACTA)
    # ------------------------------------------------------------------
    story.extend(construir_portada(cfg_r, insumos["ruta_logo"], fuente_serif))

    # ------------------------------------------------------------------
    # RESUMEN (PÁGINA 2)
    # ------------------------------------------------------------------
    add_h1("Resumen", "sec_resumen")
    add_p(textos_base["resumen"])
    add_p(f"<b>Palabras clave:</b> {textos_base['palabras_clave']}")
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # ÍNDICE GENERAL, DE TABLAS Y DE FIGURAS
    # ------------------------------------------------------------------
    story.append(Paragraph("Índice general", estilos["TituloPrincipal"]))
    toc = TableOfContents(
        levelStyles=[estilos["TOCNivel0"], estilos["TOCNivel1"]],
        dotsMinLevel=0,
    )
    story.append(toc)
    story.append(PageBreak())

    story.append(Paragraph("Índice de tablas", estilos["TituloPrincipal"]))
    tot = TableOfContents(
        notifyKind="TOTEntry",
        levelStyles=[estilos["TOCElemento"]],
        dotsMinLevel=0,
    )
    story.append(tot)
    story.append(PageBreak())

    story.append(Paragraph("Índice de figuras", estilos["TituloPrincipal"]))
    tof = TableOfContents(
        notifyKind="TOFEntry",
        levelStyles=[estilos["TOCElemento"]],
        dotsMinLevel=0,
    )
    story.append(tof)
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # 1. INTRODUCCIÓN
    # ------------------------------------------------------------------
    add_h1("1. Introducción", "sec_1")
    add_p(
        "La Inteligencia Artificial (IA) comprende el diseño de sistemas computacionales "
        "capaces de realizar tareas asociadas con la inteligencia humana, como el reconocimiento "
        "de patrones, el razonamiento lógico y la toma de decisiones fundamentadas en datos "
        "(Russell &amp; Norvig, 2021). Dentro de esta disciplina, el aprendizaje automático "
        "(<i>Machine Learning</i>) constituye la rama orientada a desarrollar algoritmos que "
        "aprenden relaciones directamente a partir de observaciones históricas sin depender de "
        "reglas programadas de forma explícita."
    )
    add_p(
        "El aprendizaje supervisado ocupa un lugar central en la ciencia de datos aplicada, "
        "pues permite entrenar modelos usando ejemplos etiquetados con una respuesta conocida "
        "(James et al., 2023). Cuando dicha respuesta adopta valores numéricos continuos u "
        "ordinales en una escala cuantitativa graduada, los modelos de regresión permiten "
        "estimar tendencias, cuantificar la asociación parcial de cada predictor y generar "
        "estimaciones interpretables."
    )
    add_p(
        "En la industria vitivinícola, la evaluación de la calidad suele realizarse mediante "
        "paneles de catadores expertos, un proceso costoso, lento y sujeto a variabilidad "
        "humana. Disponer de modelos capaces de relacionar mediciones fisicoquímicas de "
        "laboratorio con la calidad percibida ofrece apoyo objetivo para el control de "
        "calidad y la comprensión de los atributos químicos del vino (Cortez et al., 2009a). "
        "La finalidad de este proyecto es implementar de manera íntegra, reproducible y "
        "metodológicamente rigurosa las cinco fases de un flujo de regresión lineal múltiple "
        "aplicado al vino tinto, complementado con una experimentación de regularización Ridge "
        "mediante validación cruzada."
    )

    # ------------------------------------------------------------------
    # 2. PLANTEAMIENTO DEL PROBLEMA
    # ------------------------------------------------------------------
    add_h1("2. Planteamiento del problema", "sec_2")
    add_p(
        "El control fisicoquímico rutinario en bodegas proporciona mediciones objetivas de "
        "acidez, azúcar residual, cloruros, compuestos sulfurosos, densidad, pH y grado "
        "alcohólico; sin embargo, interpretar simultáneamente once variables químicas para "
        "anticipar la puntuación sensorial final de un vino tinto resulta complejo sin una "
        "herramienta analítica formal (Cortez et al., 2009a)."
    )
    add_p(
        "El problema se formuló como una tarea de <b>regresión supervisada</b> y no de "
        "clasificación porque la variable objetivo <i>quality</i> es una <b>puntuación numérica "
        "discreta de naturaleza ordinal</b> en escala de 0 a 10 (con valores observados enteros "
        "entre 3 y 8) que se trata como <b>respuesta cuantitativa</b> mediante regresión. "
        "Convertir dicha escala en categorías binarias descartaría diferencias ordinales "
        "relevantes entre puntuaciones intermedias y extremas, mientras que la regresión "
        "preserva el orden numérico de la escala, cuantifica desviaciones mediante métricas de "
        "distancia (MSE, RMSE, MAE) y evalúa la proporción de variabilidad explicada (R²)."
    )

    # ------------------------------------------------------------------
    # 3. OBJETIVOS
    # ------------------------------------------------------------------
    add_h1("3. Objetivos", "sec_3")
    add_h2("3.1 Objetivo general", "sec_3_1")
    add_p(
        "Desarrollar, entrenar y evaluar de forma crítica un modelo de aprendizaje supervisado "
        "basado en Regresión Lineal Múltiple para predecir la puntuación de calidad del vino "
        "tinto a partir de sus características fisicoquímicas, garantizando buenas prácticas "
        "de limpieza, partición, estandarización sin fuga de datos y documentación académica."
    )
    add_h2("3.2 Objetivos específicos", "sec_3_2")
    for obj in [
        "Explorar la estructura, los tipos de datos, la presencia de valores faltantes y los registros duplicados del conjunto de datos original Wine Quality (Red Wine).",
        "Limpiar y preparar el conjunto de datos eliminando duplicados exactos, conservando las observaciones atípicas ante la falta de evidencia de error y aplicando la partición 80/20 con StandardScaler.",
        "Entrenar un modelo de Regresión Lineal Múltiple mediante mínimos cuadrados ordinarios utilizando exclusivamente el conjunto de entrenamiento estandarizado.",
        "Evaluar el rendimiento del modelo principal sobre el conjunto de prueba reservado mediante las métricas MSE, RMSE, MAE y R², el análisis de residuos y la comparación frente a un modelo base, además de realizar un análisis complementario con Ridge en validación cruzada.",
        "Interpretar críticamente los coeficientes, las métricas y las limitaciones metodológicas dentro del alcance real de la evidencia obtenida.",
    ]:
        add_p(f"• {obj}", "Lista")

    # ------------------------------------------------------------------
    # 4. MARCO TEÓRICO
    # ------------------------------------------------------------------
    add_h1("4. Marco teórico", "sec_4")
    add_h2("4.1 Inteligencia Artificial", "sec_4_1")
    add_p(
        "La Inteligencia Artificial estudia agentes y sistemas computacionales que perciben su "
        "entorno y ejecutan acciones o inferencias orientadas a alcanzar objetivos de manera "
        "racional (Russell &amp; Norvig, 2021). Sus aplicaciones abarcan desde sistemas expertos "
        "simbólicos hasta modelos estadísticos de predicción basados en grandes volúmenes de datos."
    )
    add_h2("4.2 Aprendizaje automático", "sec_4_2")
    add_p(
        "El aprendizaje automático (<i>Machine Learning</i>) es una subárea de la IA centrada en "
        "algoritmos que mejoran su desempeño en una tarea específica a partir de la experiencia "
        "representada por datos históricos, ajustando parámetros internos para estimar salidas "
        "sobre ejemplos no vistos (James et al., 2023)."
    )
    add_h2("4.3 Aprendizaje supervisado", "sec_4_3")
    add_p(
        "En el aprendizaje supervisado el modelo recibe pares de entrenamiento (x<sub>i</sub>, y<sub>i</sub>), "
        "donde x<sub>i</sub> representa el vector de variables independientes o características y "
        "y<sub>i</sub> la variable objetivo observada. El objetivo consiste en aproximar una función "
        "f(x) capaz de estimar y para nuevas observaciones (Pedregosa et al., 2011)."
    )
    add_h2("4.4 Diferencia entre clasificación y regresión", "sec_4_4")
    add_p(
        "Mientras la clasificación asigna cada observación a una etiqueta categórica discreta "
        "(por ejemplo, vino aceptable frente a rechazado), la regresión estima una respuesta "
        "numérica cuantitativa sobre una escala continua u ordinal graduada, preservando las "
        "distancias e intervalos entre puntuaciones (James et al., 2023)."
    )
    add_h2("4.5 Regresión lineal simple", "sec_4_5")
    add_p_formula(
        "La regresión lineal simple modela la relación entre una única variable explicativa x "
        "y una variable dependiente y mediante una recta definida por un intercepto β<sub>0</sub> "
        "y una pendiente β<sub>1</sub>:",
        "ŷ = β<sub>0</sub> + β<sub>1</sub>·x",
    )
    add_p(
        "Donde ŷ es el valor estimado, β<sub>0</sub> representa la predicción cuando x = 0 y "
        "β<sub>1</sub> mide el cambio esperado en ŷ por cada unidad adicional de x."
    )

    add_h2("4.6 Regresión lineal múltiple", "sec_4_6")
    add_p_formula(
        "La regresión lineal múltiple generaliza el modelo anterior para incorporar p variables "
        "independientes simultáneamente (p = 11 en este estudio), permitiendo analizar el efecto "
        "parcial de cada característica manteniendo constantes las restantes:",
        "ŷ = b<sub>0</sub> + b<sub>1</sub>·x<sub>1</sub> + b<sub>2</sub>·x<sub>2</sub> + "
        "··· + b<sub>p</sub>·x<sub>p</sub>",
    )
    add_p(
        "En esta expresión, b<sub>0</sub> es el intercepto, b<sub>1</sub>,…,b<sub>p</sub> son los "
        "coeficientes de regresión y x<sub>1</sub>,…,x<sub>p</sub> corresponden a las características "
        "de entrada."
    )

    add_h2("4.7 Método de mínimos cuadrados ordinarios", "sec_4_7")
    add_p_formula(
        "El método de mínimos cuadrados ordinarios (OLS, por sus siglas en inglés) determina "
        "los parámetros b<sub>0</sub>, b<sub>1</sub>,…,b<sub>p</sub> minimizando la suma de los "
        "cuadrados de los residuos entre los valores observados y<sub>i</sub> y los valores "
        "predichos ŷ<sub>i</sub> sobre las n<sub>train</sub> observaciones de entrenamiento "
        "(James et al., 2023):",
        "min<sub>b</sub> ∑<sub>i=1..n</sub> (y<sub>i</sub> − ŷ<sub>i</sub>)<sup>2</sup>",
    )

    add_h2("4.8 Preprocesamiento de datos", "sec_4_8")
    add_p(
        "El preprocesamiento abarca la inspección de tipos, la detección de valores ausentes, "
        "el tratamiento justificado de registros duplicados, el examen de observaciones atípicas "
        "y la partición controlada en subconjuntos de entrenamiento y prueba antes de entrenar el "
        "algoritmo (McKinney, 2010)."
    )
    add_h2("4.9 Estandarización mediante StandardScaler", "sec_4_9")
    add_p_formula(
        "Cuando las variables independientes presentan escalas y unidades heterogéneas, la "
        "estandarización z transforma cada observación j restando la media muestral μ<sub>j</sub> "
        "del conjunto de entrenamiento y dividiendo entre su desviación estándar poblacional "
        "σ<sub>j</sub> (ddof = 0) (scikit-learn developers, 2026):",
        "z<sub>j</sub> = (x<sub>j</sub> − μ<sub>j</sub>) / σ<sub>j</sub>",
    )
    add_p(
        "Como resultado, cada variable transformada z<sub>j</sub> en entrenamiento queda centrada "
        "en media 0 y desviación estándar 1, permitiendo comparar directamente la magnitud relativa "
        "de los coeficientes lineales estimados sobre características estandarizadas."
    )

    add_h2("4.10 Fuga de datos o data leakage", "sec_4_10")
    add_p(
        "La fuga de datos ocurre cuando información proveniente del conjunto de prueba se filtra "
        "hacia el entrenamiento o hacia la transformación de características (por ejemplo, al "
        "calcular medias globales de escalado o mantener observaciones duplicadas entre ambos "
        "subconjuntos), produciendo estimaciones artificialmente optimistas del rendimiento "
        "(James et al., 2023; scikit-learn developers, 2026)."
    )

    add_h2("4.11 Sobreajuste y subajuste", "sec_4_11")
    add_p(
        "El sobreajuste (<i>overfitting</i>) sucede cuando un modelo memoriza particularidades o "
        "ruido del conjunto de entrenamiento logrando un error reducido en entrenamiento pero "
        "deteriorado al evaluarse fuera de muestra. En contraste, el subajuste (<i>underfitting</i>) "
        "ocurre cuando la estructura funcional resulta demasiado simple para capturar la señal "
        "subyacente; sin embargo, un ajuste bajo por sí solo no demuestra subajuste cuando existe "
        "ruido elevado o ausencia de variables relevantes (James et al., 2023)."
    )

    add_h2("4.12 Métricas de evaluación para regresión", "sec_4_12")
    p_intro_m = Paragraph(
        "Para cuantificar el ajuste en regresión sobre n observaciones de prueba, donde "
        "y<sub>i</sub> es el valor real, ŷ<sub>i</sub> el predicho y ȳ el promedio real "
        "(<i>y_media</i>), se emplean cuatro métricas complementarias (scikit-learn developers, 2026):",
        estilos["CuerpoConFormula"],
    )
    texto_para_citas.append(p_intro_m.text)
    story.append(p_intro_m)
    story.append(
        KeepTogether(
            [
                _crear_formula(
                    "MSE = (1 / n) · ∑ (y<sub>i</sub> − ŷ<sub>i</sub>)<sup>2</sup>"
                    "&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;"
                    "RMSE = √(MSE)",
                    estilos,
                ),
                Spacer(1, 4),
                _crear_formula(
                    "MAE = (1 / n) · ∑ |y<sub>i</sub> − ŷ<sub>i</sub>|"
                    "&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;"
                    "R<sup>2</sup> = 1 − [ ∑ (y<sub>i</sub> − ŷ<sub>i</sub>)<sup>2</sup> / "
                    "∑ (y<sub>i</sub> − ȳ)<sup>2</sup> ]",
                    estilos,
                ),
                Spacer(1, 5),
            ]
        )
    )
    add_p(
        "El MSE mide el <b>promedio de los errores elevados al cuadrado</b>, penalizando con "
        "mayor peso las desviaciones grandes; el RMSE devuelve ese error cuadrático a las unidades "
        "originales de la variable objetivo; el MAE expresa el promedio de las diferencias "
        "absolutas en puntos de calidad; y el R² indica la proporción de la variabilidad total "
        "observada que logra explicar el modelo frente a predecir simplemente el promedio ȳ."
    )

    # ------------------------------------------------------------------
    # 5. HERRAMIENTAS Y TECNOLOGÍAS
    # ------------------------------------------------------------------
    add_h1("5. Herramientas y tecnologías", "sec_5")
    add_p(
        f"El proyecto se implementó íntegramente en Python {ver['python']} dentro de un entorno "
        "virtual dedicado (.venv), integrando bibliotecas científicas estándar y herramientas de "
        "desarrollo y documentación automatizada. Para la manipulación tabular se empleó pandas "
        "(McKinney, 2010; pandas development team, 2026) sobre arreglos numéricos de NumPy "
        "(Harris et al., 2020); la visualización se realizó con Matplotlib (Hunter, 2007) y "
        "Seaborn (Waskom, 2021); el modelado supervisado y su persistencia se apoyaron en "
        "Scikit-learn (Pedregosa et al., 2011; scikit-learn developers, 2026) y Joblib; y la "
        "compilación documental se efectuó con ReportLab y Pillow, como se detalla en la Tabla 1."
    )
    filas_herramientas = [
        ["Python", ver["python"], "Lenguaje principal para la lógica modular, flujo por consola y procesamiento."],
        ["pandas", ver["pandas"], "Carga, exploración, limpieza tabular, separación de X e y y exportación CSV (McKinney, 2010; pandas development team, 2026)."],
        ["NumPy", ver["numpy"], "Cálculo numérico vectorizado de residuos, errores cuadráticos y raíz RMSE (Harris et al., 2020)."],
        ["Matplotlib", ver["matplotlib"], "Generación de figuras y exportación de las 21 gráficas a 300 DPI (Hunter, 2007)."],
        ["Seaborn", ver["seaborn"], "Visualizaciones estadísticas como mapas de calor, boxplots e histogramas con KDE (Waskom, 2021)."],
        ["Scikit-learn", ver["scikit-learn"], "Partición train_test_split, StandardScaler, Pipeline, KFold, LinearRegression, Ridge, DummyRegressor y métricas (Pedregosa et al., 2011)."],
        ["Joblib", ver["joblib"], "Serialización y recarga binaria de regresion_lineal.joblib y standard_scaler.joblib."],
        ["ReportLab", ver["reportlab"], "Motor Platypus utilizado para compilar este reporte técnico PDF con índices reales y portada."],
        ["Pillow", ver["pillow"], "Carga y composición en memoria de logotipos institucionales e imágenes para el reporte."],
        ["PyCharm", "IDE", "Entorno integrado empleado para edición, inspección de archivos y gestión del proyecto."],
        ["Antigravity", "Asistente IA", "Apoyo en la estructuración modular, verificación de resultados y automatización documental."],
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Herramientas y bibliotecas verificadas en el entorno del proyecto",
            ["Herramienta", "Versión", "Función desempeñada en el proyecto"],
            filas_herramientas,
            [2.9, 2.2, 11.3],
            ["L", "C", "L"],
            estilos,
            nota_fuente="Fuente: Elaboración propia a partir de requirements.txt y el entorno virtual .venv.",
        )
    )

    # ------------------------------------------------------------------
    # 6. DESCRIPCIÓN DEL CONJUNTO DE DATOS
    # ------------------------------------------------------------------
    add_h1("6. Descripción del conjunto de datos", "sec_6")
    add_p(
        "Se utilizó el conjunto de datos <i>Wine Quality (Red Wine)</i>, publicado en el UCI "
        "Machine Learning Repository (Cortez et al., 2009b) y originado en muestras de vino tinto "
        "portugués <i>Vinho Verde</i> analizadas por Cortez et al. (2009a). El repositorio oficial "
        "se encuentra disponible en https://archive.ics.uci.edu/dataset/186/wine+quality "
        "(DOI: https://doi.org/10.24432/C56S3T)."
    )
    add_p(
        f"El archivo original (winequality-red.csv) utiliza punto y coma (;) como delimitador y "
        f"contiene <b>{n_orig:,} registros</b> distribuidos en <b>{n_cols} columnas numéricas</b>: "
        "11 características fisicoquímicas continuas de tipo flotante (float64) y 1 variable "
        "objetivo entera (int64) correspondiente a la <b>puntuación numérica discreta de naturaleza "
        "ordinal</b> otorgada por catadores como mediana sensorial en escala de 0 a 10 (valores "
        "observados entre 3 y 8). Este conjunto fue seleccionado por su relevancia académica, "
        "ausencia de datos faltantes y presencia de múltiples atributos químicos medibles para "
        "evaluar regresión lineal múltiple."
    )
    add_p(
        "Para formular el problema supervisado, la matriz de características <b>X</b> se define "
        "como X = datos.drop(columns=[\"quality\"]) conservando las 11 variables independientes "
        "originales, mientras que el vector objetivo <b>y</b> se define como y = datos[\"quality\"] "
        "tratando esta escala numérica discreta como respuesta cuantitativa continua aproximada "
        "sin discretizarla en clases."
    )
    filas_vars = [
        ["fixed acidity", "X (Independiente)", "float64", "Acidez fija (ácido tartárico principal); aporta estructura al vino (g/dm³)."],
        ["volatile acidity", "X (Independiente)", "float64", "Acidez volátil (ácido acético); niveles elevados producen sabor avinagrado (g/dm³)."],
        ["citric acid", "X (Independiente)", "float64", "Ácido cítrico; aporta frescura y matices frutales (g/dm³)."],
        ["residual sugar", "X (Independiente)", "float64", "Azúcar remanente tras finalizar la fermentación (g/dm³)."],
        ["chlorides", "X (Independiente)", "float64", "Concentración de sales de cloruro presentes en el vino (g/dm³)."],
        ["free sulfur dioxide", "X (Independiente)", "float64", "Dióxido de azufre libre con función antioxidante y antimicrobiana (mg/dm³)."],
        ["total sulfur dioxide", "X (Independiente)", "float64", "Suma de dióxido de azufre libre y combinado (mg/dm³)."],
        ["density", "X (Independiente)", "float64", "Densidad del líquido vinculada con el balance entre alcohol y azúcar (g/cm³)."],
        ["pH", "X (Independiente)", "float64", "Nivel de acidez en escala logarítmica de pH (típicamente entre 2.7 y 4.0)."],
        ["sulphates", "X (Independiente)", "float64", "Aditivo sulfatado que estabiliza el vino y el nivel de SO₂ (g/dm³)."],
        ["alcohol", "X (Independiente)", "float64", "Porcentaje volumétrico de alcohol (% vol.)."],
        ["quality", "y (Dependiente)", "int64", "Puntuación numérica discreta de naturaleza ordinal (escala 0–10; observada de 3 a 8), tratada como respuesta cuantitativa."],
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Descripción de las variables independientes (X) y la variable objetivo (y)",
            ["Variable (CSV)", "Rol", "Tipo", "Descripción fisicoquímica / sensorial"],
            filas_vars,
            [3.2, 2.7, 1.5, 9.0],
            ["L", "C", "C", "L"],
            estilos,
            nota_fuente="Fuente: Elaboración propia con base en Cortez et al. (2009a, 2009b).",
        )
    )

    # ------------------------------------------------------------------
    # 7. FASE 1: EXPLORACIÓN, LIMPIEZA Y ANÁLISIS DE DATOS
    # ------------------------------------------------------------------
    add_h1("7. Fase 1: Exploración, limpieza y análisis de datos", "sec_7")
    add_p(
        "La primera fase se ejecutó mediante los módulos Exploracion_Datos.py, Limpieza_Datos.py, "
        "Identificar_X_y_Y.py y Analisis_Datos.py (McKinney, 2010; Waskom, 2021). Al cargar "
        f"winequality-red.csv con separador punto y coma (;), se verificaron {n_orig:,} registros, "
        f"{n_cols} columnas numéricas, <b>{n_nulos} valores faltantes</b> en todas las variables y "
        f"<b>{n_dup} filas completamente duplicadas</b>."
    )
    add_p(
        f"Durante la limpieza se eliminaron las {n_dup} filas idénticas con drop_duplicates() y "
        "reset_index(drop=True), guardando el resultado en winequality-red-limpio.csv sin "
        "sobrescribir el archivo original. Desde el punto de vista metodológico, esta eliminación "
        "evita que filas exactamente iguales aparezcan simultáneamente en entrenamiento y prueba "
        "(reduciendo riesgo de fuga de datos), aun reconociendo que en laboratorio dos lotes o "
        "muestras podrían llegar a coincidir en sus mediciones. En cambio, <b>no se eliminaron "
        "observaciones atípicas</b> porque corresponden a valores numéricos válidos dentro de "
        "rangos fisicoquímicos plausibles y <b>se carece de evidencia suficiente para considerarlos "
        "errores de medición o captura</b>."
    )

    df_limp = tablas["comparacion_limpieza"]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Comparación estructural antes y después de la limpieza de datos",
            list(df_limp.columns),
            [[str(v) for v in fila] for fila in df_limp.values],
            [6.4, 5.0, 5.0],
            ["L", "R", "R"],
            estilos,
            nota_fuente="Fuente: resultados/tablas/comparacion_limpieza.csv.",
        )
    )

    df_desc = tablas["estadisticas_descriptivas"]
    filas_desc = [
        [
            str(f["Variable"]),
            f"{int(f['Cantidad'])}",
            f"{f['Media']:.4f}",
            f"{f['Desviación estándar']:.4f}",
            f"{f['Mínimo']:.4f}",
            f"{f['Q1 (25%)']:.4f}",
            f"{f['Mediana (50%)']:.4f}",
            f"{f['Q3 (75%)']:.4f}",
            f"{f['Máximo']:.4f}",
        ]
        for _, f in df_desc.iterrows()
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            f"Estadísticas descriptivas del dataset limpio (n = {n_limpio:,})",
            ["Variable", "n", "Media", "Desv. Est.", "Mín.", "Q1 (25%)", "Mediana", "Q3 (75%)", "Máx."],
            filas_desc,
            [3.2, 1.1, 1.7, 1.75, 1.6, 1.7, 1.7, 1.7, 1.85],
            ["L", "R", "R", "R", "R", "R", "R", "R", "R"],
            estilos,
            nota_fuente="Fuente: resultados/tablas/estadisticas_descriptivas.csv.",
            tamano_fuente=8.5,
            padding_h=2.5,
        )
    )

    df_dist_t = tablas["distribucion_calidad"]
    filas_dist = [
        [str(int(f["Calidad"])), str(int(f["Cantidad"])), f"{float(f['Porcentaje (%)']):.2f}%"]
        for _, f in df_dist_t.iterrows()
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Distribución de puntuaciones de calidad en el dataset limpio",
            ["Puntuación (quality)", "Cantidad de registros", "Porcentaje (%)"],
            filas_dist,
            [5.4, 5.5, 5.5],
            ["C", "R", "R"],
            estilos,
            nota_fuente="Fuente: resultados/tablas/distribucion_calidad.csv.",
        )
    )

    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "01_distribucion_calidad.png",
            "Distribución de la puntuación de calidad del vino tinto en el dataset limpio",
            "Elaboración propia a partir de winequality-red-limpio.csv.",
            estilos,
            ancho_max_cm=15.0,
            alto_max_cm=8.5,
        )
    )
    add_p(
        f"Como se observa en la Tabla 5 y la Figura 1, el <b>{pct_56_limp:.2f}%</b> de las "
        f"muestras limpias ({n_56_limp:,} de {n_limpio:,} registros) se concentra en las "
        f"puntuaciones 5 ({pct_5_limp:.2f}%) y 6 ({pct_6_limp:.2f}%), mientras que las notas "
        f"extremas 3 ({pct_3_limp:.2f}%) y 8 ({pct_8_limp:.2f}%) cuentan con muy pocos casos."
    )

    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "02_matriz_correlacion.png",
            "Matriz de correlación de Pearson entre variables fisicoquímicas y calidad",
            "Elaboración propia con Seaborn sobre winequality-red-limpio.csv.",
            estilos,
            ancho_max_cm=15.4,
            alto_max_cm=13.0,
        )
    )
    add_p(
        "La Figura 2 evidencia correlaciones relevantes entre variables independientes, destacando "
        "<i>fixed acidity</i> con <i>pH</i> (r = −0.69), <i>fixed acidity</i> con <i>density</i> "
        "(r = +0.67), <i>fixed acidity</i> con <i>citric acid</i> (r = +0.67) y <i>free sulfur "
        "dioxide</i> con <i>total sulfur dioxide</i> (r = +0.67), así como una asociación "
        "negativa entre <i>alcohol</i> y <i>density</i> (r = −0.50). Esta colinealidad entre "
        "predictores resulta relevante al interpretar los coeficientes parciales de la regresión múltiple."
    )

    df_corr_obj = tablas["correlaciones_con_quality"]
    filas_corr_obj = [
        [str(f["Variable"]), f"{float(f['Correlación con quality']):+.4f}"]
        for _, f in df_corr_obj.iterrows()
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Correlación lineal de Pearson de cada variable independiente con quality",
            ["Variable independiente", "Correlación de Pearson con quality"],
            filas_corr_obj,
            [9.0, 7.4],
            ["L", "R"],
            estilos,
            nota_fuente="Fuente: resultados/tablas/correlaciones_con_quality.csv.",
        )
    )

    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "03_correlacion_con_quality.png",
            "Correlaciones ordenadas de las variables independientes con la calidad",
            "Elaboración propia con base en correlaciones_con_quality.csv.",
            estilos,
            ancho_max_cm=15.2,
            alto_max_cm=8.6,
        )
    )
    add_p(
        "La Tabla 6 y la Figura 3 muestran que <i>alcohol</i> presenta la correlación lineal "
        f"positiva más alta con <i>quality</i> ({float(df_corr_q.loc['alcohol', 'Correlación con quality']):+.4f}), "
        f"seguido de <i>sulphates</i> ({float(df_corr_q.loc['sulphates', 'Correlación con quality']):+.4f}) "
        f"y <i>citric acid</i> ({float(df_corr_q.loc['citric acid', 'Correlación con quality']):+.4f}), "
        f"mientras que <i>volatile acidity</i> muestra la mayor correlación negativa "
        f"({float(df_corr_q.loc['volatile acidity', 'Correlación con quality']):+.4f}). "
        "Siguiendo el criterio metodológico del proyecto, no se eliminó ninguna variable "
        "únicamente por su correlación bivariada baja, ya que puede aportar información en "
        "combinación con el resto de los predictores."
    )

    df_iqr = tablas["valores_atipicos_iqr"]
    filas_iqr = [
        [
            str(f["Variable"]),
            f"{float(f['Q1']):.4f}",
            f"{float(f['Q3']):.4f}",
            f"{float(f['IQR']):.4f}",
            f"{float(f['Límite inferior']):.4f}",
            f"{float(f['Límite superior']):.4f}",
            str(int(f["Valores atípicos"])),
            f"{float(f['Porcentaje (%)']):.2f}%",
        ]
        for _, f in df_iqr.iterrows()
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Detección de posibles valores atípicos mediante el método del rango intercuartílico (IQR)",
            ["Variable", "Q1", "Q3", "IQR", "Lím. inf.", "Lím. sup.", "Atípicos", "%"],
            filas_iqr,
            [3.2, 1.7, 1.7, 1.7, 2.0, 2.0, 2.0, 1.9],
            ["L", "R", "R", "R", "R", "R", "R", "R"],
            estilos,
            nota_fuente="Fuente: resultados/tablas/valores_atipicos_iqr.csv. Todas las observaciones se conservaron sin eliminar atípicos.",
            tamano_fuente=8.5,
            padding_h=2.5,
        )
    )

    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "04_boxplots_variables.png",
            "Diagramas de caja de las 11 variables independientes y conteo de atípicos IQR",
            "Elaboración propia a partir de winequality-red-limpio.csv.",
            estilos,
            ancho_max_cm=16.4,
            alto_max_cm=11.6,
        )
    )
    add_p(
        "El criterio del rango intercuartílico (IQR = Q3 − Q1) clasifica como posibles valores "
        "atípicos aquellas observaciones ubicadas <b>por debajo del límite inferior "
        "(Q1 − 1.5·IQR) o por encima del límite superior (Q3 + 1.5·IQR)</b>. La Tabla 7 y la "
        f"Figura 4 muestran que <i>residual sugar</i> ({int(iqr_ext['residual sugar']['total'])} "
        f"registros, {iqr_ext['residual sugar']['porcentaje']:.2f}%) y <i>chlorides</i> "
        f"({int(iqr_ext['chlorides']['total'])} registros, {iqr_ext['chlorides']['porcentaje']:.2f}%; "
        f"{int(iqr_ext['chlorides']['inferior'])} por debajo del límite inferior y "
        f"{int(iqr_ext['chlorides']['superior'])} por encima del superior) concentran las "
        "mayores proporciones fuera de rango, mientras que variables simétricas como <i>density</i> "
        f"({int(iqr_ext['density']['inferior'])} inferiores y {int(iqr_ext['density']['superior'])} "
        f"superiores) y <i>pH</i> ({int(iqr_ext['pH']['inferior'])} inferiores y "
        f"{int(iqr_ext['pH']['superior'])} superiores) exhiben observaciones fuera de ambos "
        "extremos. Todas estas observaciones se conservaron por ausencia de evidencia de error."
    )

    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "05_histogramas_variables.png",
            "Histogramas y curvas de densidad de las 11 variables independientes",
            "Elaboración propia a partir de winequality-red-limpio.csv.",
            estilos,
            ancho_max_cm=16.4,
            alto_max_cm=11.6,
        )
    )
    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "06_dispersion_calidad.png",
            "Gráficas de dispersión de alcohol, acidez volátil, sulfatos y ácido cítrico frente a quality",
            "Elaboración propia a partir de winequality-red-limpio.csv.",
            estilos,
            ancho_max_cm=16.0,
            alto_max_cm=11.2,
        )
    )
    add_p(
        "Los histogramas de la Figura 5 muestran distribuciones aproximadamente simétricas en "
        "<i>pH</i> y <i>density</i>, frente a marcados sesgos positivos a la derecha en "
        "<i>residual sugar</i>, <i>chlorides</i> y ambos dióxidos de azufre. A su vez, la "
        "Figura 6 ilustra la tendencia ascendente de <i>alcohol</i> y <i>sulphates</i> respecto a "
        "puntuaciones mayores, en contraste con la tendencia descendente de <i>volatile acidity</i>."
    )

    # ------------------------------------------------------------------
    # 8. FASE 2: PREPARACIÓN Y PARTICIÓN DE LOS DATOS
    # ------------------------------------------------------------------
    add_h1("8. Fase 2: Preparación y partición de los datos", "sec_8")
    add_p(
        f"A partir de winequality-red-limpio.csv ({n_limpio:,} registros), se separaron las 11 "
        "variables independientes <b>X</b> y la variable objetivo <b>y</b> = <i>quality</i> "
        "(Particion_Datos.py). Posteriormente se aplicó una división aleatoria 80/20 mediante "
        "train_test_split(..., test_size=0.20, random_state=42, shuffle=True) (Pedregosa et al., "
        "2011). Esta estrategia aleatoria reproducible se adoptó porque el conjunto de datos "
        "carece de variable temporal o identificadores individuales y grupales; sin embargo, dicha "
        "ausencia de identificadores impide comprobar estadísticamente la independencia entre "
        "muestras (por ejemplo, si distintas filas proceden de un mismo productor o lote). "
        f"Al auditar explícitamente los vectores de características entre particiones, se "
        f"verificaron <b>{rev_x['indices_compartidos']} índices compartidos</b> y "
        f"<b>{rev_x['compartidos_train_test']} vectores X idénticos</b> compartidos entre "
        "entrenamiento y prueba (aun prescindiendo de <i>quality</i>)."
    )

    df_part = tablas["resumen_particion"]
    filas_part = [
        [
            str(f["Conjunto"]),
            str(int(f["Registros"])),
            f"{float(f['Porcentaje (%)']):.2f}%",
            str(f["Dimensiones X"]),
            str(f["Dimensiones y"]),
            str(int(f["Semilla aleatoria"])),
        ]
        for _, f in df_part.iterrows()
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Resumen de la partición aleatoria 80/20 en entrenamiento y prueba",
            ["Conjunto", "Registros", "Porcentaje (%)", "Dimensiones X", "Dimensiones y", "Semilla"],
            filas_part,
            [3.2, 2.2, 2.6, 3.2, 3.0, 2.0],
            ["L", "R", "R", "C", "C", "C"],
            estilos,
            nota_fuente="Fuente: resultados/tablas/resumen_particion.csv.",
        )
    )

    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "07_particion_datos.png",
            "Proporción de registros asignados a entrenamiento y prueba",
            "Elaboración propia a partir de resumen_particion.csv.",
            estilos,
            ancho_max_cm=14.8,
            alto_max_cm=7.8,
        )
    )
    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "08_distribucion_train_test.png",
            "Comparación porcentual de la distribución de calidad entre entrenamiento y prueba",
            "Elaboración propia sobre y_train.csv y y_test.csv.",
            estilos,
            ancho_max_cm=15.0,
            alto_max_cm=8.2,
        )
    )
    add_p(
        f"La Tabla 8 y las Figuras 7 y 8 confirman {n_train:,} registros de entrenamiento "
        f"({pct_train:.2f}%) y {n_test} de prueba ({pct_test:.2f}%), manteniendo proporciones "
        "similares en cada puntuación de calidad entre ambos subconjuntos."
    )

    add_p(
        "En Escalado_Datos.py se aplicó la estandarización z = (x − μ) / σ mediante StandardScaler. "
        "Es indispensable distinguir entre ambos métodos de scikit-learn: <b>fit_transform()</b> "
        "calcula la media y desviación estándar poblacional (ddof = 0) y transforma los datos, "
        "por lo que se ejecutó <b>exclusivamente sobre X_train</b>; en cambio, <b>transform()</b> "
        "aplica los parámetros ya aprendidos sin recalcular estadísticas y se empleó únicamente "
        "sobre X_test (scikit-learn developers, 2026). Nunca se ajustó el escalador sobre el "
        "conjunto completo ni sobre X_test, y la variable objetivo <i>quality</i> mantuvo su "
        "escala original, previniendo la fuga de datos. Asimismo, se aclara que la regresión "
        "lineal ordinaria no requiere estrictamente escalado para predecir, pero se implementa "
        "como requisito metodológico para expresar los coeficientes en desviaciones estándar."
    )

    df_esc = tablas["comparacion_escalado"]
    filas_esc = [
        [
            str(f["Variable"]),
            f"{float(f['Media antes (train)']):.4f}",
            f"{float(f['Desv. est. antes (train)']):.4f}",
            f"{abs(float(f['Media después (train)'])):.4f}",
            f"{float(f['Desv. est. después (train)']):.4f}",
        ]
        for _, f in df_esc.iterrows()
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Media y desviación estándar (ddof = 0) en X_train antes y después del escalado",
            ["Variable", "Media antes", "Desv. est. antes", "Media después", "Desv. est. después"],
            filas_esc,
            [4.2, 3.0, 3.1, 3.0, 3.1],
            ["L", "R", "R", "R", "R"],
            estilos,
            nota_fuente="Fuente: resultados/tablas/comparacion_escalado.csv.",
        )
    )

    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "11_comparacion_escalado.png",
            "Comparación de medias y desviaciones estándar en X_train antes y después del escalado",
            "Elaboración propia a partir de comparacion_escalado.csv.",
            estilos,
            ancho_max_cm=16.2,
            alto_max_cm=11.0,
        )
    )
    add_p(
        "La Tabla 9 y la Figura 9 verifican que las 11 características de entrenamiento pasan de "
        f"escalas dispares (por ejemplo, <i>density</i> con media {float(df_esc_idx.loc['density', 'Media antes (train)']):.4f} "
        f"frente a <i>total sulfur dioxide</i> con media {float(df_esc_idx.loc['total sulfur dioxide', 'Media antes (train)']):.4f} "
        f"y σ = {float(df_esc_idx.loc['total sulfur dioxide', 'Desv. est. antes (train)']):.4f}) "
        "a una escala uniforme con media ≈ 0.0000 y desviación estándar = 1.0000 (las gráficas "
        "complementarias de dispersión horizontal antes y después se incluyen en el Anexo D)."
    )

    # ------------------------------------------------------------------
    # 9. FASE 3: ENTRENAMIENTO DEL MODELO
    # ------------------------------------------------------------------
    add_h1("9. Fase 3: Entrenamiento del modelo", "sec_9")
    add_p(
        "En Entrenamiento_Modelo.py se seleccionó el algoritmo LinearRegression de scikit-learn "
        "(Pedregosa et al., 2011) por su interpretabilidad directa, trazabilidad matemática y "
        "capacidad para cuantificar la contribución lineal simultánea de las 11 variables "
        "fisicoquímicas. El modelo se entrenó mediante mínimos cuadrados ordinarios (OLS) "
        f"exclusivamente con X_train_escalado ({n_train:,} × 11) e y_train ({n_train:,}), sin "
        "utilizar información del conjunto de prueba."
    )
    add_p_formula(
        f"Al concluir el ajuste se obtuvo un intercepto b<sub>0</sub> = <b>{intercepto:.6f}</b>, "
        "que representa la calidad estimada cuando todas las variables estandarizadas toman "
        "el valor cero (es decir, en la media de entrenamiento). La ecuación obtenida, "
        "empleando la notación <b>z(variable)</b> para distinguir las 11 características "
        "estandarizadas de sus valores originales en bruto, es:",
        insumos["ecuacion_real"],
    )
    add_p(
        "Donde cada término z(x<sub>j</sub>) = (x<sub>j</sub> − μ<sub>j</sub>) / σ<sub>j</sub> "
        "denota la variable fisicoquímica estandarizada mediante los parámetros (μ<sub>j</sub>, "
        "σ<sub>j</sub>) ajustados por StandardScaler exclusivamente sobre X_train (Anexo A)."
    )

    df_coef = tablas["coeficientes_regresion"]
    filas_coef = [
        [
            str(idx + 1),
            str(f["Variable"]),
            f"{float(f['Coeficiente']):+.6f}",
            f"{float(f['Valor absoluto del coeficiente']):.6f}",
            str(f["Signo del coeficiente"]),
        ]
        for idx, (_, f) in enumerate(df_coef.iterrows())
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Coeficientes estandarizados de Regresión Lineal Múltiple por magnitud absoluta",
            ["Orden", "Variable", "Coeficiente", "Valor absoluto", "Signo"],
            filas_coef,
            [1.4, 4.6, 3.6, 3.6, 3.0],
            ["C", "L", "R", "R", "C"],
            estilos,
            nota_fuente="Fuente: resultados/tablas/coeficientes_regresion.csv.",
        )
    )

    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "12_coeficientes_regresion.png",
            "Coeficientes estandarizados positivos y negativos del modelo LinearRegression",
            "Elaboración propia a partir de coeficientes_regresion.csv.",
            estilos,
            ancho_max_cm=15.4,
            alto_max_cm=8.8,
        )
    )
    add_p(
        "Dado que las características están estandarizadas, cada coeficiente expresa el cambio "
        "estimado en <i>quality</i> ante el incremento de <b>una desviación estándar</b> en dicha "
        "variable manteniendo constantes las demás. Como muestran la Tabla 10 y la Figura 10, "
        f"<i>alcohol</i> ({float(df_coef_idx.loc['alcohol', 'Coeficiente']):+.4f}) y "
        f"<i>sulphates</i> ({float(df_coef_idx.loc['sulphates', 'Coeficiente']):+.4f}) ejercen "
        "los mayores efectos lineales positivos estimados, mientras que <i>volatile acidity</i> "
        f"({float(df_coef_idx.loc['volatile acidity', 'Coeficiente']):+.4f}), "
        f"<i>total sulfur dioxide</i> ({float(df_coef_idx.loc['total sulfur dioxide', 'Coeficiente']):+.4f}), "
        f"<i>pH</i> ({float(df_coef_idx.loc['pH', 'Coeficiente']):+.4f}) y <i>chlorides</i> "
        f"({float(df_coef_idx.loc['chlorides', 'Coeficiente']):+.4f}) presentan las asociaciones "
        "negativas más pronunciadas. Estos coeficientes <b>no demuestran causalidad</b> y la "
        "colinealidad entre predictores puede modificar el signo parcial respecto a la correlación "
        f"simple (por ejemplo, <i>citric acid</i> presentó r = "
        f"{float(df_corr_q.loc['citric acid', 'Correlación con quality']):+.4f} individualmente "
        f"frente a coeficiente {float(df_coef_idx.loc['citric acid', 'Coeficiente']):+.4f} en presencia "
        "de las demás variables)."
    )
    add_p(
        "Finalmente, el modelo ajustado (modelos/regresion_lineal.joblib), el escalador "
        "(modelos/standard_scaler.joblib) y los metadatos de orden y configuración "
        "(modelos/configuracion_modelo.json) se serializaron con joblib y se verificó su recarga "
        "exacta antes de iniciar la Fase 4."
    )

    # ------------------------------------------------------------------
    # 10. FASE 4 Y EXPERIMENTACIÓN COMPLEMENTARIA
    # ------------------------------------------------------------------
    add_h1("10. Fase 4: Evaluación del rendimiento y validación cruzada", "sec_10")
    add_h2("10.1 Evaluación del modelo principal en el conjunto de prueba", "sec_10_1")
    add_p(
        "En Evaluacion_Modelo.py se cargaron mediante joblib el modelo y el escalador guardados "
        f"junto con los <b>{n_test}</b> registros de prueba reservados (X_test_escalado, y_test), "
        "generando predicciones cuantitativas continuas sin redondeo, recorte ni categorización."
    )

    formulas_limpias_met = {
        "MSE": "MSE = (1/n) · ∑ (y<sub>i</sub> − ŷ<sub>i</sub>)<sup>2</sup>",
        "RMSE": "RMSE = √(MSE)",
        "MAE": "MAE = (1/n) · ∑ |y<sub>i</sub> − ŷ<sub>i</sub>|",
        "R²": "R² = 1 − [ ∑ (y<sub>i</sub> − ŷ<sub>i</sub>)<sup>2</sup> / ∑ (y<sub>i</sub> − ȳ)<sup>2</sup> ]",
    }
    descripciones_limpias_met = {
        "MSE": "Promedio de errores elevados al cuadrado; penaliza desviaciones grandes.",
        "RMSE": "Raíz del MSE; error cuadrático típico en la escala de quality.",
        "MAE": "Promedio de diferencias absolutas en puntos de calidad.",
        "R²": "Proporción de variabilidad de quality explicada (no es exactitud de clasificación).",
    }
    df_met = tablas["metricas_evaluacion"]
    filas_met = [
        [
            str(f["Métrica"]),
            f"{float(f['Valor']):.4f}",
            formulas_limpias_met.get(str(f["Métrica"]), str(f["Fórmula"])),
            descripciones_limpias_met.get(str(f["Métrica"]), str(f["Descripción"])),
        ]
        for _, f in df_met.iterrows()
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            f"Métricas reales de evaluación en el conjunto de prueba (n = {n_test})",
            ["Métrica", "Valor", "Fórmula", "Interpretación"],
            filas_met,
            [1.6, 1.8, 5.4, 7.6],
            ["C", "R", "L", "L"],
            estilos,
            nota_fuente="Fuente: resultados/tablas/metricas_evaluacion.csv.",
            tamano_fuente=8.6,
        )
    )
    add_p(
        f"En la Tabla 11 se observa que el modelo principal alcanzó <b>MSE = {m['MSE']:.4f}</b>, "
        f"<b>RMSE = {m['RMSE']:.4f}</b>, <b>MAE = {m['MAE']:.4f}</b> y <b>R² = {m['R²']:.4f}</b>. "
        f"El MAE indica una desviación absoluta media de {m['MAE']:.4f} puntos de calidad, "
        f"mientras que el RMSE ({m['RMSE']:.4f}, <b>raíz cuadrada aproximada</b> de "
        f"{m['MSE']:.4f} debido al redondeo a cuatro decimales sobre el valor completo "
        f"MSE = {m['MSE']:.6f}) refleja mayor sensibilidad a desviaciones extremas. El coeficiente "
        f"R² de {m['R²']:.4f} indica que el modelo explica el {m['R²']*100:.2f}% de la "
        f"variabilidad observada en prueba, dejando un {(1 - m['R²'])*100:.2f}% de variabilidad "
        "no explicada cuyo origen específico no puede atribuirse únicamente a partir del R²."
    )

    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "13_valores_reales_vs_predichos.png",
            "Comparación entre calidad real y calidad predicha en el conjunto de prueba",
            "Elaboración propia a partir de predicciones_modelo.csv.",
            estilos,
            ancho_max_cm=11.8,
            alto_max_cm=9.0,
        )
    )
    add_p(
        "La Figura 11 contrasta la calidad real frente a la predicha: las observaciones de notas "
        "5 y 6 se agrupan próximas a la diagonal y = x, mientras que las calidades reales bajas "
        "(3, 4) quedan sobreestimadas por encima de la diagonal y las altas (7, 8) quedan "
        f"subestimadas por debajo, concentrándose las predicciones entre {pred_min:.2f} y {pred_max:.2f}."
    )

    df_res = tablas["estadisticas_residuos"]
    filas_res = []
    for _, f in df_res.iterrows():
        nombre_est = str(f["Estadística"])
        val_num = float(f["Valor"])
        val_str = (
            f"{int(round(val_num))}"
            if "Cantidad" in nombre_est
            else f"{val_num:.4f}"
        )
        filas_res.append([nombre_est, val_str, str(f["Interpretación"])])

    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Estadísticas descriptivas de los residuos en prueba (Calidad_real − Calidad_predicha)",
            ["Estadística", "Valor", "Interpretación"],
            filas_res,
            [4.8, 2.0, 9.6],
            ["L", "R", "L"],
            estilos,
            nota_fuente="Fuente: resultados/tablas/estadisticas_residuos.csv.",
        )
    )

    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "15_grafica_residuos.png",
            "Dispersión de residuos frente a valores predichos en el conjunto de prueba",
            "Elaboración propia a partir de predicciones_modelo.csv.",
            estilos,
            ancho_max_cm=15.0,
            alto_max_cm=8.3,
        )
    )
    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "16_distribucion_residuos.png",
            "Histograma y densidad de los residuos en el conjunto de prueba",
            "Elaboración propia a partir de predicciones_modelo.csv.",
            estilos,
            ancho_max_cm=15.0,
            alto_max_cm=8.3,
        )
    )
    add_p(
        f"Según la Tabla 12, los residuos presentan media {res_stats['Media de los residuos']:+.4f}, "
        f"mediana {res_stats['Mediana de los residuos']:+.4f}, desviación estándar "
        f"{res_stats['Desviación estándar']:.4f} y rango de {res_stats['Residuo mínimo']:+.4f} "
        f"a {res_stats['Residuo máximo']:+.4f}, con {int(res_stats['Cantidad de residuos positivos'])} "
        f"subestimaciones y {int(res_stats['Cantidad de residuos negativos'])} sobreestimaciones. "
        "En la Figura 12 aparecen bandas diagonales paralelas porque <i>quality</i> es discreta "
        "entera mientras las predicciones son reales continuas. El histograma de la Figura 13 "
        "muestra mayor concentración en torno a cero sin permitir concluir normalidad únicamente "
        "por inspección visual."
    )

    df_tt = tablas["comparacion_train_test"]
    filas_tt = [
        [str(f["Métrica"]), f"{float(f['Entrenamiento']):.4f}", f"{float(f['Prueba']):.4f}"]
        for _, f in df_tt.iterrows()
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Comparación de métricas entre el conjunto de entrenamiento y el de prueba",
            ["Métrica", f"Entrenamiento (n = {n_train:,})", f"Prueba (n = {n_test})"],
            filas_tt,
            [4.4, 6.0, 6.0],
            ["C", "R", "R"],
            estilos,
            nota_fuente="Fuente: resultados/tablas/comparacion_train_test.csv.",
        )
    )

    df_base = tablas["comparacion_modelo_base"]
    filas_base = [
        [
            str(f["Métrica"]),
            f"{float(f['Modelo base (DummyRegressor)']):.4f}",
            f"{float(f['Regresión Lineal Múltiple']):.4f}",
            f"{float(f['Diferencia (RLM - Base)']):+.4f}",
        ]
        for _, f in df_base.iterrows()
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Comparación de Regresión Lineal Múltiple frente a DummyRegressor(strategy='mean')",
            ["Métrica", "DummyRegressor (media)", "Regresión Lineal Múltiple", "Diferencia (RLM − Base)"],
            filas_base,
            [2.4, 4.6, 4.8, 4.6],
            ["C", "R", "R", "R"],
            estilos,
            nota_fuente="Fuente: resultados/tablas/comparacion_modelo_base.csv.",
        )
    )

    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "18_comparacion_train_test.png",
            "Comparación gráfica de errores y R² entre entrenamiento y prueba",
            "Elaboración propia a partir de comparacion_train_test.csv.",
            estilos,
            ancho_max_cm=15.6,
            alto_max_cm=7.4,
        )
    )
    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "19_comparacion_modelo_base.png",
            "Comparación gráfica contra el modelo de referencia DummyRegressor",
            "Elaboración propia a partir de comparacion_modelo_base.csv.",
            estilos,
            ancho_max_cm=15.6,
            alto_max_cm=7.4,
        )
    )
    add_p(
        "Las Tablas 13 y 14, junto con las Figuras 14 y 15, muestran dos resultados principales en "
        "la partición 80/20 evaluada. En primer lugar, <b>no se observa una brecha pronunciada "
        f"entre entrenamiento (RMSE = {float(tt.loc['RMSE', 'Entrenamiento']):.4f}) y prueba "
        f"(RMSE = {float(tt.loc['RMSE', 'Prueba']):.4f})</b>, aunque el R² moderado "
        f"({float(tt.loc['R²', 'Entrenamiento']):.4f} y {float(tt.loc['R²', 'Prueba']):.4f}) "
        "<b>sugiere una capacidad predictiva limitada de la aproximación lineal, sin demostrar por "
        "sí solo la existencia de subajuste</b> ni descartar la influencia de variabilidad muestral "
        "en una partición única. En segundo lugar, el modelo supera en esta muestra al predictor base "
        f"DummyRegressor (R² = {float(mb.loc['R²', 'Modelo base (DummyRegressor)']):.4f}, "
        f"RMSE = {float(mb.loc['RMSE', 'Modelo base (DummyRegressor)']):.4f})."
    )

    df_cal = tablas["rendimiento_por_calidad"]
    filas_cal = [
        [
            str(int(f["Calidad_real"])),
            str(int(f["Cantidad_registros"])),
            f"{float(f['Promedio_predicciones']):.4f}",
            f"{float(f['MAE']):.4f}",
            f"{float(f['RMSE']):.4f}",
            f"{float(f['Residuo_promedio']):+.4f}",
        ]
        for _, f in df_cal.iterrows()
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Rendimiento en el conjunto de prueba desagregado por puntuación real de calidad",
            ["Calidad real", "Registros (n)", "Prom. predicho", "MAE", "RMSE", "Residuo prom."],
            filas_cal,
            [2.4, 2.5, 3.0, 2.7, 2.7, 3.1],
            ["C", "R", "R", "R", "R", "R"],
            estilos,
            nota_fuente="Fuente: resultados/tablas/rendimiento_por_calidad.csv.",
        )
    )

    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "20_error_por_calidad.png",
            "Error absoluto medio (MAE) y tamaño muestral por puntuación real de calidad",
            "Elaboración propia a partir de rendimiento_por_calidad.csv.",
            estilos,
            ancho_max_cm=15.0,
            alto_max_cm=8.3,
        )
    )
    add_p(
        f"La Tabla 15 y la Figura 16 muestran menores errores en las notas centrales "
        f"6 (MAE = {float(df_cal_idx.loc[6, 'MAE']):.4f}, n = {int(df_cal_idx.loc[6, 'Cantidad_registros'])}) "
        f"y 5 (MAE = {float(df_cal_idx.loc[5, 'MAE']):.4f}, n = {int(df_cal_idx.loc[5, 'Cantidad_registros'])}), "
        f"mientras que en las calidades extremas 3 (MAE = {float(df_cal_idx.loc[3, 'MAE']):.4f}, "
        f"n = {int(df_cal_idx.loc[3, 'Cantidad_registros'])}), 4 (MAE = {float(df_cal_idx.loc[4, 'MAE']):.4f}, "
        f"n = {int(df_cal_idx.loc[4, 'Cantidad_registros'])}) y 8 (MAE = {float(df_cal_idx.loc[8, 'MAE']):.4f}, "
        f"n = {int(df_cal_idx.loc[8, 'Cantidad_registros'])}) los errores aumentan con predicciones "
        "contraídas hacia el centro. Las métricas en calidades 3 y 8 deben interpretarse con cautela "
        "por el escaso número de observaciones en prueba."
    )

    # --- Subsección 10.2 Experimentación complementaria ---
    add_h2("10.2 Experimentación complementaria: Ridge con validación cruzada", "sec_10_2")
    add_p(
        "Atendiendo a la recomendación académica de evaluar al menos un ajuste de "
        "hiperparámetros sin alterar ni reentrenar el modelo principal, se implementó en "
        "Experimentacion_Complementaria.py una evaluación mediante <b>validación cruzada de "
        "cinco particiones</b> (KFold con n_splits=5, shuffle=True, random_state=42) aplicada "
        f"<b>exclusivamente sobre X_train sin escalar ({n_train:,} registros)</b> e y_train. "
        "El conjunto de prueba reservado (n = 272) nunca se utilizó para ajustar o seleccionar "
        "hiperparámetros."
    )
    add_p_formula(
        "Para garantizar que los parámetros de estandarización (μ<sub>j</sub>, σ<sub>j</sub>) "
        "se calculen de manera independiente en cada partición y prevenir fuga de datos interna, "
        "StandardScaler y el estimador se encapsularon dentro de un <b>Pipeline</b> de scikit-learn "
        "(Pedregosa et al., 2011). En la regresión Ridge, la función objetivo añade una penalización "
        "L2 controlada por el hiperparámetro alpha (α ≥ 0) sobre la magnitud cuadrática de los "
        "coeficientes estandarizados:",
        "min<sub>b</sub> [ ∑<sub>i</sub> (y<sub>i</sub> − ŷ<sub>i</sub>)<sup>2</sup> + "
        "α · ∑<sub>j=1..p</sub> b<sub>j</sub><sup>2</sup> ]",
    )

    filas_cv = [
        [
            str(f["Modelo"]),
            str(f["Alpha"]),
            f"{float(f['RMSE_promedio_CV']):.4f}",
            f"{float(f['RMSE_desv_est_CV']):.4f}",
            f"{float(f['R2_promedio_CV']):.4f}",
            f"{float(f['R2_desv_est_CV']):.4f}",
        ]
        for _, f in df_cv.iterrows()
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Validación cruzada 5-fold en X_train: LinearRegression vs. Ridge",
            ["Modelo", "Alpha", "RMSE prom. CV", "RMSE DE CV", "R² prom. CV", "R² DE CV"],
            filas_cv,
            [3.2, 2.2, 2.8, 2.7, 2.8, 2.7],
            ["L", "C", "R", "R", "R", "R"],
            estilos,
            nota_fuente=(
                f"Fuente: resultados/tablas/experimentacion_ridge_cv.csv (KFold 5-fold exclusivamente "
                f"sobre X_train sin escalar, n = {n_train:,}; métricas de validación cruzada distintas "
                f"de la evaluación final en prueba)."
            ),
        )
    )

    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "21_comparacion_ridge_cv.png",
            "Comparación RMSE y R² promedio ± DE en CV 5-fold entre Ridge y LinearRegression",
            "Elaboración propia a partir de experimentacion_ridge_cv.csv.",
            estilos,
            ancho_max_cm=15.8,
            alto_max_cm=7.8,
        )
    )
    add_p(
        f"La Tabla 16 y la Figura 17 distinguen claramente estas métricas de validación cruzada "
        f"(sobre particiones internas de entrenamiento) de la evaluación en prueba del modelo "
        f"principal (Tabla 11). Mientras LinearRegression obtuvo en CV 5-fold un RMSE promedio de "
        f"<b>{float(cv_ols['RMSE_promedio_CV']):.4f} ± {float(cv_ols['RMSE_desv_est_CV']):.4f}</b> "
        f"y R² promedio de <b>{float(cv_ols['R2_promedio_CV']):.4f} ± {float(cv_ols['R2_desv_est_CV']):.4f}</b>, "
        f"los valores de Ridge para alpha = 0.01 a 100 produjeron variaciones mínimas (hasta RMSE = "
        f"{float(mejor_ridge['RMSE_promedio_CV']):.4f} ± {float(mejor_ridge['RMSE_desv_est_CV']):.4f} "
        f"y R² = {float(mejor_ridge['R2_promedio_CV']):.4f} ± {float(mejor_ridge['R2_desv_est_CV']):.4f} "
        f"en alpha = {mejor_ridge['Alpha']}). Dado que la mejora es inferior a 0.001 puntos de RMSE "
        f"—más de 30 veces menor que la dispersión natural entre particiones (± {float(cv_ols['RMSE_desv_est_CV']):.4f})—, "
        "se justifica conservar LinearRegression como modelo principal sin sustituir los artefactos ni los "
        "resultados originales de la Fase 4."
    )

    # ------------------------------------------------------------------
    # 11. FUNCIONAMIENTO DEL PROGRAMA
    # ------------------------------------------------------------------
    add_h1("11. Funcionamiento del programa", "sec_11")
    add_p(
        "La solución está organizada en módulos Python independientes con responsabilidades "
        "separadas y coordinadas desde main.py mediante un menú interactivo en consola con "
        "opciones 0 a 22, rutas relativas respecto a la carpeta del proyecto (utilidades.py) y "
        "validación automática de vigencia entre fases."
    )
    filas_modulos = [
        ["utilidades.py", "Transversal", "Rutas relativas (pathlib), constantes, carga CSV, formato de consola, guardado a 300 DPI y control de vigencia."],
        ["Exploracion_Datos.py", "Fase 1", "Inspección inicial del CSV original: head(), tail(), shape, info(), nulos, duplicados, describe() y value_counts()."],
        ["Limpieza_Datos.py", "Fase 1", "Validación numérica y rango [0, 10], eliminación de 240 duplicados, exportación de winequality-red-limpio.csv."],
        ["Identificar_X_y_Y.py", "Fase 1", "Separación explícita de las 11 variables independientes X y la variable numérica dependiente y."],
        ["Analisis_Datos.py", "Fase 1", "Estadísticas describe().T, correlaciones de Pearson, detección IQR y gráficas 01 a 06."],
        ["Particion_Datos.py", "Fase 2", "Partición aleatoria 80/20 (random_state=42), verificación de índices disjuntos y gráficas 07 a 08."],
        ["Escalado_Datos.py", "Fase 2", "Ajuste exclusivo de StandardScaler con X_train, verificación ddof=0, exportación y gráficas 09 a 11."],
        ["Entrenamiento_Modelo.py", "Fase 3", "Entrenamiento de LinearRegression, tabla de coeficientes, ecuación, gráfica 12, JSON y joblib."],
        ["Evaluacion_Modelo.py", "Fase 4", "Carga del modelo, predicciones en prueba, métricas, residuos, DummyRegressor, gráficas 13 a 20 e informe."],
        ["Experimentacion_Complementaria.py", "Complementaria", "Validación cruzada 5-fold en X_train con Pipeline(StandardScaler) y Ridge (alpha 0.01 a 100); gráfica 21."],
        ["Generar_Reporte.py", "Fase 5", "Validación de insumos, generación automática de Markdown y compilación del PDF corregido en ReportLab."],
        ["main.py", "Orquestador", "Menú principal en consola (opciones 0 a 22), control de dependencias y ejecución completa sin bloqueos."],
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Arquitectura modular de los archivos Python del proyecto",
            ["Archivo", "Etapa", "Responsabilidad principal"],
            filas_modulos,
            [5.6, 2.7, 8.1],
            ["L", "C", "L"],
            estilos,
            nota_fuente="Fuente: Elaboración propia a partir de la inspección del código fuente del repositorio.",
            tamano_fuente=8.4,
            padding_h=3.0,
        )
    )

    dibujo_flujo = crear_diagrama_flujo_sistema(fuente_serif)
    p_pie_flujo = Paragraph(
        f"Figura {sig_figura()}. Diagrama de flujo general de las fases del sistema",
        estilos["PieFigura"],
    )
    p_pie_flujo._tof_info = (0, p_pie_flujo.text, f"figura_{num_figura}")
    p_fuente_flujo = Paragraph(
        "Fuente: Elaboración propia con base en main.py y utilidades.py.",
        estilos["FuenteFigura"],
    )
    story.append(KeepTogether([Spacer(1, 4), dibujo_flujo, p_pie_flujo, p_fuente_flujo]))

    add_h2("11.1 Instalación portable, ejecución y fragmentos clave", "sec_11_1")
    add_p(
        "Gracias al uso exclusivo de rutas relativas en utilidades.py, el proyecto puede "
        "instalarse y ejecutarse desde cualquier computadora ubicándose en la carpeta raíz del "
        "repositorio y creando o activando un entorno virtual local (.venv):"
    )
    story.append(
        _crear_bloque_codigo(
            "# Crear entorno virtual e instalar dependencias desde la raiz del proyecto\n"
            "python -m venv .venv\n"
            ".\\.venv\\Scripts\\Activate.ps1      # En Linux/macOS: source .venv/bin/activate\n"
            "pip install -r requirements.txt\n"
            "python main.py",
            estilos,
        )
    )
    story.append(Spacer(1, 5))
    add_p(
        "Los tres fragmentos siguientes ilustran los controles esenciales del flujo supervisado: "
        "(1) estandarización sin fuga de datos (Escalado_Datos.py), (2) ajuste del modelo por "
        "mínimos cuadrados ordinarios con LinearRegression().fit() (Entrenamiento_Modelo.py) y "
        "(3) evaluación continua en el conjunto de prueba reservado (Evaluacion_Modelo.py):"
    )
    story.append(
        _crear_bloque_codigo(
            "# Escalado_Datos.py: ajuste exclusivo con entrenamiento (sin fuga)\n"
            "escalador = StandardScaler()\n"
            "X_train_escalado = pd.DataFrame(\n"
            "    escalador.fit_transform(X_train),\n"
            "    columns=X_train.columns,\n"
            "    index=X_train.index,\n"
            ")\n"
            "X_test_escalado = pd.DataFrame(\n"
            "    escalador.transform(X_test),\n"
            "    columns=X_test.columns,\n"
            "    index=X_test.index,\n"
            ")",
            estilos,
        )
    )
    story.append(Spacer(1, 5))
    story.append(
        KeepTogether(
            [
                _crear_bloque_codigo(
                    "# Entrenamiento_Modelo.py: ajuste OLS exclusivo con X_train_escalado e y_train\n"
                    "modelo = LinearRegression()\n"
                    "modelo.fit(X_train_escalado, y_train)",
                    estilos,
                ),
                Spacer(1, 3),
                Paragraph(
                    "En este bloque, modelo.fit(X_train_escalado, y_train) calcula por mínimos "
                    "cuadrados ordinarios el intercepto (modelo.intercept_) y los 11 coeficientes "
                    "(modelo.coef_) sin utilizar en ningún momento el conjunto de prueba.",
                    estilos["Cuerpo"],
                ),
            ]
        )
    )
    story.append(
        _crear_bloque_codigo(
            "# Evaluacion_Modelo.py: prediccion continua y calculo de metricas en prueba\n"
            "y_pred = np.asarray(modelo.predict(X_test_escalado), dtype=float)\n"
            "mse = float(mean_squared_error(y_test, y_pred))\n"
            "rmse = float(np.sqrt(mse))\n"
            "mae = float(mean_absolute_error(y_test, y_pred))\n"
            "r2 = float(r2_score(y_test, y_pred))",
            estilos,
        )
    )
    story.append(Spacer(1, 5))

    # ------------------------------------------------------------------
    # 12. DISCUSIÓN DE RESULTADOS
    # ------------------------------------------------------------------
    add_h1("12. Discusión de resultados", "sec_12")
    add_p(
        "La distinción rigurosa entre hechos observados y posibles explicaciones permite valorar "
        "objetivamente el alcance del modelo obtenido. Como <b>resultados observados</b>, la "
        f"Regresión Lineal Múltiple alcanzó en prueba MSE = {m['MSE']:.4f}, RMSE = {m['RMSE']:.4f}, "
        f"MAE = {m['MAE']:.4f} y R² = {m['R²']:.4f}, superando en esta partición al modelo base "
        f"DummyRegressor (MAE = {float(mb.loc['MAE', 'Modelo base (DummyRegressor)']):.4f}, "
        f"RMSE = {float(mb.loc['RMSE', 'Modelo base (DummyRegressor)']):.4f}, "
        f"R² = {float(mb.loc['R²', 'Modelo base (DummyRegressor)']):.4f}) y sin presentar una "
        f"brecha pronunciada respecto a entrenamiento (RMSE = {float(tt.loc['RMSE', 'Entrenamiento']):.4f}, "
        f"R² = {float(tt.loc['R²', 'Entrenamiento']):.4f}). Además, las predicciones mostraron "
        f"errores inferiores en las notas 5 y 6 (MAE de {float(df_cal_idx.loc[5, 'MAE']):.2f} y "
        f"{float(df_cal_idx.loc[6, 'MAE']):.2f}) que en las calidades 3, 4 y 8 (MAE entre "
        f"{float(df_cal_idx.loc[4, 'MAE']):.2f} y {float(df_cal_idx.loc[3, 'MAE']):.2f}), mientras "
        f"que los residuos tuvieron media {res_stats['Media de los residuos']:+.4f} y bandas "
        "diagonales debidas al carácter discreto de <i>quality</i>."
    )
    add_p(
        "Como <b>posibles interpretaciones</b> destacan cuatro aspectos acotados al conjunto de "
        "datos evaluado y a muestras similares de vino tinto <i>Vinho Verde</i>. Primero, la "
        f"concentración del {pct_56_limp:.2f}% de las observaciones limpias en las notas 5 y 6 "
        "orienta el ajuste por mínimos cuadrados ordinarios hacia el centro de la escala, "
        "reduciendo el error promedio en calidades intermedias pero alejando las predicciones de "
        "los extremos. Segundo, aunque <i>alcohol</i> y <i>volatile acidity</i> exhiben asociaciones "
        "lineales consistentes con la literatura (Cortez et al., 2009a), el R² moderado sugiere "
        "capacidad predictiva limitada del plano lineal, sin demostrar por sí solo subajuste ni "
        "identificar las causas exactas de la variación restante. Tercero, la estabilidad en la "
        "partición única 80/20 se complementó con validación cruzada 5-fold sobre entrenamiento, "
        f"revelando una dispersión esperada entre pliegues de ± {float(cv_ols['RMSE_desv_est_CV']):.4f} "
        "en RMSE y mejoras marginales con Ridge que respaldan conservar LinearRegression. Por último, "
        "la colinealidad entre variables de acidez y densidad explica atenuaciones e inversiones de "
        "signo en los coeficientes parciales."
    )

    # ------------------------------------------------------------------
    # 13. LIMITACIONES Y TRABAJOS FUTUROS
    # ------------------------------------------------------------------
    add_h1("13. Limitaciones y trabajos futuros", "sec_13")
    add_h2("13.1 Limitaciones observadas y potenciales", "sec_13_1")
    for lim in [
        f"<b>Desbalance en la variable objetivo (observada):</b> El {pct_56_test:.2f}% del conjunto de prueba pertenece a calidades 5 y 6, mientras que los extremos 3 (n = {int(df_cal_idx.loc[3, 'Cantidad_registros'])}) y 8 (n = {int(df_cal_idx.loc[8, 'Cantidad_registros'])}) tienen muy pocas observaciones y no existen muestras con notas 0, 1, 2, 9 o 10.",
        f"<b>Variabilidad no explicada y capacidad lineal limitada (observada):</b> Un R² de {m['R²']:.4f} indica que el {(1 - m['R²'])*100:.2f}% de la variabilidad observada en prueba corresponde a variación no explicada por el modelo lineal, sin que sea posible atribuirla directamente a causas específicas únicamente a partir de esta métrica.",
        f"<b>Ausencia de identificadores y partición única (metodológica):</b> Aunque se comprobaron {rev_x['compartidos_train_test']} vectores X idénticos compartidos entre entrenamiento y prueba, la falta de identificadores de muestra, lote o productor impide verificar independencia estadística entre registros y una partición única 80/20 no elimina la incertidumbre muestral.",
        "<b>Subjetividad de la etiqueta sensorial (potencial):</b> La puntuación discreta <i>quality</i> proviene de la mediana de apreciaciones humanas de catadores expertos (Cortez et al., 2009a).",
        "<b>Multicolinealidad y naturaleza discreta (observada):</b> Existen correlaciones elevadas entre predictores (como <i>fixed acidity</i> con <i>pH</i> y <i>density</i>), y el carácter entero de <i>quality</i> produce bandas diagonales en los residuos.",
        "<b>Alcance muestral restringido (potencial):</b> El análisis corresponde a vino tinto portugués <i>Vinho Verde</i> del conjunto evaluado; los resultados no deben generalizarse automáticamente a otras poblaciones, variedades o vinos blancos.",
    ]:
        add_p(f"• {lim}", "Lista")

    add_h2("13.2 Trabajos futuros propuestos", "sec_13_2")
    add_p(
        "Dado que ya se realizó una experimentación complementaria con regularización Ridge "
        "y validación cruzada de 5 particiones sobre entrenamiento, se proponen como líneas "
        "futuras: explorar regularización Lasso o ElasticNet para selección automática de "
        "variables ante la colinealidad observada; evaluar términos polinómicos e interacciones "
        "químicas mediante validación cruzada repetida o anidada; probar regresión ordinal acorde "
        "con la naturaleza discreta de <i>quality</i>; y comparar algoritmos no lineales como "
        "Random Forest o Gradient Boosting (James et al., 2023)."
    )

    # ------------------------------------------------------------------
    # 14. CONCLUSIONES
    # ------------------------------------------------------------------
    add_h1("14. Conclusiones", "sec_14")
    add_p(
        "Se completaron las cinco fases del proyecto junto con la experimentación complementaria "
        "solicitada, construyendo una arquitectura modular reproducible y portable en Python que "
        "abarca exploración, eliminación de duplicados exactos, partición 80/20 sin índices ni "
        "vectores X idénticos compartidos, estandarización exclusiva en entrenamiento, ajuste "
        "mediante LinearRegression, evaluación en el conjunto de prueba reservado, validación "
        "cruzada con Ridge y generación automática de documentación técnica."
    )
    add_p(
        f"De acuerdo con la evidencia medida en la partición evaluada (MSE = {m['MSE']:.4f}, "
        f"RMSE = {m['RMSE']:.4f}, MAE = {m['MAE']:.4f}, R² = {m['R²']:.4f}), el modelo de "
        "Regresión Lineal Múltiple supera al predictor base de media constante DummyRegressor y no "
        "presenta una brecha pronunciada entre entrenamiento y prueba. Asimismo, la validación "
        "cruzada 5-fold mostró desempeño prácticamente equivalente entre LinearRegression y Ridge, "
        "respaldando conservar el modelo OLS como línea base interpretable. No obstante, su "
        "capacidad explicativa moderada y los errores elevados en las calidades extremas "
        "indican que solo debe emplearse como herramienta orientativa en datos similares al "
        "conjunto estudiado, y nunca como predictor preciso para puntuaciones extremas o "
        "poblaciones vitivinícolas distintas."
    )

    # ------------------------------------------------------------------
    # 15. REFERENCIAS BIBLIOGRÁFICAS (APA 7, ALFABÉTICA, SANGRÍA FRANCESA, IZQ.)
    # ------------------------------------------------------------------
    add_h1("15. Referencias bibliográficas", "sec_15")
    for ref in REFERENCIAS_APA7:
        story.append(KeepTogether([Paragraph(ref["texto"], estilos["ReferenciaAPA"])]))

    # ------------------------------------------------------------------
    # 16. ANEXOS (CON ANEXO C EN PÁGINA HORIZONTAL)
    # ------------------------------------------------------------------
    story.append(PageBreak())
    add_h1("16. Anexos", "sec_16")

    add_h2("Anexo A. Parámetros aprendidos por StandardScaler", "sec_16_a")
    add_p(
        "La Tabla 18 detalla las medias (mean_), desviaciones estándar (scale_) y varianzas "
        "(var_) exactas aprendidas exclusivamente del conjunto de entrenamiento (X_train) y "
        "almacenadas en modelos/standard_scaler.joblib."
    )
    df_param = tablas["parametros_escalador"]
    filas_param = [
        [
            str(f["Variable"]),
            f"{float(f['Media aprendida (mean_)']):.6f}",
            f"{float(f['Desv. est. aprendida (scale_)']):.6f}",
            f"{float(f['Varianza aprendida (var_)']):.6f}",
        ]
        for _, f in df_param.iterrows()
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Parámetros internos aprendidos por StandardScaler en entrenamiento",
            ["Variable", "Media aprendida (mean_)", "Desv. est. (scale_)", "Varianza (var_)"],
            filas_param,
            [4.6, 3.9, 3.9, 4.0],
            ["L", "R", "R", "R"],
            estilos,
            nota_fuente="Fuente: resultados/tablas/parametros_escalador.csv.",
        )
    )

    add_h2("Anexo B. Muestra de predicciones reales en el conjunto de prueba", "sec_16_b")
    df_pred = df_pred_all.head(15)
    filas_pred = [
        [
            str(int(f["Registro"])),
            str(int(f["Calidad_real"])),
            f"{float(f['Calidad_predicha']):.4f}",
            f"{float(f['Residuo']):+.4f}",
            f"{float(f['Error_absoluto']):.4f}",
            f"{float(f['Error_cuadratico']):.4f}",
        ]
        for _, f in df_pred.iterrows()
    ]
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Primeros 15 registros de predicciones obtenidas en el conjunto de prueba",
            ["Registro", "Calidad real", "Calidad predicha", "Residuo", "Error absoluto", "Error cuadrático"],
            filas_pred,
            [2.0, 2.4, 3.0, 2.9, 3.0, 3.1],
            ["C", "C", "R", "R", "R", "R"],
            estilos,
            nota_fuente=f"Fuente: Primeras 15 observaciones de resultados/tablas/predicciones_modelo.csv ({n_test} registros totales).",
        )
    )

    # Cambiar a página horizontal para presentar la matriz de correlación legible a 8.5 pt
    story.append(NextPageTemplate("ContenidoHorizontal"))
    story.append(PageBreak())
    add_h2("Anexo C. Matriz completa de correlación de Pearson", "sec_16_c")
    df_mcorr = tablas["matriz_correlacion"]
    cols_mcorr = list(df_mcorr.columns)
    filas_mcorr = []
    for _, fila in df_mcorr.iterrows():
        fila_fmt = [str(fila.iloc[0])] + [f"{float(v):.2f}" for v in fila.iloc[1:]]
        filas_mcorr.append(fila_fmt)
    story.extend(
        _crear_tabla_academica(
            sig_tabla(),
            "Matriz numérica completa de correlación de Pearson (dataset limpio)",
            cols_mcorr,
            filas_mcorr,
            [3.54] + [1.70] * (len(cols_mcorr) - 1),  # 23.94 cm en Carta horizontal
            ["L"] + ["R"] * (len(cols_mcorr) - 1),
            estilos,
            nota_fuente="Fuente: resultados/tablas/matriz_correlacion.csv.",
            tamano_fuente=8.5,
            padding_h=3.0,
        )
    )
    story.append(NextPageTemplate("Contenido"))
    story.append(PageBreak())

    add_h2("Anexo D. Gráficas complementarias de escalado y evaluación", "sec_16_d")
    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "09_variables_sin_escalar.png",
            "Distribución horizontal de las variables de entrenamiento antes del escalado",
            "Elaboración propia sobre X_train.csv.",
            estilos,
            ancho_max_cm=15.6,
            alto_max_cm=8.6,
        )
    )
    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "10_variables_escaladas.png",
            "Distribución horizontal de las variables de entrenamiento después de StandardScaler",
            "Elaboración propia sobre X_train_escalado.csv.",
            estilos,
            ancho_max_cm=15.6,
            alto_max_cm=8.6,
        )
    )
    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "14_comparacion_predicciones.png",
            "Comparación línea a línea de los primeros 30 registros de prueba (orden no temporal)",
            "Elaboración propia sobre predicciones_modelo.csv.",
            estilos,
            ancho_max_cm=15.6,
            alto_max_cm=8.0,
        )
    )
    story.append(
        _crear_figura_academica(
            sig_figura(),
            RUTA_GRAFICAS / "17_boxplot_residuos.png",
            "Diagrama de caja de los residuos en el conjunto de prueba",
            "Elaboración propia sobre predicciones_modelo.csv.",
            estilos,
            ancho_max_cm=15.2,
            alto_max_cm=7.0,
        )
    )

    add_h2("Anexo E. Estructura de carpetas y verificación de artefactos", "sec_16_e")
    story.append(
        _crear_bloque_codigo(
            "ProyectoIA/\n"
            "├── main.py, utilidades.py, requirements.txt, README.md\n"
            "├── Exploracion_Datos.py, Limpieza_Datos.py, Identificar_X_y_Y.py\n"
            "├── Analisis_Datos.py, Particion_Datos.py, Escalado_Datos.py\n"
            "├── Entrenamiento_Modelo.py, Evaluacion_Modelo.py\n"
            "├── Experimentacion_Complementaria.py, Generar_Reporte.py\n"
            "├── winequality-red.csv, winequality-red-limpio.csv\n"
            "├── logos_institucionales.png\n"
            "├── modelos/\n"
            "│   ├── regresion_lineal.joblib, standard_scaler.joblib\n"
            "│   └── configuracion_modelo.json\n"
            "├── documentacion/\n"
            "│   ├── respaldo/  (copias originales del PDF y Markdown)\n"
            "│   ├── configuracion_reporte.json\n"
            "│   ├── Reporte_Tecnico_ProyectoIA.pdf  (respaldo original)\n"
            "│   ├── Reporte_Tecnico_ProyectoIA_Corregido.pdf  (versión final)\n"
            "│   └── Reporte_Tecnico_ProyectoIA.md\n"
            "└── resultados/\n"
            "    ├── datos_procesados/ (6 archivos CSV de entrenamiento y prueba)\n"
            "    ├── graficas/         (21 figuras PNG a 300 DPI)\n"
            "    ├── tablas/           (17 tablas estadísticas, evaluación y CV)\n"
            "    └── informes/         (resumen_evaluacion.md)",
            estilos,
        )
    )

    # Verificar correspondencia bidireccional entre citas y referencias
    validar_citas_y_referencias("\n".join(texto_para_citas))

    return story, num_tabla, num_figura, textos_base["resumen"]


def _df_a_markdown(df: pd.DataFrame, decimales: int = 4) -> str:
    """Convierte un DataFrame a tabla Markdown limpia con formato uniforme."""
    encabezados = [str(c) for c in df.columns]
    lineas = [
        "| " + " | ".join(encabezados) + " |",
        "| " + " | ".join(["---"] * len(encabezados)) + " |",
    ]
    for _, fila in df.iterrows():
        celdas = []
        for val in fila:
            if isinstance(val, float) and not float(val).is_integer():
                celdas.append(f"{val:.{decimales}f}")
            elif isinstance(val, float) and float(val).is_integer():
                celdas.append(str(int(val)))
            else:
                celdas.append(str(val))
        lineas.append("| " + " | ".join(celdas) + " |")
    return "\n".join(lineas)


def generar_markdown_reporte(insumos: dict, ruta_md: Path = RUTA_REPORTE_MD) -> Path:
    """Genera la versión editable en Markdown del reporte técnico corregido."""
    asegurar_carpeta_documentacion()
    cfg_r = insumos["config_reporte"]
    cfg_m = insumos["config_modelo"]
    tablas = insumos["tablas"]
    m = insumos["metricas"]
    tt = insumos["train_test"]
    mb = insumos["modelo_base"]
    res_stats = insumos["residuos_stats"]
    rev_x = insumos["revision_x"]
    iqr_ext = insumos["iqr_extremos"]
    ver = insumos["versiones"]
    textos_base = _generar_contenido_compartido(insumos)

    tab_limp = tablas["comparacion_limpieza"]
    n_orig = int(tab_limp.loc[tab_limp["Característica"] == "Registros", "Antes"].iloc[0])
    n_train = int(cfg_m["registros_entrenamiento"])
    n_test = int(cfg_m["registros_prueba"])
    n_limpio = n_train + n_test
    intercepto = float(cfg_m["intercepto"])
    df_cv = tablas["experimentacion_ridge_cv"]
    cv_ols = df_cv[df_cv["Modelo"] == "LinearRegression"].iloc[0]
    df_cal_idx = tablas["rendimiento_por_calidad"].set_index("Calidad_real")
    pct_56_test = (
        (
            int(df_cal_idx.loc[5, "Cantidad_registros"])
            + int(df_cal_idx.loc[6, "Cantidad_registros"])
        )
        / n_test
    ) * 100

    integrantes_md = "\n".join(f"- {nombre}" for nombre in cfg_r["integrantes"])
    refs_md = "\n\n".join(ref["md"] for ref in REFERENCIAS_APA7)
    grupo_md = (
        f"\n**Grupo:** {cfg_r['grupo'].strip()}\n"
        if cfg_r.get("grupo", "").strip()
        else ""
    )

    contenido = f"""# {cfg_r['titulo']}

**Universidad:** {cfg_r['universidad']}  
**Facultad:** {cfg_r['facultad']}  
**Carrera:** {cfg_r['carrera']}  
**Materia:** {cfg_r['materia']}  
**Tipo de trabajo:** {cfg_r['tipo_trabajo']}  
{grupo_md}
**Integrantes:**
{integrantes_md}

**Profesor:** {cfg_r['profesor']}  
**Lugar y fecha:** {cfg_r['lugar']} — {cfg_r['fecha']}

---

## Resumen

{textos_base['resumen']}

**Palabras clave:** {textos_base['palabras_clave']}

---

## 1. Introducción

La Inteligencia Artificial (IA) comprende el diseño de sistemas capaces de ejecutar inferencias y resolver problemas complejos a partir de datos (Russell & Norvig, 2021). El aprendizaje automático ajusta modelos matemáticos desde observaciones históricas, y dentro del aprendizaje supervisado la regresión resulta adecuada para estimar variables continuas u ordinales cuantitativas graduadas (James et al., 2023). En el contexto vitivinícola, modelar la relación entre mediciones fisicoquímicas y calidad sensorial apoya el control objetivo en laboratorio (Cortez et al., 2009a).

## 2. Planteamiento del problema

Se plantea estimar la puntuación de calidad del vino tinto (`quality`) a partir de 11 variables fisicoquímicas. Se eligió regresión en lugar de clasificación porque `quality` es una **puntuación numérica discreta de naturaleza ordinal** (escala 0–10, observada en enteros de 3 a 8) tratada como **respuesta cuantitativa**, cuyas distancias ordinales se perderían al discretizar en clases.

## 3. Objetivos

- **General:** Desarrollar, entrenar y evaluar críticamente un modelo de Regresión Lineal Múltiple para predecir la calidad del vino tinto a partir de sus propiedades fisicoquímicas.
- **Específicos:** Explorar el dataset original; limpiar duplicados exactos conservando observaciones atípicas ante la ausencia de evidencia de error; preparar partición 80/20 con `StandardScaler` sin fuga de datos; entrenar `LinearRegression`; evaluar con `MSE`, `RMSE`, `MAE`, `R²`, residuos y modelo base; realizar un análisis complementario con `Ridge` y validación cruzada; e interpretar limitaciones y alcances dentro de la evidencia medida.

## 4. Marco teórico

Incluye fundamentos de Inteligencia Artificial (Russell & Norvig, 2021), aprendizaje automático y supervisado (James et al., 2023), diferenciación entre clasificación y regresión, regresión lineal simple ($\\hat{{y}} = \\beta_0 + \\beta_1 x$), regresión lineal múltiple ($\\hat{{y}} = b_0 + \\sum_{{j=1}}^p b_j x_j$), mínimos cuadrados ordinarios ($\\min_b \\sum (y_i - \\hat{{y}}_i)^2$), preprocesamiento (McKinney, 2010), estandarización $z_j = (x_j - \\mu_j)/\\sigma_j$ (scikit-learn developers, 2026), fuga de datos, sobreajuste/subajuste y métricas de regresión (definiendo `MSE` como el promedio de los errores elevados al cuadrado):

$$\\text{{MSE}} = \\frac{{1}}{{n}}\\sum (y_i - \\hat{{y}}_i)^2, \\quad \\text{{RMSE}} = \\sqrt{{\\text{{MSE}}}}, \\quad \\text{{MAE}} = \\frac{{1}}{{n}}\\sum |y_i - \\hat{{y}}_i|, \\quad R^2 = 1 - \\frac{{\\sum (y_i - \\hat{{y}}_i)^2}}{{\\sum (y_i - \\bar{{y}})^2}}$$

## 5. Herramientas y tecnologías

Se emplearon Python `{ver['python']}`, pandas (`{ver['pandas']}`), NumPy (`{ver['numpy']}`), Matplotlib (`{ver['matplotlib']}`), Seaborn (`{ver['seaborn']}`), Scikit-learn (`{ver['scikit-learn']}`), Joblib (`{ver['joblib']}`), ReportLab (`{ver['reportlab']}`), Pillow (`{ver['pillow']}`), PyCharm y Antigravity (Harris et al., 2020; Hunter, 2007; pandas development team, 2026; Pedregosa et al., 2011; Waskom, 2021).

## 6. Descripción del conjunto de datos

Dataset *Wine Quality (Red Wine)* del UCI Machine Learning Repository (Cortez et al., 2009b), enlace https://archive.ics.uci.edu/dataset/186/wine+quality, DOI https://doi.org/10.24432/C56S3T. Contiene {n_orig:,} registros originales y 12 columnas numéricas (11 variables independientes `X` y la variable objetivo numérica discreta de naturaleza ordinal `y = quality`).

## 7. Fase 1: Exploración, limpieza y análisis de datos

Se eliminaron 240 duplicados exactos obteniendo {n_limpio:,} registros únicos y 0 valores faltantes. Las observaciones detectadas por el método IQR (por debajo de `Q1 - 1.5*IQR` o por encima de `Q3 + 1.5*IQR`, incluyendo valores en ambos extremos en `density`: {int(iqr_ext['density']['inferior'])} inf. / {int(iqr_ext['density']['superior'])} sup., `pH`: {int(iqr_ext['pH']['inferior'])} inf. / {int(iqr_ext['pH']['superior'])} sup. y `chlorides`: {int(iqr_ext['chlorides']['inferior'])} inf. / {int(iqr_ext['chlorides']['superior'])} sup.) se conservaron debido a la ausencia de evidencia suficiente para considerarlas errores.

{_df_a_markdown(tablas['comparacion_limpieza'], decimales=0)}

{_df_a_markdown(tablas['estadisticas_descriptivas'])}

{_df_a_markdown(tablas['distribucion_calidad'], decimales=2)}

{_df_a_markdown(tablas['correlaciones_con_quality'])}

{_df_a_markdown(tablas['valores_atipicos_iqr'])}

Gráficas asociadas: `../resultados/graficas/01_distribucion_calidad.png` a `06_dispersion_calidad.png`.

## 8. Fase 2: Preparación y partición de los datos

División aleatoria 80/20 con `random_state=42` (`{n_train:,}` entrenamiento / `{n_test}` prueba) por ausencia de variable temporal o identificadores individuales/grupales (sin comprobar independencia estadística entre muestras, aunque se verificaron `{rev_x['indices_compartidos']}` índices compartidos y `{rev_x['compartidos_train_test']}` vectores `X` idénticos compartidos entre entrenamiento y prueba) y estandarización con `StandardScaler` (`fit_transform()` solo en `X_train`, `transform()` en `X_test`).

{_df_a_markdown(tablas['resumen_particion'], decimales=2)}

{_df_a_markdown(tablas['comparacion_escalado'])}

## 9. Fase 3: Entrenamiento del modelo

Modelo `LinearRegression` entrenado solo con `X_train_escalado` e `y_train`. Intercepto: `{intercepto:.6f}`. La notación `z(variable)` indica cada predictor estandarizado:

```text
{insumos['ecuacion_real']}
```

{_df_a_markdown(tablas['coeficientes_regresion'])}

## 10. Fase 4: Evaluación del rendimiento y validación cruzada

### 10.1 Evaluación del modelo principal en prueba (`n = {n_test}`)

`RMSE = {m['RMSE']:.4f}` corresponde a la raíz cuadrada aproximada de `MSE = {m['MSE']:.4f}` debido al redondeo a cuatro decimales (valor completo `{m['MSE']:.6f}`). El `R² = {m['R²']:.4f}` deja `{(1 - m['R²'])*100:.2f}%` de variabilidad no explicada sin causas atribuibles directamente desde esta métrica.

{_df_a_markdown(tablas['metricas_evaluacion'])}

{_df_a_markdown(tablas['estadisticas_residuos'])}

{_df_a_markdown(tablas['comparacion_train_test'])}

{_df_a_markdown(tablas['comparacion_modelo_base'])}

{_df_a_markdown(tablas['rendimiento_por_calidad'])}

### 10.2 Experimentación complementaria: Ridge con validación cruzada (`X_train`, 5-fold)

Evaluación complementaria exclusiva sobre `X_train` sin escalar (`{n_train:,}` registros) mediante `Pipeline(StandardScaler -> Modelo)` y `KFold(n_splits=5, shuffle=True, random_state=42)`, sin tocar el conjunto de prueba ni sustituir `LinearRegression`:

{_df_a_markdown(df_cv)}

Gráfica asociada: `../resultados/graficas/21_comparacion_ridge_cv.png`.

## 11. Funcionamiento del programa

Ejecución portable en cualquier computadora mediante rutas relativas (`utilidades.py`):

```bash
python -m venv .venv
# Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1  |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Fragmento clave del entrenamiento (`Entrenamiento_Modelo.py`):

```python
modelo = LinearRegression()
modelo.fit(X_train_escalado, y_train)
```

```mermaid
flowchart TD
    A[winequality-red.csv] --> B[Fase 1: Exploracion y Limpieza]
    B --> C[winequality-red-limpio.csv]
    C --> D[Fase 2: Particion 80/20 y StandardScaler]
    D --> E[Fase 3: LinearRegression + joblib]
    E --> F[Fase 4: Evaluacion en Prueba y Ridge CV en X_train]
    F --> G[Fase 5: Reporte PDF Corregido y Markdown]
```

## 12. Discusión de resultados

En la partición 80/20 evaluada (`MAE = {m['MAE']:.4f}`, `RMSE = {m['RMSE']:.4f}`, `R² = {m['R²']:.4f}`) se supera a `DummyRegressor` (`RMSE = {float(mb.loc['RMSE', 'Modelo base (DummyRegressor)']):.4f}`, `R² = {float(mb.loc['R²', 'Modelo base (DummyRegressor)']):.4f}`) sin observarse una brecha pronunciada frente a entrenamiento (`R² = {float(tt.loc['R²', 'Entrenamiento']):.4f}`). El `R²` moderado sugiere capacidad predictiva lineal limitada sin demostrar por sí solo subajuste, mientras que en validación cruzada (`X_train`) `LinearRegression` (`RMSE = {float(cv_ols['RMSE_promedio_CV']):.4f} ± {float(cv_ols['RMSE_desv_est_CV']):.4f}`) y `Ridge` obtienen resultados casi idénticos.

## 13. Limitaciones y trabajos futuros

Concentración del `{pct_56_test:.2f}%` de los registros de prueba en notas 5 y 6, escasez en notas extremas (3 y 8), ausencia de identificadores para probar independencia muestral (`{rev_x['compartidos_train_test']}` vectores `X` idénticos compartidos), `{(1 - m['R²'])*100:.2f}%` de variabilidad no explicada, subjetividad sensorial, multicolinealidad y alcance restringido a datos similares al vino tinto *Vinho Verde* evaluado. Trabajos futuros: Lasso/ElasticNet, términos polinómicos/interacciones, regresión ordinal y Random Forest / Gradient Boosting.

## 14. Conclusiones

La evidencia medida en la partición evaluada y la validación cruzada 5-fold muestra un comportamiento estable de `LinearRegression` por encima del promedio constante, pero con capacidad predictiva moderada y errores altos en calidades extremas, por lo que su utilidad se restringe a línea base orientativa en muestras comparables al conjunto estudiado.

## 15. Referencias bibliográficas

{refs_md}

## 16. Anexos

### Anexo A. Parámetros aprendidos por StandardScaler

{_df_a_markdown(tablas['parametros_escalador'], decimales=6)}

### Anexo B. Primeros 15 registros de predicciones reales en prueba

{_df_a_markdown(tablas['predicciones_modelo'].head(15))}

### Anexo C. Matriz de correlación de Pearson

{_df_a_markdown(tablas['matriz_correlacion'], decimales=2)}
"""

    with open(ruta_md, "w", encoding="utf-8") as archivo:
        archivo.write(contenido)
    return ruta_md


def generar_reporte_tecnico(
    ruta_pdf: Path = RUTA_REPORTE_PDF_CORREGIDO,
) -> dict:
    """Valida insumos, genera el Markdown editable y compila el reporte PDF corregido."""
    imprimir_encabezado("FASE 5: GENERACION DEL REPORTE TECNICO CORREGIDO EN PDF")
    verificar_insumos_reporte()
    asegurar_carpeta_documentacion()

    insumos = cargar_insumos_proyecto()
    ruta_md = generar_markdown_reporte(insumos, RUTA_REPORTE_MD)
    print(f"Version editable Markdown guardada en: {ruta_md}")

    fuente_serif, fuente_mono = registrar_fuentes_academicas()
    estilos = crear_estilos(fuente_serif, fuente_mono)
    story, total_tablas, total_figuras, _ = _construir_story_pdf(
        insumos, estilos, fuente_serif
    )

    doc = ReporteDocTemplate(
        str(ruta_pdf),
        fuente_serif=fuente_serif,
        config_reporte=insumos["config_reporte"],
        pagesize=letter,
        title=insumos["config_reporte"]["titulo"],
        author=", ".join(insumos["config_reporte"]["integrantes"]),
        subject="Proyecto Final de Inteligencia Artificial — Regresión Lineal Múltiple",
    )
    doc.multiBuild(story)

    # Comprobación de portada única y ausencia de páginas vacías
    pagina_resumen = doc.paginas_secciones.get("Resumen")
    if pagina_resumen != 2:
        raise ValueError(
            f"La portada debe ocupar exactamente 1 página, pero Resumen inició en la página {pagina_resumen}."
        )

    paginas_vacias = [
        p
        for p in range(1, doc.total_paginas + 1)
        if doc.conteo_flowables_pagina.get(p, 0) == 0
    ]
    if paginas_vacias:
        raise ValueError(
            f"Se detectaron páginas vacías inesperadas en el PDF: {paginas_vacias}."
        )

    print(f"Reporte tecnico PDF generado en:      {ruta_pdf}")
    print(f"Total de paginas del documento:       {doc.total_paginas}")
    print(f"Total de tablas numeradas:            {total_tablas}")
    print(f"Total de figuras numeradas:           {total_figuras}")
    print(f"Total de secciones en el indice:      {len(doc.paginas_secciones)}")

    return {
        "ruta_pdf": ruta_pdf,
        "ruta_md": ruta_md,
        "ruta_config": RUTA_CONFIG_REPORTE,
        "total_paginas": doc.total_paginas,
        "total_tablas": total_tablas,
        "total_figuras": total_figuras,
        "paginas_secciones": dict(doc.paginas_secciones),
        "paginas_tablas": dict(doc.paginas_tablas),
        "paginas_figuras": dict(doc.paginas_figuras),
        "metricas": insumos["metricas"],
    }


if __name__ == "__main__":
    try:
        generar_reporte_tecnico()
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        print(f"Error al generar el reporte tecnico: {error}")
