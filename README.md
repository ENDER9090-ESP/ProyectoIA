# Proyecto Final de Inteligencia Artificial — Regresión Lineal Múltiple

## Predicción de la calidad del vino tinto a partir de características fisicoquímicas (Fases 1 a 5)

---

## 1. Nombre del proyecto

**Desarrollo y evaluación de un modelo de aprendizaje supervisado para la predicción de la calidad del vino tinto mediante regresión lineal múltiple**

## 2. Objetivo general

Desarrollar un flujo reproducible de análisis de datos, aprendizaje supervisado y documentación técnica en Python para predecir la puntuación de calidad del vino tinto (`quality`) a partir de sus propiedades fisicoquímicas.

El proyecto implementa de manera integral las **5 fases**:

- **Fase 1 (Exploración y limpieza):** Carga del dataset, definición del problema de regresión, identificación de `X` e `y`, exploración inicial, eliminación metodológica de duplicados, análisis estadístico y generación de gráficas exploratorias.
- **Fase 2 (Preparación de los datos):** Partición aleatoria reproducible 80/20 en entrenamiento y prueba, y estandarización mediante `StandardScaler` evitando la fuga de datos.
- **Fase 3 (Entrenamiento del modelo):** Entrenamiento de `LinearRegression` (Regresión Lineal Múltiple) exclusivamente sobre el conjunto de entrenamiento estandarizado y almacenamiento del modelo, escalador y configuración.
- **Fase 4 (Evaluación del rendimiento):** Carga del modelo entrenado y del conjunto de prueba reservado (`X_test`, `y_test`), generación de predicciones continuas, cálculo de `MSE`, `RMSE`, `MAE` y `R²`, análisis de residuos, comparación entrenamiento vs. prueba, comparación contra `DummyRegressor`, análisis por puntuación de calidad, gráficas 13–20 e informe de evaluación.
- **Fase 5 (Reporte técnico en PDF):** Verificación de insumos existentes sin reentrenamiento, generación de versión editable Markdown (`documentacion/Reporte_Tecnico_ProyectoIA.md`) y compilación con ReportLab Platypus del reporte técnico académico (`documentacion/Reporte_Tecnico_ProyectoIA.pdf`) con portada formal, índices con paginación real, flujo metodológico, 19 tablas, 21 figuras y referencias APA 7.

## 3. Descripción del dataset

El conjunto de datos utilizado corresponde a muestras de vino tinto de la variante portuguesa *Vinho Verde* (región norte de Portugal). Contiene mediciones obtenidas mediante pruebas fisicoquímicas de laboratorio y una calificación sensorial otorgada por catadores expertos.

Características estructurales del archivo original (`winequality-red.csv`):

- **Registros originales:** 1,599 filas.
- **Columnas:** 12 columnas numéricas (11 variables independientes y 1 variable dependiente).
- **Formato del archivo:** texto delimitado por punto y coma (`;`).
- **Valores faltantes:** 0 en todas las columnas.
- **Registros completamente duplicados:** 240 filas idénticas (conservando 1,359 registros únicos en `winequality-red-limpio.csv`).

## 4. Fuente del dataset

- **Repositorio:** UCI Machine Learning Repository — *Wine Quality*
- **Enlace oficial:** https://archive.ics.uci.edu/dataset/186/wine+quality
- **Archivo utilizado:** `winequality-red.csv` (mantenido intacto sin sobrescribirse).

## 5. Tipo de aprendizaje supervisado

El problema pertenece al paradigma de **aprendizaje supervisado de regresión**, ya que se busca estimar una variable objetivo numérica (`quality`) en función de variables explicativas continuas.

- **Algoritmo utilizado:** Regresión Lineal Múltiple (`LinearRegression` de `scikit-learn`).
- **Alcance actual:** Este repositorio implementa las **Fases 1, 2, 3, 4 y 5** completas (desde la exploración inicial hasta la generación automatizada del reporte técnico final en PDF y Markdown). No se incluyen interfaces externas para capturar nuevos vinos manualmente, búsqueda de hiperparámetros ni algoritmos ajenos a la Regresión Lineal Múltiple.
- **Prevención de fuga de datos:** Tanto el escalado (`StandardScaler`) como el entrenamiento (`LinearRegression`) se ajustaron únicamente sobre los datos de entrenamiento (`X_train`, `y_train`). En las Fases 4 y 5 el modelo, el escalador y las tablas guardadas se consumen sin reentrenar ni reescalar.

## 6. Explicación de las variables `X` y `y`

Se conservan los nombres originales de las columnas del CSV:

### Variables independientes (`X` — 11 características fisicoquímicas)

1. `fixed acidity`: Acidez fija (principalmente ácido tartárico), asociada a la estructura del vino.
2. `volatile acidity`: Acidez volátil (ácido acético); niveles elevados aportan sabor avinagrado.
3. `citric acid`: Ácido cítrico en pequeñas concentraciones; aporta frescura y acidez frutal.
4. `residual sugar`: Azúcar residual que permanece tras la fermentación.
5. `chlorides`: Concentración de cloruros (sales) presentes en el vino.
6. `free sulfur dioxide`: Dióxido de azufre libre, que actúa como agente antimicrobiano y antioxidante.
7. `total sulfur dioxide`: Suma de dióxido de azufre libre y combinado.
8. `density`: Densidad del vino, influenciada por el porcentaje de alcohol y azúcar.
9. `pH`: Grado de acidez o alcalinidad en la escala de pH (típicamente entre 2.7 y 4.0).
10. `sulphates`: Aditivo sulfatado que contribuye a la estabilidad microbiológica y al contenido de SO₂.
11. `alcohol`: Porcentaje de volumen alcohólico del vino.

### Variable dependiente (`y` — variable objetivo)

- `quality`: Puntuación sensorial de calidad en escala de 0 a 10 (en este dataset se observan valores enteros entre **3** y **8**). Se mantiene en formato numérico (`int64`) y **no se transforma en categorías**, porque el objetivo del proyecto es resolver un problema de regresión.

En el dataset limpio (`n = 1,359`), las dimensiones totales antes de particionar son:
- `X`: `(1359, 11)`
- `y`: `(1359,)`

## 7. Bibliotecas utilizadas

Las dependencias verificadas en el entorno virtual `.venv` (Python 3.14.0) y registradas en `requirements.txt` son:

| Biblioteca | Versión | Uso principal |
|---|---:|---|
| `pandas` | `3.1.1` | Carga, exploración, limpieza y manipulación tabular |
| `numpy` | `2.5.5` | Soporte numérico y arreglos vectoriales |
| `matplotlib` | `3.11.2` | Creación y exportación de gráficas a 300 DPI |
| `seaborn` | `0.13.2` | Mapas de calor, boxplots e histogramas estadísticos |
| `scikit-learn` | `1.9.1` | Partición `train_test_split`, `StandardScaler` y `LinearRegression` |
| `joblib` | `1.6.0` | Serialización y recarga segura del modelo y el escalador |
| `reportlab` | `5.1.0` | Generación del reporte técnico PDF con portada, diagramas e índices reales |
| `pillow` | `13.0.0` | Carga y composición en memoria de logotipos e imágenes para el reporte PDF |

## 8. Instrucciones de instalación (portables en cualquier computadora)

El código emplea exclusivamente rutas relativas respecto a la ubicación de `utilidades.py` (`Path(__file__).resolve().parent`), por lo que puede ejecutarse en cualquier computadora sin depender de rutas absolutas locales.

1. Abrir una terminal y ubicarse dentro de la carpeta raíz del proyecto (`ProyectoIA`):
   ```bash
   cd ProyectoIA
   ```
2. Crear un entorno virtual nuevo (si no existe `.venv`):
   ```bash
   python -m venv .venv
   ```
3. Activar el entorno virtual:
   - En **Windows (PowerShell)**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - En **Linux / macOS**:
     ```bash
     source .venv/bin/activate
     ```
4. Instalar exactamente las dependencias declaradas en `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## 9. Instrucciones de ejecución

### Ejecución mediante el menú principal (recomendado)

Con el entorno virtual activado (o invocando directamente el intérprete de `.venv`):

```powershell
python main.py
```
*(En Windows PowerShell sin activar el entorno: `.\.venv\Scripts\python.exe main.py`; en Linux/macOS: `./.venv/bin/python main.py`.)*

Opciones disponibles en consola:

- **Fase 1: Exploración y limpieza**
  1. Explorar dataset original
  2. Limpiar dataset
  3. Identificar variables `X` y `y`
  4. Realizar análisis estadístico
  5. Generar gráficas exploratorias
  6. Ejecutar fase 1 completa
- **Fase 2: Preparación de los datos**
  7. Separar datos en entrenamiento y prueba
  8. Aplicar escalado de los datos
  9. Ejecutar fase 2 completa
- **Fase 3: Entrenamiento del modelo**
  10. Entrenar modelo de regresión lineal múltiple
  11. Mostrar coeficientes del modelo
- **Fase 4: Evaluación del rendimiento**
  12. Evaluar modelo mediante MSE, RMSE, MAE y R²
  13. Mostrar valores reales y predicciones
  14. Analizar residuos
  15. Comparar entrenamiento y prueba
  16. Comparar con modelo de referencia
  17. Analizar rendimiento por calidad
  18. Generar gráficas de evaluación
  19. Ejecutar fase 4 completa
- **Experimentación complementaria**
  20. Ejecutar experimentacion complementaria (Ridge CV 5-fold)
- **Fase 5: Documentación técnica**
  21. Generar reporte técnico PDF
- **Ejecución general**
  22. Ejecutar todas las fases implementadas (Fases 1 a 5 y experimentación complementaria en orden)
  0. Salir

### Ejecución individual por módulo

```powershell
python Exploracion_Datos.py
python Limpieza_Datos.py
python Identificar_X_y_Y.py
python Analisis_Datos.py
python Particion_Datos.py
python Escalado_Datos.py
python Entrenamiento_Modelo.py
python Evaluacion_Modelo.py
python Experimentacion_Complementaria.py
python Generar_Reporte.py
```

> **Nota de dependencias entre etapas:** `Particion_Datos.py` requiere `winequality-red-limpio.csv`; `Escalado_Datos.py` requiere los archivos de partición vigentes; `Entrenamiento_Modelo.py` requiere los conjuntos escalados vigentes; `Evaluacion_Modelo.py` requiere que el modelo, el escalador y la configuración estén vigentes respecto a la partición y el escalado actuales; `Experimentacion_Complementaria.py` requiere únicamente `X_train.csv` e `y_train.csv` sin escalar y nunca usa el conjunto de prueba; y `Generar_Reporte.py` valida que todos los artefactos existan y lee exclusivamente resultados guardados sin reentrenar el modelo. Si se vuelve a ejecutar la partición o el escalado, los artefactos posteriores desactualizados se invalidan automáticamente para evitar incoherencias.

## 10. Estructura actualizada del proyecto

```text
ProyectoIA/
├── .venv/                                     # Entorno virtual Python 3.14 del proyecto
├── main.py                                    # Menú principal interactivo (opciones 0 a 22)
├── utilidades.py                              # Rutas relativas, constantes, carga CSV y control de vigencia
├── Exploracion_Datos.py                       # Fase 1: Exploración inicial del dataset original
├── Limpieza_Datos.py                          # Fase 1: Validación y limpieza hacia winequality-red-limpio.csv
├── Identificar_X_y_Y.py                       # Fase 1: Separación de variables X e y para regresión
├── Analisis_Datos.py                          # Fase 1: Estadísticas, detección IQR y gráficas 01-06
├── Particion_Datos.py                         # Fase 2: División train/test 80/20 y gráficas 07-08
├── Escalado_Datos.py                          # Fase 2: Estandarización StandardScaler y gráficas 09-11
├── Entrenamiento_Modelo.py                    # Fase 3: Entrenamiento LinearRegression, JSON, joblib y gráfica 12
├── Evaluacion_Modelo.py                       # Fase 4: Evaluación en prueba, residuos, DummyRegressor, gráficas 13-20 e informe
├── Experimentacion_Complementaria.py          # Experimentación complementaria: Ridge CV 5-fold en X_train y gráfica 21
├── Generar_Reporte.py                         # Fase 5: Reporte técnico PDF corregido (ReportLab) y Markdown
├── logos_institucionales.png                  # Logotipos UAT y FIT utilizados en la portada del reporte
├── winequality-red.py                         # Script inicial de referencia conservado sin cambios
├── winequality-red.csv                        # Dataset original UCI (no modificado)
├── winequality-red-limpio.csv                 # Dataset limpio generado (1,359 x 12, sep=';')
├── requirements.txt                           # Dependencias exactas verificadas en .venv
├── README.md                                  # Documentación académica del proyecto
├── documentacion/                             # Entregables documentales de la Fase 5
│   ├── respaldo/                              # Copias de seguridad de la versión original (41 páginas)
│   │   ├── Reporte_Tecnico_ProyectoIA_original.pdf
│   │   └── Reporte_Tecnico_ProyectoIA_original.md
│   ├── configuracion_reporte.json             # Metadatos institucionales editables para la portada
│   ├── Reporte_Tecnico_ProyectoIA.pdf         # Versión PDF original conservada como respaldo
│   ├── Reporte_Tecnico_ProyectoIA_Corregido.pdf # Reporte técnico académico corregido y revisado
│   └── Reporte_Tecnico_ProyectoIA.md          # Versión editable en Markdown sincronizada con los resultados
├── modelos/                                   # Artefactos serializados del modelo entrenado
│   ├── regresion_lineal.joblib                # Modelo LinearRegression entrenado
│   ├── standard_scaler.joblib                 # Escalador StandardScaler ajustado con X_train
│   └── configuracion_modelo.json              # Metadatos, orden de columnas e intercepto
└── resultados/
    ├── datos_procesados/                      # Conjuntos generados en la Fase 2
    │   ├── X_train.csv
    │   ├── X_test.csv
    │   ├── y_train.csv
    │   ├── y_test.csv
    │   ├── X_train_escalado.csv
    │   └── X_test_escalado.csv
    ├── graficas/                              # Gráficas en formato PNG a 300 DPI
    │   ├── 01_distribucion_calidad.png
    │   ├── 02_matriz_correlacion.png
    │   ├── 03_correlacion_con_quality.png
    │   ├── 04_boxplots_variables.png
    │   ├── 05_histogramas_variables.png
    │   ├── 06_dispersion_calidad.png
    │   ├── 07_particion_datos.png
    │   ├── 08_distribucion_train_test.png
    │   ├── 09_variables_sin_escalar.png
    │   ├── 10_variables_escaladas.png
    │   ├── 11_comparacion_escalado.png
    │   ├── 12_coeficientes_regresion.png
    │   ├── 13_valores_reales_vs_predichos.png
    │   ├── 14_comparacion_predicciones.png
    │   ├── 15_grafica_residuos.png
    │   ├── 16_distribucion_residuos.png
    │   ├── 17_boxplot_residuos.png
    │   ├── 18_comparacion_train_test.png
    │   ├── 19_comparacion_modelo_base.png
    │   ├── 20_error_por_calidad.png
    │   └── 21_comparacion_ridge_cv.png
    ├── informes/                              # Reportes generados automáticamente
    │   └── resumen_evaluacion.md
    └── tablas/                                # Tablas exportadas en formato CSV
        ├── comparacion_limpieza.csv
        ├── estadisticas_descriptivas.csv
        ├── distribucion_calidad.csv
        ├── matriz_correlacion.csv
        ├── correlaciones_con_quality.csv
        ├── valores_atipicos_iqr.csv
        ├── resumen_particion.csv
        ├── comparacion_escalado.csv
        ├── parametros_escalador.csv
        ├── coeficientes_regresion.csv
        ├── predicciones_modelo.csv
        ├── metricas_evaluacion.csv
        ├── estadisticas_residuos.csv
        ├── comparacion_train_test.csv
        ├── comparacion_modelo_base.csv
        ├── rendimiento_por_calidad.csv
        └── experimentacion_ridge_cv.csv
```

---

## Fase 1: Exploración, Limpieza y Análisis Estadístico

## 11. Exploración y limpieza de datos

Durante la exploración (`Exploracion_Datos.py`) se comprobó que `winequality-red.csv` contiene 1,599 registros y 12 columnas numéricas (`11` tipo `float64` y `quality` tipo `int64`), sin valores nulos (`0`). Al verificar duplicados, se encontraron **240 filas completamente idénticas**.

En la etapa de limpieza (`Limpieza_Datos.py`) se ejecutaron las siguientes comprobaciones reales sobre los datos:

1. Verificación de valores faltantes (`0` valores nulos antes y después).
2. Confirmación de tipos numéricos en las 12 columnas.
3. Validación de la escala de `quality` dentro del rango `[0, 10]` (valores reales observados entre `3` y `8`).
4. Eliminación de filas idénticas mediante `drop_duplicates()` y reinicio de índices con `reset_index(drop=True)`.
5. Exportación al archivo `winequality-red-limpio.csv` conservando el separador `;` y sin columna de índice.

| Característica | Antes | Después |
|---|---:|---:|
| Registros | 1599 | 1359 |
| Columnas | 12 | 12 |
| Valores faltantes | 0 | 0 |
| Duplicados | 240 | 0 |

## 12. Justificación de la eliminación de duplicados

Las 240 filas completamente idénticas se eliminaron como una **decisión metodológica preventiva** para reducir la probabilidad de que observaciones exactamente iguales queden repartidas simultáneamente en el conjunto de entrenamiento y en el de prueba. Si un mismo registro aparece en ambos conjuntos, la evaluación del modelo puede resultar artificialmente optimista (*fuga de datos*). Se reconoce que, desde el punto de vista fisicoquímico, dos vinos distintos podrían coincidir en todas sus mediciones; no obstante, en modelado supervisado prima evitar la duplicación entre particiones.

## 13. Justificación de la conservación de valores atípicos

Aunque el método del rango intercuartílico (IQR) identifica observaciones alejadas de la mediana en varias variables (como `residual sugar` o `chlorides`), **no se eliminan registros automáticamente**. En el dominio enológico, valores altos de azúcar residual, sulfatos o dióxido de azufre pueden corresponder a muestras reales de vino con perfiles específicos, y su eliminación indiscriminada distorsionaría la variabilidad natural del fenómeno que el modelo debe aprender.

## 14. Resultados reales de la Fase 1

Todos los cálculos corresponden al dataset limpio (`1,359` registros y `12` columnas), guardados en `resultados/tablas/`.

### 14.1. Estadísticas descriptivas (`estadisticas_descriptivas.csv`)

| Variable | Cantidad | Media | Desv. estándar | Mínimo | Q1 (25%) | Mediana (50%) | Q3 (75%) | Máximo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed acidity` | 1359 | 8.3106 | 1.7370 | 4.6000 | 7.1000 | 7.9000 | 9.2000 | 15.9000 |
| `volatile acidity` | 1359 | 0.5295 | 0.1830 | 0.1200 | 0.3900 | 0.5200 | 0.6400 | 1.5800 |
| `citric acid` | 1359 | 0.2723 | 0.1955 | 0.0000 | 0.0900 | 0.2600 | 0.4300 | 1.0000 |
| `residual sugar` | 1359 | 2.5234 | 1.3523 | 0.9000 | 1.9000 | 2.2000 | 2.6000 | 15.5000 |
| `chlorides` | 1359 | 0.0881 | 0.0494 | 0.0120 | 0.0700 | 0.0790 | 0.0910 | 0.6110 |
| `free sulfur dioxide` | 1359 | 15.8933 | 10.4473 | 1.0000 | 7.0000 | 14.0000 | 21.0000 | 72.0000 |
| `total sulfur dioxide` | 1359 | 46.8260 | 33.4089 | 6.0000 | 22.0000 | 38.0000 | 63.0000 | 289.0000 |
| `density` | 1359 | 0.9967 | 0.0019 | 0.9901 | 0.9956 | 0.9967 | 0.9978 | 1.0037 |
| `pH` | 1359 | 3.3098 | 0.1550 | 2.7400 | 3.2100 | 3.3100 | 3.4000 | 4.0100 |
| `sulphates` | 1359 | 0.6587 | 0.1707 | 0.3300 | 0.5500 | 0.6200 | 0.7300 | 2.0000 |
| `alcohol` | 1359 | 10.4323 | 1.0821 | 8.4000 | 9.5000 | 10.2000 | 11.1000 | 14.9000 |
| `quality` | 1359 | 5.6233 | 0.8236 | 3.0000 | 5.0000 | 6.0000 | 6.0000 | 8.0000 |

### 14.2. Distribución de la calidad del vino (`distribucion_calidad.csv`)

| Puntuación (`quality`) | Cantidad | Porcentaje (%) |
|---:|---:|---:|
| 3 | 10 | 0.74% |
| 4 | 53 | 3.90% |
| 5 | 577 | 42.46% |
| 6 | 535 | 39.37% |
| 7 | 167 | 12.29% |
| 8 | 17 | 1.25% |

El **81.83%** de las muestras limpias (`1,112` de `1,359`) se concentra en las puntuaciones **5** y **6**.

### 14.3. Correlaciones con la variable objetivo `quality` (`correlaciones_con_quality.csv`)

| Variable independiente | Correlación de Pearson con `quality` |
|---|---:|
| `alcohol` | `+0.4803` |
| `sulphates` | `+0.2488` |
| `citric acid` | `+0.2281` |
| `fixed acidity` | `+0.1190` |
| `residual sugar` | `+0.0136` |
| `free sulfur dioxide` | `-0.0505` |
| `pH` | `-0.0552` |
| `chlorides` | `-0.1310` |
| `total sulfur dioxide` | `-0.1779` |
| `density` | `-0.1843` |
| `volatile acidity` | `-0.3952` |

Entre variables independientes destacan correlaciones fuertes como `fixed acidity` vs `pH` (`-0.69`), `fixed acidity` vs `density` (`+0.67`), `fixed acidity` vs `citric acid` (`+0.67`) y `free sulfur dioxide` vs `total sulfur dioxide` (`+0.67`). No se elimina ninguna variable únicamente por su correlación individual.

### 14.4. Posibles valores atípicos mediante IQR (`valores_atipicos_iqr.csv`)

| Variable | Q1 | Q3 | IQR | Límite inferior | Límite superior | Atípicos detectados | Porcentaje (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed acidity` | 7.1000 | 9.2000 | 2.1000 | 3.9500 | 12.3500 | 41 | 3.02% |
| `volatile acidity` | 0.3900 | 0.6400 | 0.2500 | 0.0150 | 1.0150 | 19 | 1.40% |
| `citric acid` | 0.0900 | 0.4300 | 0.3400 | -0.4200 | 0.9400 | 1 | 0.07% |
| `residual sugar` | 1.9000 | 2.6000 | 0.7000 | 0.8500 | 3.6500 | 126 | 9.27% |
| `chlorides` | 0.0700 | 0.0910 | 0.0210 | 0.0385 | 0.1225 | 87 | 6.40% |
| `free sulfur dioxide` | 7.0000 | 21.0000 | 14.0000 | -14.0000 | 42.0000 | 26 | 1.91% |
| `total sulfur dioxide` | 22.0000 | 63.0000 | 41.0000 | -39.5000 | 124.5000 | 45 | 3.31% |
| `density` | 0.9956 | 0.9978 | 0.0022 | 0.9923 | 1.0011 | 35 | 2.58% |
| `pH` | 3.2100 | 3.4000 | 0.1900 | 2.9250 | 3.6850 | 28 | 2.06% |
| `sulphates` | 0.5500 | 0.7300 | 0.1800 | 0.2800 | 1.0000 | 55 | 4.05% |
| `alcohol` | 9.5000 | 11.1000 | 1.6000 | 7.1000 | 13.5000 | 12 | 0.88% |

## 15. Limitaciones del dataset

- **Concentración en calidades intermedias:** Más del 81% de los vinos tiene puntuación `5` o `6`, mientras que las notas extremas (`3` y `8`) suman apenas 27 registros y no existen muestras con calificaciones `0`, `1`, `2`, `9` ni `10`.
- **Especificidad geográfica y varietal:** Todas las muestras corresponden exclusivamente a vino tinto *Vinho Verde* de Portugal.
- **Variable objetivo discreta y evaluación sensorial:** Las calificaciones originales son enteros otorgados mediante evaluación humana subjetiva (mediana de al menos tres catadores).
- **Relaciones lineales moderadas y colinealidad:** Ninguna variable individual supera una correlación lineal de `0.48` con `quality`, y existen predictores altamente correlacionados entre sí.

---

## Fase 2: Preparación de los Datos (Partición y Escalado)

## 16. Preparación y partición de los datos

En el módulo `Particion_Datos.py` se parte del dataset limpio (`winequality-red-limpio.csv`, **1,359** registros y **12** columnas), separando las 11 variables independientes (`X`) y la variable dependiente numérica (`y = quality`) sin convertir `quality` en categorías.

### 16.1. Justificación de la división 80/20

Se utiliza una partición aleatoria estratificada por proporción clásica **80% entrenamiento / 20% prueba** mediante `train_test_split(..., test_size=0.20, shuffle=True)`. La partición aleatoria es adecuada porque cada fila representa una muestra independiente de vino y el conjunto de datos **no contiene marca temporal ni identificadores de grupos o lotes** que exijan una partición cronológica o por conglomerados. El 80% permite disponer de suficientes ejemplos para estimar los 12 parámetros del modelo lineal (intercepto y 11 pendientes), reservando un 20% intacto para evaluar la capacidad de generalización en la Fase 4.

### 16.2. Reproducibilidad mediante `random_state`

Se fija **`random_state=42`** junto con `shuffle=True`. Esto garantiza que cualquier ejecución posterior en cualquier equipo produzca exactamente los mismos subconjuntos `X_train`, `X_test`, `y_train` e `y_test`, asegurando trazabilidad y reproducibilidad científica durante la exposición académica.

### 16.3. Cantidad real de datos de entrenamiento y prueba (`resumen_particion.csv`)

| Conjunto | Registros | Porcentaje real (%) | Dimensiones `X` | Dimensiones `y` | Semilla aleatoria |
|---|---:|---:|---|---|---:|
| Entrenamiento (`train`) | 1,087 | 79.99% | `(1087, 11)` | `(1087,)` | `42` |
| Prueba (`test`) | 272 | 20.01% | `(272, 11)` | `(272,)` | `42` |
| **Total** | **1,359** | **100.00%** | `(1359, 11)` | `(1359,)` | `42` |

Se verificó automáticamente que el número de índices compartidos entre entrenamiento y prueba es **0** (sin superposición).

## 17. Estandarización mediante `StandardScaler`

En `Escalado_Datos.py` se aplica la estandarización de las 11 variables independientes después de dividir los datos.

### 17.1. Justificación y fórmula de estandarización

Las variables fisicoquímicas tienen unidades y escalas muy dispares: `density` ronda `0.9967`, `chlorides` se ubica alrededor de `0.0896`, `pH` promedia `3.3095`, `alcohol` ronda `10.4355%`, mientras `total sulfur dioxide` promedia `46.8491` y alcanza valores de hasta `289`.

`StandardScaler` transforma cada característica usando la fórmula del puntaje estándar ($z$):

$$z = \frac{x - \mu}{\sigma}$$

donde $\mu$ es la media de la variable en el conjunto de entrenamiento y $\sigma$ es su desviación estándar poblacional (`ddof=0`). Aunque la regresión lineal ordinaria no exige obligatoriamente el escalado para producir predicciones, se implementa por requisito académico y porque al estandarizar las variables en unidades de desviación estándar es posible **comparar directamente la magnitud de los coeficientes aprendidos**.

### 17.2. Prevención de fuga de datos (*data leakage*)

Para evitar que información estadística del conjunto de prueba influya en la preparación del entrenamiento, se cumplen estrictamente las siguientes reglas:

1. `escalador.fit_transform(X_train)` se aplica **únicamente sobre `X_train`** (`1,087` registros).
2. `escalador.transform(X_test)` transforma `X_test` (`272` registros) utilizando la media y desviación aprendidas exclusivamente de `X_train`.
3. Nunca se utiliza `fit_transform()` sobre `X_test` ni se ajusta el escalador sobre el dataset completo.
4. La variable dependiente `quality` (`y_train` e `y_test`) **no se escala** y conserva su escala original de puntuación (`3` a `8`).
5. No se exige que `X_test_escalado` tenga media `0` y desviación `1`, precisamente porque utiliza los parámetros del conjunto de entrenamiento.

### 17.3. Comparación real antes y después del escalado en `X_train` (`comparacion_escalado.csv`)

| Variable | Media antes (`train`) | Desv. est. (`ddof=0`) antes | Media después (`train`) | Desv. est. (`ddof=0`) después |
|---|---:|---:|---:|---:|
| `fixed acidity` | 8.2934 | 1.7252 | `0.0000` | `1.0000` |
| `volatile acidity` | 0.5277 | 0.1784 | `0.0000` | `1.0000` |
| `citric acid` | 0.2733 | 0.1953 | `0.0000` | `1.0000` |
| `residual sugar` | 2.5255 | 1.3651 | `0.0000` | `1.0000` |
| `chlorides` | 0.0896 | 0.0525 | `0.0000` | `1.0000` |
| `free sulfur dioxide` | 15.7406 | 10.3311 | `0.0000` | `1.0000` |
| `total sulfur dioxide` | 46.8491 | 33.2848 | `0.0000` | `1.0000` |
| `density` | 0.9967 | 0.0019 | `0.0000` | `1.0000` |
| `pH` | 3.3095 | 0.1522 | `0.0000` | `1.0000` |
| `sulphates` | 0.6597 | 0.1738 | `0.0000` | `1.0000` |
| `alcohol` | 10.4355 | 1.0909 | `0.0000` | `1.0000` |

Los parámetros exactos aprendidos por `StandardScaler` (`mean_`, `scale_`, `var_`) se almacenan en `resultados/tablas/parametros_escalador.csv` y el objeto ajustado se guarda en `modelos/standard_scaler.joblib`.

---

## Fase 3: Entrenamiento del Modelo de Regresión Lineal Múltiple

## 18. Regresión lineal múltiple y método de mínimos cuadrados

En `Entrenamiento_Modelo.py` se entrena el modelo `LinearRegression` de `scikit-learn`. Matemáticamente, el modelo expresa la calidad estimada como una combinación lineal de las 11 variables estandarizadas:

$$y_{\text{pred}} = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_{11} x_{11}$$

donde:
- $y_{\text{pred}}$ es la puntuación de calidad predicha.
- $b_0$ es el intercepto del modelo.
- $b_1, \dots, b_{11}$ son los coeficientes estimados para cada característica fisicoquímica estandarizada $x_1, \dots, x_{11}$.

### 18.1. Método de mínimos cuadrados ordinarios (OLS)

`LinearRegression` emplea el método de **mínimos cuadrados ordinarios (OLS)** para encontrar los valores de $b_0$ y $b_1, \dots, b_{11}$ que minimizan la suma de los errores cuadrados entre los valores reales $y_i$ y las predicciones $\hat{y}_i$ sobre los **1,087** registros de entrenamiento:

$$\min_{b_0, \dots, b_{11}} \sum_{i=1}^{n_{\text{train}}} (y_i - \hat{y}_i)^2$$

### 18.2. Proceso de entrenamiento

1. Se cargan exclusivamente `X_train_escalado.csv` (`1087 x 11`) e `y_train.csv` (`1087`).
2. Se verifica el orden exacto de las 11 columnas y la coincidencia de índices.
3. Se ejecuta `modelo.fit(X_train_escalado, y_train)`.
4. `X_test_escalado` e `y_test` **no se utilizan** para entrenar, seleccionar características ni ajustar parámetros.

### 18.3. Intercepto, ecuación obtenida y coeficientes reales (`coeficientes_regresion.csv`)

- **Intercepto obtenido (`modelo.intercept_`):** **`5.636615`** (`5.636614535418583`).
- **Ecuación obtenida (redondeada a 4 decimales):**
  ```text
  y_pred = 5.6366 - 0.0497*(fixed acidity) - 0.1785*(volatile acidity) - 0.0191*(citric acid) - 0.0052*(residual sugar) - 0.1160*(chlorides) + 0.0487*(free sulfur dioxide) - 0.1236*(total sulfur dioxide) + 0.0421*(density) - 0.1215*(pH) + 0.1485*(sulphates) + 0.3340*(alcohol)
  ```

Tabla real de coeficientes ordenados por su valor absoluto (de mayor a menor magnitud):

| Orden | Variable | Coeficiente | Valor absoluto | Signo |
|---:|---|---:|---:|---|
| 1 | `alcohol` | `+0.333970` | `0.333970` | Positivo |
| 2 | `volatile acidity` | `-0.178494` | `0.178494` | Negativo |
| 3 | `sulphates` | `+0.148499` | `0.148499` | Positivo |
| 4 | `total sulfur dioxide` | `-0.123600` | `0.123600` | Negativo |
| 5 | `pH` | `-0.121548` | `0.121548` | Negativo |
| 6 | `chlorides` | `-0.115971` | `0.115971` | Negativo |
| 7 | `fixed acidity` | `-0.049675` | `0.049675` | Negativo |
| 8 | `free sulfur dioxide` | `+0.048723` | `0.048723` | Positivo |
| 9 | `density` | `+0.042064` | `0.042064` | Positivo |
| 10 | `citric acid` | `-0.019148` | `0.019148` | Negativo |
| 11 | `residual sugar` | `-0.005231` | `0.005231` | Negativo |

### 18.4. Interpretación del intercepto y los coeficientes

- **Interpretación del intercepto (`5.636615`):** Representa la calidad estimada por el modelo cuando todas las variables estandarizadas valen `0`, es decir, para un vino tinto hipotético cuyos valores fisicoquímicos coinciden exactamente con la media de cada característica en el conjunto de entrenamiento.
- **Interpretación de los coeficientes estandarizados:** Como todas las variables de `X` fueron estandarizadas, cada coeficiente indica el cambio estimado en la puntuación de `quality` ante un aumento de **una desviación estándar** en esa variable, **manteniendo constantes las demás variables** (*ceteris paribus*).
  - **Signo positivo** (como `alcohol`: `+0.3340`, `sulphates`: `+0.1485`): indica una relación positiva estimada con la calidad; por cada desviación estándar adicional de `alcohol` (`+1.0909%` en entrenamiento), el modelo estima `+0.3340` puntos de calidad manteniendo fijas las demás variables.
  - **Signo negativo** (como `volatile acidity`: `-0.1785`, `total sulfur dioxide`: `-0.1236`): indica una relación negativa estimada con la calidad manteniendo constantes las demás características.
- **No causalidad y efecto de la colinealidad:** Los coeficientes reflejan asociaciones lineales estadísticas, no relaciones causa-efecto. Además, debido a la colinealidad entre variables fisicoquímicas (por ejemplo, entre `fixed acidity`, `citric acid`, `density` y `pH`, o entre `density` y `alcohol`), el signo parcial de una variable en el modelo múltiple puede diferir de su correlación individual con `quality` (por ejemplo, `citric acid` presenta correlación simple positiva `+0.2281` con `quality`, pero coeficiente parcial `-0.0191` al controlar por el resto de variables).

## 19. Almacenamiento del modelo y artefactos

Para reutilizar el modelo en la futura fase de evaluación sin reentrenar y garantizar la compatibilidad de columnas, se almacenan en `modelos/`:

- `modelos/regresion_lineal.joblib`: Objeto `LinearRegression` entrenado.
- `modelos/standard_scaler.joblib`: Objeto `StandardScaler` ajustado con `X_train`.
- `modelos/configuracion_modelo.json`: Metadatos completos (algoritmo, dataset, variable objetivo, orden exacto de las 11 variables independientes, tamaño de entrenamiento `1,087`, tamaño de prueba `272`, `random_state=42`, porcentajes teóricos y reales, método de escalado e intercepto).

Al finalizar el entrenamiento, el programa recarga ambos archivos `.joblib` y verifica que los parámetros y el orden de características (`feature_names_in_`) coincidan exactamente con los entrenados.

---

## Fase 4: Evaluación del Rendimiento

## 20. Evaluación del rendimiento y generación de predicciones

En el módulo `Evaluacion_Modelo.py` se evalúa el modelo entrenado utilizando los **272** registros del conjunto de prueba (`X_test_escalado`, `y_test`) reservados en la Fase 2, sin reentrenar `LinearRegression`, sin volver a ajustar `StandardScaler` y sin redondear, recortar o categorizar las predicciones antes de calcular las métricas.

Cada observación de prueba se almacena en `resultados/tablas/predicciones_modelo.csv` con las columnas:
`Registro`, `Calidad_real`, `Calidad_predicha`, `Residuo`, `Error_absoluto` y `Error_cuadratico`.

## 21. Métricas de evaluación reales (`metricas_evaluacion.csv`)

Las cuatro métricas calculadas sobre los **272** registros de prueba son:

| Métrica | Fórmula | Valor real (prueba) | Interpretación |
|---|---|---:|---|
| **MSE** | $\frac{1}{n}\sum (y_{\text{real}} - y_{\text{pred}})^2$ | `0.4310` | Promedio de errores al cuadrado (en unidades de calidad²); penaliza en mayor medida las desviaciones grandes. |
| **RMSE** | $\sqrt{\text{MSE}}$ | `0.6565` | Raíz del MSE (`sqrt(0.4310)`); expresa el error cuadrático típico en la misma escala que `quality`. |
| **MAE** | $\frac{1}{n}\sum \|y_{\text{real}} - y_{\text{pred}}\|$ | `0.5041` | En promedio, las predicciones se desvían `0.5041` puntos de calidad respecto al valor real. |
| **R²** | $1 - \frac{\sum (y_{\text{real}} - y_{\text{pred}})^2}{\sum (y_{\text{real}} - \bar{y})^2}$ | `0.3915` | El modelo lineal explica aproximadamente el `39.15%` de la variabilidad de `quality` en prueba (no es exactitud de clasificación). |

## 22. Comparación de valores reales y predichos

- En la gráfica `13_valores_reales_vs_predichos.png`, los vinos con calidad real `5` y `6` se agrupan cerca de la diagonal `y = x`, mientras que en los extremos (`3`, `4`, `7`, `8`) las predicciones tienden hacia el rango central (~`4.8` a `6.9`).
- En `14_comparacion_predicciones.png` se comparan los primeros 30 registros de prueba (en orden no temporal), evidenciando que el modelo sigue la tendencia general pero suaviza los picos extremos.

## 23. Análisis de residuos (`estadisticas_residuos.csv`)

Se define cada residuo como `Residuo = Calidad_real - Calidad_predicha`:
- **Residuo positivo (`> 0`):** el modelo subestimó la calidad real (`122` registros en prueba).
- **Residuo negativo (`< 0`):** el modelo sobreestimó la calidad real (`150` registros en prueba).
- **Residuo igual a cero (tolerancia `1e-6`):** `0` registros en prueba.

Estadísticas reales de los residuos en prueba:
- **Media:** `-0.0677`
- **Mediana:** `-0.0966`
- **Desviación estándar (`ddof=1`):** `0.6542`
- **Mínimo / Máximo:** `-2.2067` / `+1.9077`

Dado que `quality` solo toma valores enteros (`3`–`8`) y la salida del modelo es continua, la gráfica `15_grafica_residuos.png` muestra bandas diagonales características. El histograma (`16_distribucion_residuos.png`) y el boxplot (`17_boxplot_residuos.png`) muestran mayor densidad cerca de cero y algunas colas extremas, sin afirmar normalidad únicamente a partir de la inspección visual del histograma.

## 24. Comparación entrenamiento vs. prueba, modelo de referencia y generalización

### 24.1. Entrenamiento vs. prueba (`comparacion_train_test.csv`)

| Métrica | Entrenamiento (`n=1,087`) | Prueba (`n=272`) |
|---|---:|---:|
| **MSE** | `0.4337` | `0.4310` |
| **RMSE** | `0.6585` | `0.6565` |
| **MAE** | `0.5107` | `0.5041` |
| **R²** | `0.3520` | `0.3915` |

### 24.2. Modelo de referencia `DummyRegressor(strategy="mean")` (`comparacion_modelo_base.csv`)

| Métrica | Modelo base (`DummyRegressor`) | Regresión Lineal Múltiple | Diferencia (`RLM - Base`) |
|---|---:|---:|---:|
| **MSE** | `0.7128` | `0.4310` | `-0.2818` |
| **RMSE** | `0.8443` | `0.6565` | `-0.1878` |
| **MAE** | `0.7049` | `0.5041` | `-0.2007` |
| **R²** | `-0.0063` | `0.3915` | `+0.3978` |

### 24.3. Análisis de sobreajuste, subajuste y generalización

- **Sobreajuste:** No se observan indicios de sobreajuste, pues los errores y el `R²` en prueba (`RMSE = 0.6565`, `R² = 0.3915`) son prácticamente iguales (incluso levemente mejores por variación muestral) que en entrenamiento (`RMSE = 0.6585`, `R² = 0.3520`).
- **Subajuste:** Sí se aprecia una capacidad explicativa limitada (subajuste respecto a la complejidad del fenómeno): el modelo lineal explica menos del `40%` de la variabilidad tanto en entrenamiento como en prueba.
- **Generalización:** La estabilidad entre entrenamiento y prueba, sumada a la mejora frente a `DummyRegressor` (`-0.2007` en MAE y `+0.3978` en `R²`), confirma que el modelo aprende patrones lineales útiles y generalizables, aunque con error moderado (`~0.50` puntos de MAE).

## 25. Rendimiento por puntuación de calidad (`rendimiento_por_calidad.csv`)

| Calidad real | Registros (`n`) | Promedio predicciones | MAE | RMSE | Residuo promedio |
|---:|---:|---:|---:|---:|---:|
| 3 | 4 | `5.0466` | `2.0466` | `2.0531` | `-2.0466` |
| 4 | 11 | `5.3208` | `1.3208` | `1.3475` | `-1.3208` |
| 5 | 120 | `5.3738` | `0.4145` | `0.5042` | `-0.3738` |
| 6 | 103 | `5.8046` | `0.3459` | `0.4323` | `+0.1954` |
| 7 | 31 | `6.2045` | `0.7955` | `0.8833` | `+0.7955` |
| 8 | 3 | `6.5400` | `1.4600` | `1.4680` | `+1.4600` |

Los menores errores se presentan en `quality = 6` (`MAE = 0.3459`) y `quality = 5` (`MAE = 0.4145`), que agrupan el `81.99%` del conjunto de prueba (`223` de `272` vinos). En los extremos (`3`, `4`, `7`, `8`) los errores crecen notablemente y las puntuaciones `3` (`n=4`) y `8` (`n=3`) ofrecen estimaciones menos estables debido al bajo número de observaciones.

## 26. Limitaciones y conclusiones preliminares

- **Limitaciones observadas en los resultados:** fuerte concentración en calidades `5` y `6` (`81.99%` en prueba), escasez de ejemplos en calidades extremas (`3`, `4` y `8`), errores elevados en dichos extremos y un `R²` moderado (`0.3915`).
- **Limitaciones potenciales del dominio:** subjetividad sensorial en `quality`, posibles relaciones fisicoquímicas no lineales, colinealidad entre predictores y alcance restringido a vinos similares a los de la muestra *Vinho Verde*.
- **Conclusiones preliminares:** La Regresión Lineal Múltiple supera claramente al promedio simple (`DummyRegressor`) y muestra consistencia entre entrenamiento y prueba, pero no debe presentarse como un predictor de alta precisión para calidades extremas. El resumen completo se guarda automáticamente en `resultados/informes/resumen_evaluacion.md`.

## 27. Descripción de las gráficas, tablas e informe generados

### Gráficas PNG a 300 DPI (`resultados/graficas/`)

**Fase 1 (Exploración y análisis):**
1. `01_distribucion_calidad.png`: Conteo exacto de vinos por puntuación (`3` a `8`) en el dataset limpio.
2. `02_matriz_correlacion.png`: Mapa de calor (`coolwarm`, `-1` a `1`, dos decimales) de correlaciones de Pearson.
3. `03_correlacion_con_quality.png`: Barras horizontales ordenadas con la correlación individual de cada variable respecto a `quality`.
4. `04_boxplots_variables.png`: Diagramas de caja individuales para las 11 variables y su conteo IQR de atípicos.
5. `05_histogramas_variables.png`: Histogramas con KDE para observar asimetrías y distribuciones.
6. `06_dispersion_calidad.png`: Gráficas de dispersión de `alcohol`, `volatile acidity`, `sulphates` y `citric acid` frente a `quality`.

**Fase 2 (Partición y escalado):**
7. `07_particion_datos.png`: Comparación visual entre la cantidad y porcentaje real de registros de entrenamiento (`1,087` / `79.99%`) y prueba (`272` / `20.01%`).
8. `08_distribucion_train_test.png`: Comparación porcentual de la distribución de `quality` entre entrenamiento y prueba.
9. `09_variables_sin_escalar.png`: Diagramas de caja horizontales de `X_train` en sus escalas originales.
10. `10_variables_escaladas.png`: Diagramas de caja horizontales de `X_train_escalado` centrados alrededor de `0` tras `StandardScaler`.
11. `11_comparacion_escalado.png`: Panel cuádruple comparando medias y desviaciones estándar (`ddof=0`) de `X_train` antes y después de estandarizar.

**Fase 3 (Entrenamiento del modelo):**
12. `12_coeficientes_regresion.png`: Barras horizontales con los 11 coeficientes estandarizados del modelo `LinearRegression`.

**Fase 4 (Evaluación del rendimiento):**
13. `13_valores_reales_vs_predichos.png`: Dispersión entre calidad real y calidad predicha con diagonal `y = x`.
14. `14_comparacion_predicciones.png`: Comparación lineal de los primeros 30 registros de prueba (orden no temporal).
15. `15_grafica_residuos.png`: Residuos frente a valores predichos con línea de referencia en `0`.
16. `16_distribucion_residuos.png`: Histograma con KDE de los residuos en prueba.
17. `17_boxplot_residuos.png`: Diagrama de caja horizontal de los residuos en prueba.
18. `18_comparacion_train_test.png`: Panel doble comparando `MSE`, `RMSE`, `MAE` y `R²` entre entrenamiento y prueba.
19. `19_comparacion_modelo_base.png`: Panel doble comparando `DummyRegressor` frente a `LinearRegression` en prueba.
20. `20_error_por_calidad.png`: Barras de `MAE` y conteo `(n)` por cada puntuación real de calidad.

**Experimentación complementaria (Ridge CV 5-fold en entrenamiento):**
21. `21_comparacion_ridge_cv.png`: Comparación de `RMSE` y `R²` promedio ± desviación estándar en validación cruzada 5-fold (`X_train`) para `Ridge` (`alpha = 0.01, 0.1, 1, 10, 100`) frente a `LinearRegression`.

### Tablas CSV (`resultados/tablas/`) e Informe (`resultados/informes/`)

- **Fase 1:** `comparacion_limpieza.csv`, `estadisticas_descriptivas.csv`, `distribucion_calidad.csv`, `matriz_correlacion.csv`, `correlaciones_con_quality.csv`, `valores_atipicos_iqr.csv`.
- **Fase 2:** `resumen_particion.csv`, `comparacion_escalado.csv`, `parametros_escalador.csv`.
- **Fase 3:** `coeficientes_regresion.csv`.
- **Fase 4:** `predicciones_modelo.csv`, `metricas_evaluacion.csv`, `estadisticas_residuos.csv`, `comparacion_train_test.csv`, `comparacion_modelo_base.csv`, `rendimiento_por_calidad.csv`.
- **Experimentación complementaria:** `experimentacion_ridge_cv.csv`.
- **Informe Markdown:** `resultados/informes/resumen_evaluacion.md`.

---

## Fase 5: Reporte Técnico Académico en PDF (`Generar_Reporte.py`)

La Fase 5 compila toda la documentación técnica, matemática y experimental (incluyendo la experimentación complementaria con `Ridge` en validación cruzada) sin volver a entrenar el modelo principal ni alterar los resultados guardados:

1. **Respaldo de seguridad (`documentacion/respaldo/`):** Conserva `Reporte_Tecnico_ProyectoIA.pdf` y las copias originales `Reporte_Tecnico_ProyectoIA_original.pdf` y `.md`.
2. **Verificación previa (`verificar_insumos_reporte`):** Comprueba antes de generar el PDF que existan `requirements.txt`, `modelos/configuracion_modelo.json`, `resultados/informes/resumen_evaluacion.md`, los conjuntos de partición en `datos_procesados/`, las 17 tablas CSV y las 21 gráficas PNG.
3. **Configuración institucional (`documentacion/configuracion_reporte.json`):** Almacena la institución, facultad, carrera, materia, título, integrantes, profesor, lugar, fecha y el campo opcional `"grupo": ""` editable antes de recompilar.
4. **Generación automática (`documentacion/Reporte_Tecnico_ProyectoIA_Corregido.pdf` y `Reporte_Tecnico_ProyectoIA.md`):** Compila con ReportLab 5.1.0 la versión corregida del documento en tamaño Carta (con Anexo C en página horizontal), portada institucional (`logos_institucionales.png`), Resumen y palabras clave, índices reales, diagrama de flujo vectorial, ecuaciones con variables estandarizadas `z(...)`, desarrollo de Fases 1–4, apartado de experimentación complementaria con Ridge CV, fragmentos clave (incluyendo `LinearRegression().fit(...)`), discusión crítica, limitaciones, conclusiones, 11 referencias verificadas en formato APA 7 y cinco anexos técnicos.

### Cómo regenerar el reporte técnico corregido

- Desde el menú interactivo (`python main.py`): seleccionar la **opción `20`** para ejecutar la experimentación complementaria Ridge CV, la **opción `21`** para generar el reporte técnico PDF corregido sin reentrenar, o la **opción `22`** para ejecutar todo el flujo completo.
- Directamente por consola:
  ```powershell
  python Experimentacion_Complementaria.py
  python Generar_Reporte.py
  ```
