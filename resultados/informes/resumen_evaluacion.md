# Resumen de Evaluación del Modelo — Fase 4

## 1. Información general de la evaluación

- **Modelo evaluado:** Regresión Lineal Múltiple (LinearRegression)
- **Dataset:** `winequality-red-limpio.csv`
- **Registros de entrenamiento:** 1087
- **Registros de prueba (evaluados):** 272
- **Variables independientes:** 11

## 2. Métricas obtenidas en el conjunto de prueba

| Métrica | Valor obtenido | Interpretación individual |
|---|---:|---|
| **MSE** | `0.4310` | Promedio de los errores al cuadrado (unidades de calidad²); penaliza en mayor medida las desviaciones grandes. |
| **RMSE** | `0.6565` | Raíz cuadrada del MSE (`sqrt(0.4310)`); error cuadrático típico expresado en puntos de calidad. |
| **MAE** | `0.5041` | En promedio, las predicciones se desvían `0.5041` puntos respecto a la calificación real de calidad. |
| **R²** | `0.3915` | El modelo lineal explica aproximadamente el `39.15%` de la variabilidad de `quality` en el conjunto de prueba (no equivale a exactitud de clasificación). |

## 3. Comparación entre entrenamiento y prueba

| Métrica | Entrenamiento | Prueba |
|---|---:|---:|
| MSE | 0.4337 | 0.4310 |
| RMSE | 0.6585 | 0.6565 |
| MAE | 0.5107 | 0.5041 |
| R² | 0.3520 | 0.3915 |

## 4. Comparación con el modelo de referencia (`DummyRegressor`)

El modelo de referencia (`DummyRegressor(strategy="mean")`) predice constantemente la media de `quality` observada en entrenamiento.

| Métrica | Modelo base (`DummyRegressor`) | Regresión Lineal Múltiple | Diferencia (RLM - Base) |
|---|---:|---:|---:|
| MSE | 0.7128 | 0.4310 | -0.2818 |
| RMSE | 0.8443 | 0.6565 | -0.1878 |
| MAE | 0.7049 | 0.5041 | -0.2007 |
| R² | -0.0063 | 0.3915 | +0.3978 |

La regresión lineal múltiple supera al modelo base al disminuir los errores (`MSE`, `RMSE` y `MAE`) y aumentar `R²` de `-0.0063` a `0.3915`, lo que confirma que las 11 variables fisicoquímicas aportan información lineal útil por encima de predecir simplemente el promedio.

## 5. Análisis de los residuos (`Calidad_real - Calidad_predicha`)

- **Media de los residuos:** `-0.0677`
- **Mediana de los residuos:** `-0.0966`
- **Desviación estándar:** `0.6542`
- **Rango observado:** mínimo `-2.2067` hasta máximo `+1.9077`
- **Conteo por signo (tolerancia `1e-6`):** `122` positivos (subestimación), `150` negativos (sobreestimación) y `0` iguales a cero.

Dado que `quality` solo toma valores enteros (`3` a `8`) mientras el modelo emite predicciones continuas, los residuos muestran bandas diagonales en la gráfica de dispersión y diferencias sistemáticas en los extremos de la escala. No se concluye normalidad únicamente a partir del histograma de residuos.

## 6. Rendimiento por puntuación de calidad

| Calidad real | Cantidad de registros | Promedio de predicciones | MAE | RMSE | Residuo promedio |
|---:|---:|---:|---:|---:|---:|
| 3 | 4 | 5.0466 | 2.0466 | 2.0531 | -2.0466 |
| 4 | 11 | 5.3208 | 1.3208 | 1.3475 | -1.3208 |
| 5 | 120 | 5.3738 | 0.4145 | 0.5042 | -0.3738 |
| 6 | 103 | 5.8046 | 0.3459 | 0.4323 | +0.1954 |
| 7 | 31 | 6.2045 | 0.7955 | 0.8833 | +0.7955 |
| 8 | 3 | 6.5400 | 1.4600 | 1.4680 | +1.4600 |

En el conjunto de prueba, **223 de 272 registros (81.99%)** tienen puntuación real `5` o `6`, donde el MAE es menor. En las calidades extremas (`3`, `4`, `7` y `8`), tanto el promedio de predicciones tiende hacia el centro de la escala (entre ~`5.1` y ~`6.5`) como el error promedio aumenta considerablemente. Además, puntuaciones como `3` (`n=4`) y `8` (`n=3`) contienen muy pocas muestras de prueba, por lo que sus estimaciones puntuales son menos estables.

## 7. Indicios de sobreajuste, subajuste y capacidad de generalización

- **Sobreajuste:** No se observan indicios marcados de sobreajuste, ya que el rendimiento en entrenamiento (`R² = 0.3520`, `RMSE = 0.6585`) es muy cercano al obtenido en prueba (`R² = 0.3915`, `RMSE = 0.6565`), con una diferencia de `R²` de `-0.0395` y de `RMSE` de `-0.0020`.
- **Subajuste:** Sí existen indicios de capacidad explicativa limitada o subajuste frente a la complejidad real del fenómeno: tanto en entrenamiento (`R² = 0.3520`) como en prueba (`R² = 0.3915`), un modelo estrictamente lineal explica menos de la mitad de la variabilidad de `quality`, dejando un error promedio de `MAE = 0.5041` puntos.

- **Capacidad de generalización observada:** La brecha entre entrenamiento y prueba es reducida (`MAE`: `0.5107` vs `0.5041`; `RMSE`: `0.6585` vs `0.6565`), lo que muestra estabilidad al evaluar datos no vistos. Además, frente al modelo de referencia que predice siempre la media (`RMSE = 0.8443`, `R² = -0.0063`), la regresión lineal múltiple reduce el RMSE en `0.1878` puntos e incrementa `R²` en `+0.3978`. No obstante, que entrenamiento y prueba sean similares no implica por sí solo alta precisión: el error sigue siendo moderado y requiere interpretar sus limitaciones.

## 8. Limitaciones del modelo

### Limitaciones observadas directamente en los resultados
1. **Concentración en calidades 5 y 6:** En prueba, el `81.99%` de los casos corresponde a notas `5` y `6`, provocando predicciones conservadoras hacia valores intermedios.
2. **Escasez en puntuaciones extremas:** Hay pocos vinos con calidad `3` (`4` en prueba), `4` (`11` en prueba) y `8` (`3` en prueba), donde los errores (`MAE` y `RMSE`) resultan más elevados.
3. **Capacidad explicativa moderada:** Un `R²` de `0.3915` indica que una parte importante de la variación en la calidad sensorial no queda capturada por una combinación lineal simple de las variables disponibles.

### Limitaciones potenciales y estructurales del dataset
4. **Subjetividad de la variable objetivo:** La calificación `quality` proviene de evaluaciones sensoriales discretas realizadas por catadores humanos.
5. **Posibles relaciones no lineales:** Algunos efectos fisicoquímicos sobre la calidad percibida pueden seguir comportamientos no lineales que `LinearRegression` no modela directamente.
6. **Colinealidad entre características:** La correlación entre variables independientes (observada en la Fase 1) puede influir en la estabilidad individual de los coeficientes.
7. **Generalización poblacional:** Los resultados son aplicables a vinos con características similares al vino tinto *Vinho Verde* representado en el dataset.

## 9. Conclusiones preliminares

- El modelo de Regresión Lineal Múltiple entrenado ofrece un desempeño **moderado y consistente** entre entrenamiento y prueba, mejorando claramente frente al modelo base de media constante (`DummyRegressor`).
- Su utilidad principal es capturar tendencias lineales globales (como el aporte positivo del alcohol y negativo de la acidez volátil), pero presenta errores considerables en calidades extremas y no debe considerarse un predictor exacto para decisiones críticas sin explorar en trabajos futuros relaciones no lineales o mayor equilibrio en los datos.
