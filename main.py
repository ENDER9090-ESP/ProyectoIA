"""Programa principal del proyecto de Inteligencia Artificial (Fases 1 a 5).

Ofrece un menú interactivo en consola para ejecutar de forma individual,
por fase o secuencial todas las actividades implementadas:
- Fase 1: Exploración, limpieza, identificación de X e y, estadísticas y gráficas.
- Fase 2: Partición entrenamiento/prueba (80/20) y escalado con StandardScaler.
- Fase 3: Entrenamiento del modelo de Regresión Lineal Múltiple e inspección de coeficientes.
- Fase 4: Evaluación del rendimiento (MSE, RMSE, MAE, R²), residuos, referencias y resumen.
- Fase 5: Generación del reporte técnico académico completo en PDF y Markdown.
"""

from Analisis_Datos import generar_graficas, realizar_analisis_estadistico
from Entrenamiento_Modelo import entrenar_modelo, mostrar_coeficientes_modelo
from Escalado_Datos import escalar_datos
from Evaluacion_Modelo import (
    ejecutar_analisis_residuos,
    ejecutar_comparacion_modelo_base,
    ejecutar_comparacion_train_test,
    ejecutar_graficas_evaluacion,
    ejecutar_rendimiento_por_calidad,
    evaluar_metricas_principales,
    evaluar_modelo,
    mostrar_valores_reales_y_predicciones,
)
from Experimentacion_Complementaria import ejecutar_experimentacion_complementaria
from Exploracion_Datos import explorar_dataset
from Generar_Reporte import generar_reporte_tecnico
from Identificar_X_y_Y import identificar_variables
from Limpieza_Datos import limpiar_dataset
from Particion_Datos import particionar_datos
from utilidades import (
    RUTA_DATASET_LIMPIO,
    datos_escalados_actualizados,
    datos_particion_actualizados,
    existe_dataset_limpio,
    existe_modelo_entrenado,
    existen_datos_escalados,
    existen_datos_particion,
    imprimir_encabezado,
    modelo_entrenado_actualizado,
)

# Evitar que ventanas interactivas de matplotlib bloqueen el menú en consola;
# todas las gráficas se guardan automáticamente en PNG a 300 DPI en resultados/graficas/.
MOSTRAR_GRAFICAS_EN_MENU = False


def mostrar_menu() -> None:
    """Imprime en consola el menú principal organizado por fases."""
    print(
        """
==================================================
       PROYECTO FINAL DE INTELIGENCIA ARTIFICIAL
       Prediccion de la calidad del vino tinto
==================================================

FASE 1: EXPLORACION Y LIMPIEZA

1. Explorar dataset original
2. Limpiar dataset
3. Identificar variables X y y
4. Realizar analisis estadistico
5. Generar graficas exploratorias
6. Ejecutar fase 1 completa

FASE 2: PREPARACION DE LOS DATOS

7. Separar datos en entrenamiento y prueba
8. Aplicar escalado de los datos
9. Ejecutar fase 2 completa

FASE 3: ENTRENAMIENTO DEL MODELO

10. Entrenar modelo de regresion lineal multiple
11. Mostrar coeficientes del modelo

FASE 4: EVALUACION DEL RENDIMIENTO

12. Evaluar modelo mediante MSE, RMSE, MAE y R2
13. Mostrar valores reales y predicciones
14. Analizar residuos
15. Comparar entrenamiento y prueba
16. Comparar con modelo de referencia
17. Analizar rendimiento por calidad
18. Generar graficas de evaluacion
19. Ejecutar fase 4 completa

EXPERIMENTACION COMPLEMENTARIA

20. Ejecutar experimentacion complementaria (Ridge CV 5-fold)

FASE 5: DOCUMENTACION TECNICA

21. Generar reporte tecnico PDF

EJECUCION GENERAL

22. Ejecutar todas las fases implementadas

0. Salir
"""
    )


def verificar_dataset_limpio() -> bool:
    """Comprueba que exista el dataset limpio antes de ejecutar etapas posteriores."""
    if existe_dataset_limpio():
        return True

    print(
        "\nAviso: Aun no existe el archivo limpio 'winequality-red-limpio.csv'.\n"
        "Ejecute primero la opcion 2 (Limpiar dataset) antes de continuar."
    )
    return False


def verificar_datos_particion() -> bool:
    """Comprueba que la partición train/test exista y esté actualizada."""
    if not existen_datos_particion():
        print(
            "\nAviso: Aun no existen los archivos de particion (X_train, X_test, y_train, y_test).\n"
            "Ejecute primero la opcion 7 (Separar datos en entrenamiento y prueba)."
        )
        return False

    if not datos_particion_actualizados():
        print(
            "\nAviso: Los archivos de particion quedaron desactualizados frente al dataset limpio.\n"
            "Ejecute nuevamente la opcion 7 para regenerar la particion."
        )
        return False

    return True


def verificar_datos_escalados() -> bool:
    """Comprueba que los datos escalados y el escalador existan y estén vigentes."""
    if not existen_datos_escalados():
        print(
            "\nAviso: Aun no existen los datos escalados ni el escalador guardado.\n"
            "Ejecute primero la opcion 8 (Aplicar escalado de los datos)."
        )
        return False

    if not datos_escalados_actualizados():
        print(
            "\nAviso: Los archivos escalados quedaron desactualizados tras una nueva particion.\n"
            "Ejecute nuevamente la opcion 8 (Aplicar escalado de los datos)."
        )
        return False

    return True


def verificar_modelo_entrenado() -> bool:
    """Comprueba que el modelo entrenado exista y corresponda al escalado actual."""
    if not existe_modelo_entrenado():
        print(
            "\nAviso: Aun no existe un modelo entrenado guardado en 'modelos/'.\n"
            "Ejecute primero las fases 2 y 3 (particion, escalado y entrenamiento)."
        )
        return False

    if not datos_particion_actualizados() or not datos_escalados_actualizados() or not modelo_entrenado_actualizado():
        print(
            "\nAviso: El modelo o los datos procesados estan desactualizados.\n"
            "Ejecute nuevamente las fases 2 y 3 (opciones 7, 8 y 10) antes de evaluar."
        )
        return False

    return True


def ejecutar_fase_1_completa() -> None:
    """Ejecuta en orden todas las actividades de la Fase 1."""
    imprimir_encabezado("EJECUCION COMPLETA DE LA FASE 1")
    explorar_dataset()
    limpiar_dataset()

    if not verificar_dataset_limpio():
        return

    identificar_variables(RUTA_DATASET_LIMPIO)
    realizar_analisis_estadistico(RUTA_DATASET_LIMPIO)
    generar_graficas(RUTA_DATASET_LIMPIO, mostrar=MOSTRAR_GRAFICAS_EN_MENU)
    print("\nLa Fase 1 finalizo correctamente.")


def ejecutar_fase_2_completa() -> None:
    """Ejecuta en orden partición 80/20 y escalado con StandardScaler (Fase 2)."""
    if not verificar_dataset_limpio():
        return

    imprimir_encabezado("EJECUCION COMPLETA DE LA FASE 2")
    particionar_datos(RUTA_DATASET_LIMPIO, mostrar_graficas=MOSTRAR_GRAFICAS_EN_MENU)
    escalar_datos(mostrar_graficas=MOSTRAR_GRAFICAS_EN_MENU)
    print("\nLa Fase 2 finalizo correctamente.")


def ejecutar_todas_las_fases() -> None:
    """Ejecuta secuencialmente las Fases 1, 2, 3, 4 y 5 sin cálculos repetidos."""
    imprimir_encabezado("EJECUCION COMPLETA DE LAS FASES 1 A 5")

    # Fase 1: Exploración, limpieza, identificación de X e y, estadísticas y gráficas
    explorar_dataset()
    limpiar_dataset()

    if not verificar_dataset_limpio():
        return

    identificar_variables(RUTA_DATASET_LIMPIO)
    realizar_analisis_estadistico(RUTA_DATASET_LIMPIO)
    generar_graficas(RUTA_DATASET_LIMPIO, mostrar=MOSTRAR_GRAFICAS_EN_MENU)

    # Fase 2: Partición entrenamiento/prueba y escalado
    particionar_datos(RUTA_DATASET_LIMPIO, mostrar_graficas=MOSTRAR_GRAFICAS_EN_MENU)
    escalar_datos(mostrar_graficas=MOSTRAR_GRAFICAS_EN_MENU)

    # Fase 3: Entrenamiento del modelo de Regresión Lineal Múltiple y almacenamiento
    entrenar_modelo(mostrar_graficas=MOSTRAR_GRAFICAS_EN_MENU)

    # Fase 4: Evaluación, tablas, gráficas y resumen de evaluación
    evaluar_modelo(mostrar_graficas=MOSTRAR_GRAFICAS_EN_MENU)

    # Experimentación complementaria: Ridge con validación cruzada 5-fold sobre X_train
    ejecutar_experimentacion_complementaria(mostrar_graficas=MOSTRAR_GRAFICAS_EN_MENU)

    # Fase 5: Generación del reporte técnico PDF y Markdown
    generar_reporte_tecnico()

    print(
        "\nTodas las fases implementadas (1 a 5 y experimentacion complementaria) "
        "finalizaron correctamente."
    )


def ejecutar_menu() -> None:
    """Inicia el ciclo principal del menú en consola con manejo básico de errores."""
    while True:
        mostrar_menu()
        try:
            opcion = input("Seleccione una opcion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo del programa.")
            break

        try:
            if opcion == "1":
                explorar_dataset()
            elif opcion == "2":
                limpiar_dataset()
            elif opcion == "3":
                if verificar_dataset_limpio():
                    identificar_variables(RUTA_DATASET_LIMPIO)
            elif opcion == "4":
                if verificar_dataset_limpio():
                    realizar_analisis_estadistico(RUTA_DATASET_LIMPIO)
            elif opcion == "5":
                if verificar_dataset_limpio():
                    generar_graficas(RUTA_DATASET_LIMPIO, mostrar=MOSTRAR_GRAFICAS_EN_MENU)
            elif opcion == "6":
                ejecutar_fase_1_completa()
            elif opcion == "7":
                if verificar_dataset_limpio():
                    particionar_datos(RUTA_DATASET_LIMPIO, mostrar_graficas=MOSTRAR_GRAFICAS_EN_MENU)
            elif opcion == "8":
                if verificar_datos_particion():
                    escalar_datos(mostrar_graficas=MOSTRAR_GRAFICAS_EN_MENU)
            elif opcion == "9":
                ejecutar_fase_2_completa()
            elif opcion == "10":
                if verificar_datos_escalados():
                    entrenar_modelo(mostrar_graficas=MOSTRAR_GRAFICAS_EN_MENU)
            elif opcion == "11":
                if verificar_modelo_entrenado():
                    mostrar_coeficientes_modelo()
            elif opcion == "12":
                if verificar_modelo_entrenado():
                    evaluar_metricas_principales()
            elif opcion == "13":
                if verificar_modelo_entrenado():
                    mostrar_valores_reales_y_predicciones()
            elif opcion == "14":
                if verificar_modelo_entrenado():
                    ejecutar_analisis_residuos()
            elif opcion == "15":
                if verificar_modelo_entrenado():
                    ejecutar_comparacion_train_test()
            elif opcion == "16":
                if verificar_modelo_entrenado():
                    ejecutar_comparacion_modelo_base()
            elif opcion == "17":
                if verificar_modelo_entrenado():
                    ejecutar_rendimiento_por_calidad()
            elif opcion == "18":
                if verificar_modelo_entrenado():
                    ejecutar_graficas_evaluacion(mostrar=MOSTRAR_GRAFICAS_EN_MENU)
            elif opcion == "19":
                if verificar_modelo_entrenado():
                    evaluar_modelo(mostrar_graficas=MOSTRAR_GRAFICAS_EN_MENU)
            elif opcion == "20":
                if verificar_datos_particion():
                    ejecutar_experimentacion_complementaria(
                        mostrar_graficas=MOSTRAR_GRAFICAS_EN_MENU
                    )
            elif opcion == "21":
                generar_reporte_tecnico()
            elif opcion == "22":
                ejecutar_todas_las_fases()
            elif opcion == "0":
                print("Saliendo del programa. Hasta luego.")
                break
            else:
                print(
                    f"\nOpcion invalida ('{opcion}'). "
                    "Por favor seleccione un numero del 0 al 22."
                )
        except (FileNotFoundError, KeyError, ValueError, OSError) as error:
            print(f"\nError al ejecutar la opcion seleccionada: {error}")
        except Exception as error:
            print(f"\nOcurrio un error inesperado: {error}")


if __name__ == "__main__":
    ejecutar_menu()
