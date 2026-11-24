import pandas as pd

# Cargar el dataset
datos = pd.read_csv("winequality-red.csv", sep=";")

# Explorar el contenido
print("Dimensiones:", datos.shape)
print("\nPrimeros registros:")
print(datos.head())

print("\nValores faltantes:")
print(datos.isnull().sum())

print("\nRegistros duplicados:")
print(datos.duplicated().sum())

# Definir las características (X)
X = datos.drop(columns=["quality"])

# Definir la variable objetivo (y)
# 0 = Calidad baja, 1 = Calidad buena
y = (datos["quality"] >= 6).astype(int)

print("\nCantidad de registros por categoría:")
print(y.value_counts())
