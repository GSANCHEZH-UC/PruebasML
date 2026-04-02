
import pandas as pd
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, confusion_matrix
import seaborn as sns
import shap
from sklearn.neural_network import MLPClassifier
import joblib

import utilidadesML as ml

## Cargar los datos del estudiante desde el caso de prueba
estudiantes = pd.read_csv('CASOS-MASIVO.csv', sep=";")
input("\n✓ Datos de los estudiantes cargados exitosamente desde CASOS-MASIVO.csv\nPresiona Enter para continuar...")   

## Definir las variables independientes (X) y la variable dependiente (y) para el caso de prueba
X = estudiantes.drop(columns=['DETRACTOR'])
Y = estudiantes['DETRACTOR']

print("✓ Datos de los estudiantes cargados exitosamente desde CASOS-MASIVO.csv\n")
print("Datos de los estudiantes:")
print(X)
print("\nVariable dependiente (DETRACTOR):")
print(Y)

input("\nPresiona Enter para cargar el modelo entrenado y realizar las predicciones...")

## Cargar el modelo entrenado
modelo_cargado = joblib.load('ModeloEntrenado.pkl')
print("✓ Modelo cargado exitosamente desde ModeloEntrenado.pkl\n")
input("Presiona Enter para realizar las predicciones...")

## Realizar una predicción utilizando el modelo cargado
Y = modelo_cargado.predict(X)
print("Predicciones para los estudiantes:")
print(Y)
input("\nPresiona Enter para mostrar las probabilidades por clase para cada estudiante...")

## Obtener las probabilidades de cada clase para los estudiantes
probabilidades = modelo_cargado.predict_proba(X)    
print(f"\nProbabilidades por clase para cada estudiante:")
for i in range(len(X)):
    print(f"Estudiante {i+1}:")
    for j, clase in enumerate(modelo_cargado.classes_):
        print(f"  Clase {clase}: {probabilidades[i][j]:.4f} ({probabilidades[i][j]*100:.2f}%)")
    print("-" * 80)

input("\nPresiona Enter para agregar las predicciones al DataFrame y guardar los resultados...")

## Agregar la predicción de DETRACTOR al DataFrame de estudiantes
estudiantes['DETRACTOR'] = Y
print("\nDatos de los estudiantes con predicción:") 
print(estudiantes)
input("\nPresiona Enter para guardar los resultados en CASOS-MASIVO-PREDICCIONES.csv...")

## Guardar los resultados en un nuevo archivo CSV
estudiantes.to_csv('CASOS-MASIVO-PREDICCIONES.csv', index=False, sep=";")
print("\n✓ Predicciones guardadas exitosamente en CASOS-MASIVO-PREDICCIONES.csv")
input("Presiona Enter para finalizar el proceso...")

## Cambios Hechos por GSANCHEZH-1967
