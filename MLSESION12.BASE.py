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

## Cargar el dataset desde los datos del NPS2025
estudiantes = pd.read_csv('DATANPS2025.csv', sep=";")

## Tomar los datos de los estudiantes del campus virtual y analizar la matriz de correlación
estudiantes_DISTANCIA = estudiantes[estudiantes['X_CAMPUS'] == 'LIM'].copy()

## Definir las variables independientes (X) y la variable dependiente (y)
X = estudiantes_DISTANCIA[['SATIF_ACAD', 'SATISF_SERV', 'SATISF_EXP', 'SATISF_REG_ACAD', 'SATISF_ACT_SOST', 'SATISF_APY_LOGRO_ACAD', 'SATISF_INF_TEC', 'SATISF_CAMP_VIRT', 'SATISF_COM_IMAG']]
y = estudiantes_DISTANCIA['DETRACTOR']
XCopia = X.copy()

## Definir el conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(XCopia, y, test_size=0.30, random_state=42)

## Modelo de clasificación utilizando MLPClassifier (una red neuronal multicapa)
model = MLPClassifier(random_state=42, max_iter=1000) 

## Entrenar el modelo y evaluar su rendimiento
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

## Evaluar el rendimiento del modelo utilizando métricas de clasificación
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo MLP: {accuracy:.3f}')
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

## Visualizar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - MLP')
plt.show()

## Guardar el modelo entrenado
joblib.dump(model, 'ModeloEntrenado.pkl')
input("\n✓ Modelo entrenado y guardado exitosamente en ModeloEntrenado.pkl\nPresiona Enter para continuar...")



##==============================================================================
## EJEMPLO: Cargar el modelo guardado y realizar una predicción
##==============================================================================

print("\n" + "="*80)
print("EJEMPLO DE USO: Cargando el modelo guardado para hacer una predicción")
print("="*80 + "\n")

## Cargar el modelo entrenado
modelo_cargado = joblib.load('ModeloEntrenado.pkl')
print("✓ Modelo cargado exitosamente desde ModeloEntrenado.pkl\n")
input("Presiona Enter para realizar la predicción...")

## Crear un ejemplo con valores ficticios para hacer una consulta
## Los valores deben estar en el mismo orden que las características del modelo
## ['SATIF_ACAD', 'SATISF_SERV', 'SATISF_EXP', 'SATISF_REG_ACAD', 'SATISF_ACT_SOST', 
##  'SATISF_APY_LOGRO_ACAD', 'SATISF_INF_TEC', 'SATISF_CAMP_VIRT', 'SATISF_COM_IMAG']

##ejemplo_estudiante = [[4.5, 4.0, 4.2, 4.1, 3.8, 4.3, 4.0, 4.4, 4.1]]
ejemplo_estudiante = [[1, 0, 1, 2, 1, 2, 1, 1, 1]]

print("Datos ficticios del estudiante para predicción:")
print("SATIF_ACAD: 4.5")
print("SATISF_SERV: 4.0")
print("SATISF_EXP: 4.2")
print("SATISF_REG_ACAD: 4.1")
print("SATISF_ACT_SOST: 3.8")
print("SATISF_APY_LOGRO_ACAD: 4.3")
print("SATISF_INF_TEC: 4.0")
print("SATISF_CAMP_VIRT: 4.4")
print("SATISF_COM_IMAG: 4.1\n")

## Realizar la predicción
prediccion = modelo_cargado.predict(ejemplo_estudiante)
probabilidades = modelo_cargado.predict_proba(ejemplo_estudiante)
input("Presiona Enter para mostrar el resultado de la clasificación...")    

print("RESULTADO DE LA CLASIFICACIÓN:")
print("-" * 80)
print(f"Clasificación predicha: {prediccion[0]}")
print(f"Etiquetas disponibles: {modelo_cargado.classes_}")
print(f"\nProbabilidades por clase:")
for i, clase in enumerate(modelo_cargado.classes_):
    print(f"  Clase {clase}: {probabilidades[0][i]:.4f} ({probabilidades[0][i]*100:.2f}%)")
print("-" * 80)

