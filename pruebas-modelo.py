import numpy as np

import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from prettytable import PrettyTable
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

from tensorflow.keras.models import load_model

#-----------------------------------------------------------------------------------#

###### Script para realizar la evaluación del rendimiento del modelo entrenado ######

#-----------------------------------------------------------------------------------#

# Directorio raíz donde se encuentran las carpetas de cada letra
root_dir = 'Dataset-pruebas'

# Se carga nuestro modelo entrenado
cnn_model = load_model('ModeloEntrenadoMouths.h5')

cnn_model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', 'Precision', 'Recall', 'AUC'])


# Listas para almacenar las imágenes y sus etiquetas
data = []
labels = []

#-----------------------------------------------------------------------------------#

def guardar_imagenes(data, n): 
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, int(n/2), i + 1)
        plt.imshow(data[i], cmap='gray', vmin=0, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig('Dataset de prueba.png')

#-----------------------------------------------------------------------------------#

# Recorre cada carpeta (cada estado) dentro del directorio raíz
for label in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, label)
    
    # Asegurarse de que solo se procesen carpetas
    if os.path.isdir(folder_path):
        # Recorre cada imagen dentro de la carpeta
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            # Cargar la imagen usando OpenCV
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Se normaliza el valor de los pixeles al rango [0, 1]
            img = img / 255.0
            # Añadir la imagen y su etiqueta a las listas
            data.append(img)
            labels.append(label)

# Conversión de las listas en arrays de NumPy para su uso en ML
data = np.array(data)
labels = np.array(labels)

print(f"Total de imágenes cargadas: {len(data)}")
print(f"Etiquetas asociadas: {set(labels)}")
print(data.shape)

# Conversión de etiquetas de texto a números
label_encoder = LabelEncoder()
label_encoder.fit(labels)
labels1 = label_encoder.transform(labels)

print(labels1[:10])

guardar_imagenes(data, 20)

#-----------------------------------------------------------------------------------#

print("Evaluando modelo...");

loss, accuracy, precision, recall, auc = cnn_model.evaluate(data, labels1, verbose=0)

# Crear una tabla
table = PrettyTable()
table.field_names = ["Metric", "Value"]
table.add_row(["Loss", "{:.4f}".format(loss)])
table.add_row(["Accuracy", "{:.4f}".format(accuracy)])
table.add_row(["Precision", "{:.4f}".format(precision)])
table.add_row(["Recall", "{:.4f}".format(recall)])
table.add_row(["ROC AUC Score", "{:.4f}".format(auc)])
# Mostrar la tabla
print("\n"+str(table)+"\n")

# Obtener las predicciones del modelo
y_pred = cnn_model.predict(data)
# Se castean las probabilidades a su correspondencia númerica
y_pred_classes = (y_pred >= 0.5).astype(int)


# Calcular precision, recall, f1-score y roc auc
accuracy2 = accuracy_score(labels1, y_pred_classes)
precision2 = precision_score(labels1, y_pred_classes, average='weighted')
recall2 = recall_score(labels1, y_pred_classes, average='weighted')
f1 = f1_score(labels1, y_pred_classes, average='weighted')
roc_auc = roc_auc_score(labels1, y_pred_classes, multi_class='ovr')

# Crear una tabla
table = PrettyTable()
table.field_names = ["Metric", "Value"]
table.add_row(["Accuracy", "{:.4f}".format(accuracy2)])
table.add_row(["Precision", "{:.4f}".format(precision2)])
table.add_row(["Recall", "{:.4f}".format(recall2)])
table.add_row(["F1 Score", "{:.4f}".format(f1)])
table.add_row(["ROC AUC Score", "{:.4f}".format(roc_auc)])
# Mostrar la tabla
print("\n"+str(table)+"\n")

print("Modelo evaluado!");

print(classification_report(labels1, y_pred_classes, zero_division=0))
