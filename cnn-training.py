import pandas as pd
import numpy as np

import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from prettytable import PrettyTable

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard

#-----------------------------------------------------------------------------------#

### Script para realizar el entrenamiento del modelo (red neuronal convolucional) ###

#-----------------------------------------------------------------------------------#

# Directorio raíz donde se encuentran las carpetas de cada estado
root_dir = 'Dataset'

# Listas para almacenar las imágenes y sus etiquetas
data = []
labels = []

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
#print(data[0])

# Se divide el dataset en sets de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print(" ---------- ")
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

# Conversión de etiquetas de texto a números
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

print(y_train[:10])

#-----------------------------------------------------------------------------------#
def guardar_imagenes(data, n): 
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, int(n/2), i + 1)
        plt.imshow(data[i], cmap='gray', vmin=0, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig('prueba.png')

guardar_imagenes(X_train, 20)
#-----------------------------------------------------------------------------------#

# Modelo de CNN con Drop Out

cnn_model = tf.keras.models.Sequential([

    # Capas convolucionales
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)), #Capa de entrada convolucional, con 32 kernel de 3x3
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2), #Capa de pooling 2x2

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # Capa convolucional con 64 kernel
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2), #Capa de pooling 2x2

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), # Capa convolucional con 128 kernel
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2), #Capa de pooling 2x2

    # Capas densas de clasificación
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'), # Capa densa con 256 neuronas
    tf.keras.layers.Dense(1, activation='sigmoid')  # Capa densa de salida

])

# Se compila el modelo
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

# Se imprime un resumen del modelo
cnn_model.summary()

#-----------------------------------------------------------------------------------#

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor="val_loss",          # Monitor validation loss for early stopping
    patience=10,                 # Number of epochs with no improvement after which training will be stopped
    min_delta=0.001,             # The minimum amount of change in the monitored metric to be considered an improvement
    restore_best_weights=True    # Restore the model weights from the epoch with the best monitored value
)

#-----------------------------------------------------------------------------------#

# Entrenamiento del modelo

print("Entrenando modelo...");

BoardCNN = TensorBoard(log_dir = "C:/Users/juanp/OneDrive/Documentos/GitHub/PG_Sistema-de-Deteccion-de-Fatiga-para-Conductores/Boards")
history = cnn_model.fit(X_train,
                        y_train,
                        verbose=1,
                        epochs=100,
                        validation_split=0.2,  # Usar una parte de los datos de entrenamiento como set de validación
                        batch_size=64,
                        callbacks=[early_stopping, BoardCNN]
                        #steps_per_epoch=int(np.ceil((len(X_train)*0.8) / float(32))),
                        ##validation_steps=int(np.ceil((len(X_train)*0.2) / float(32)))
                        )

print("Modelo entrenado!");

#-----------------------------------------------------------------------------------#

print("Evaluando modelo...");

loss, accuracy, precision, recall, auc = cnn_model.evaluate(X_test, y_test, verbose=0)

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

print("Modelo evaluado!");

#-----------------------------------------------------------------------------------#

print("Guardando modelo...");

cnn_model.save('ModeloEntrenado.h5')
cnn_model.save_weights('PesosModelo.weights.h5')

print("Modelo guardado!");

#tensorboard --logdir="C:/Users/juanp/OneDrive/Documentos/GitHub/LESCO-VI/Boards"
