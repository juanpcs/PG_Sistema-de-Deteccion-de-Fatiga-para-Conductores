import cv2
import mediapipe as mp
import time
import os
import numpy as np
import logging
from tensorflow.keras.models import load_model

#-----------------------------------------------------------------------------------#

 #####      Script para probar el funcionamiento del modelo en tiempo real      #####

#-----------------------------------------------------------------------------------#

# Configuración del logger
logging.basicConfig(
    filename='errores.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Se inicializa la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # fijar el ancho a 640 píxeles
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # fijar el alto a 480 píxeles

# Se inicializa mediapipe para la detección de caras
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.9)
mp_drawing = mp.solutions.drawing_utils

# Se define el tamaño estándar para las imágenes recortadas
standard_size = (128, 128)  

# Se carga nuestro modelo entrenado
cnn_model = load_model('ModeloEntrenadoV3.h5')

#Diccionario utilizado para castear la predicción del modelo (número) a su respectivo estado
numeros_a_estados = {
    0: 'Alert',  1: 'Microsleep',  2: 'Yawning'
}

while True:

    #Array para guardar los datos de la imagen del estado a predecir
    data_aux = []

    ret, frame = cap.read()

    if not ret:
        print("Error: No se pudo capturar la imagen")
        logging.error("Error al capturar la imagen", exc_info=True)
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(frame_rgb)

    if results.detections is not None:

        detection = results.detections[0]

        try:

            #Se obtienen las coordenadas de la cara detectada en el frame
            y_min = int(detection.location_data.relative_bounding_box.ymin * 480)
            height = int(detection.location_data.relative_bounding_box.height * 480)
            x_min = int(detection.location_data.relative_bounding_box.xmin * 640)
            width = int(detection.location_data.relative_bounding_box.width * 640)

            # Se recorta la imagen para obtener solo la región de la cara
            cropped_face = frame[y_min:y_min+height, x_min:x_min+width]

            # Se redimensiona la imagen recortada al tamaño estándar
            resized_face = cv2.resize(cropped_face, standard_size)

            # Se muestra la imagen recortada en una ventana separada
            #cv2.imshow('Cara Detectada', cropped_face)

            # Se muestra la imagen recortada en una ventana separada
            cv2.imshow('Cara redimensionada', resized_face)

            # Se normaliza el valor de los pixeles al rango [0, 1]
            img = resized_face / 255.0

            # Se añade la imagen a la lista
            data_aux.append(img)

            # Se realiza la predicción mediante el modelo entrenado (vector one hot)
            prediccion_one_hot = cnn_model.predict(np.array(data_aux))

            # Se castea la predicción en formato one hot a su representación númerica
            prediccion_numero = np.argmax(prediccion_one_hot)

            # Se castea la representación númerica a su correspondiente estado mediante la ayuda de un diccionario
            prediccion = numeros_a_estados[prediccion_numero]

            # Se dibuja el bounding box en la imagen
            cv2.rectangle(frame, (x_min, y_min), (x_min+width, y_min+height), (0, 255, 0), 2)

            # Se define la posición del texto (10 píxeles arriba del rectángulo)
            text_position = (x_min, y_min - 10)

            # Se dibuja el texto encima del rectángulo
            cv2.putText(frame,                     # Imagen sobre la que se dibuja el texto
                        prediccion,                # Texto a mostrar
                        text_position,             # Posición (x, y) donde se coloca el texto
                        cv2.FONT_HERSHEY_SIMPLEX,  # Fuente del texto
                        1,                         # Tamaño de la fuente
                        (0, 255, 0),               # Color del texto en formato BGR
                        2,                         # Grosor del texto
                        cv2.LINE_AA)               # Tipo de línea para una mejor apariencia
            
        except Exception as e:
            if "!ssize.empty()" in str(e):
                mensaje = "Error: Asegurarse que el rostro se encuentre dentro los limites de deteccion de la camara"
                print(mensaje)
                logging.error(mensaje, exc_info=True)
                #logging.warning(mensaje, exc_info=True)
            else:
                print("Error al procesar la cara detectada")
                logging.error("Error al procesar la cara detectada", exc_info=True)

    # Se muestra la imagen con las anotaciones en la ventana principal
    cv2.imshow('Sistema de deteccion de fatiga para conductores', frame)        

    # Se sale del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Ejecucion finalizada")
        break

cap.release()
cv2.destroyAllWindows()