import cv2
import mediapipe as mp
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

# Se define el tamaño estándar para las imágenes recortadas
standard_size = (64, 64)  

# Se cargan nuestros modelo entrenados
cnn_model = load_model('ModeloEntrenadoEyes.h5')
cnn_model2 = load_model('ModeloEntrenadoMouths.h5')


while True:

    #Arrays para guardar los datos de las imágenes del estado a predecir
    data_aux = []
    data_aux2 = []

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

            x_RE = int(detection.location_data.relative_keypoints[0].x * 640)
            y_RE = int(detection.location_data.relative_keypoints[0].y * 480)

            x_RET = int(detection.location_data.relative_keypoints[4].x * 640)
            y_RET = int(detection.location_data.relative_keypoints[4].y * 480)

            x_MC = int(detection.location_data.relative_keypoints[3].x * 640)
            y_MC = int(detection.location_data.relative_keypoints[3].y * 480)

            x_NT = int(detection.location_data.relative_keypoints[2].x * 640)
            y_NT = int(detection.location_data.relative_keypoints[2].y * 480)

            # Se recorta la imagen para obtener solo las regiones del ojo derecho y de la boca
            cropped_eye = frame[y_min:y_NT-15, x_RET+10:x_NT-10]
            cropped_mouth = frame[y_NT:y_min+height-15, x_min+20:x_min+width-20]

            # Se redimensionan las imágenes recortadas al tamaño estándar
            resized_eye = cv2.resize(cropped_eye, standard_size)
            resized_mouth = cv2.resize(cropped_mouth, standard_size)

            # CSe convierten las imágenes a escala de grises
            gray_eye = cv2.cvtColor(resized_eye, cv2.COLOR_BGR2GRAY)
            gray_mouth = cv2.cvtColor(resized_mouth, cv2.COLOR_BGR2GRAY)

            # Guarda las imágenes resultantes
            cv2.imwrite('eye_bn.jpg', gray_eye)
            cv2.imwrite('mouth_bn.jpg', gray_mouth)

            # Se normaliza el valor de los pixeles al rango [0, 1]
            img1 = gray_eye / 255.0
            img2 = gray_mouth / 255.0

            # Se añaden las imágenes a las listas
            data_aux.append(img1)
            data_aux2.append(img2)

            # Se obtienen las probabilidades dadas por los modelos
            prediccion_prob = cnn_model.predict(np.array(data_aux))
            prediccion_prob2 = cnn_model2.predict(np.array(data_aux2))

            # Se castean las probabilidades a su estado correspondiente
            prediccion = "Eye open" if prediccion_prob >= 0.5 else "Eye close"
            prediccion2 = "Yawn" if prediccion_prob2 >= 0.5 else "No yawn"

            # Se dibujan el bounding box del ojo derecho en la imagen
            cv2.rectangle(frame, (x_RET+10, y_min), (x_NT-10, y_NT-15), (0, 255, 0), 2)
            # Se dibuja el bounding box de la boca en la imagen
            cv2.rectangle(frame, (x_min+20, y_NT), (x_min+width-20, y_min+height-15), (0, 255, 0), 2)

            # Se definen las posiciones de los textos
            text_position = (x_min+10, y_min)
            text_position2 = (x_NT, y_NT-10)

            # Se dibujan los textos encima de los rectángulos
            cv2.putText(frame,                     # Imagen sobre la que se dibuja el texto
                        prediccion,                # Texto a mostrar
                        text_position,             # Posición (x, y) donde se coloca el texto
                        cv2.FONT_HERSHEY_SIMPLEX,  # Fuente del texto
                        1,                         # Tamaño de la fuente
                        (0, 255, 0),               # Color del texto en formato BGR
                        2,                         # Grosor del texto
                        cv2.LINE_AA)               # Tipo de línea para una mejor apariencia
            cv2.putText(frame,
                        prediccion2,
                        text_position2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)
            
        except Exception as e:
            if "!ssize.empty()" in str(e):
                mensaje = "Error: Asegurarse que el rostro se encuentre dentro los limites de deteccion de la camara"
                print(mensaje)
                logging.error(mensaje, exc_info=True)

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