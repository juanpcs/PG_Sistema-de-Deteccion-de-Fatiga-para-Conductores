import cv2
import mediapipe as mp
import time
import os

#-----------------------------------------------------------------------------------#

##### Script para probar la detección y captura de rostros utilizando Mediapipe #####

#-----------------------------------------------------------------------------------#

# Se inicializa la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # fijar el ancho a 640 píxeles
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # fijar el alto a 480 píxeles

# Se inicializa mediapipe para la detección de caras
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.9)
mp_drawing = mp.solutions.drawing_utils

# Se crea un directorio para guardar las fotos si no existe
output_dir = "Carpeta"
os.makedirs(output_dir, exist_ok=True)

# Contador para nombrar las imágenes
img_counter = 0

# Tiempo de inicio para controlar la captura de fotos
start_time = time.time()

# Se define el tamaño estándar para las imágenes recortadas
standard_size = (128, 128)  
    
while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: No se pudo capturar la imagen")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(frame_rgb)

    if results.detections is not None:
        for detection in results.detections:
                
            cv2.imshow("Frame", frame)

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
            cv2.imshow('Cara Detectada', cropped_face)

            # Se obtiene el tiempo actual
            current_time = time.time()

            # Se verifica si han pasado x segundos desde la última captura
            if current_time - start_time >= 0.5:
                # Se guarda la imagen recortada
                img_name = f"cara_stndr_{img_counter}.png"
                cv2.imwrite(os.path.join(output_dir, img_name), resized_face)
                print(f"Imagen estandarizada guardada como '{img_name}'")
                img_counter += 1
                        
                # Se actualiza el tiempo de inicio
                start_time = current_time

    # Se sale del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()