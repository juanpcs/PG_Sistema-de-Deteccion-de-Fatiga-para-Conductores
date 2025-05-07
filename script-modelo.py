import cv2
import numpy as np
import subprocess
import time
import mediapipe as mp
import RPi.GPIO as GPIO
from tensorflow.keras.models import load_model
from utils import *

# ---------- Configuración del buzzer ----------
BUZZER_PIN = 17  # GPIO17 (pin físico 11)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
# Crear señal PWM por software en GPIO17 a 440 Hz
pwm = GPIO.PWM(BUZZER_PIN, 440)

# --------------- Inicialización ---------------
# Se inicializa mediapipe para la detección de caras
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.9)

# Se define el tamaño estándar para las imágenes recortadas
standard_size = (64, 64)  

# Se cargan nuestros modelo entrenados
cnn_model = load_model('ModeloEntrenadoEyes.h5')
cnn_model2 = load_model('ModeloEntrenadoMouths.h5')

eye_states = []
mouth_states = []

print("Iniciando monitoreo...")

start_time = time.time()

try:
    while True:

        subprocess.run([
            "rpicam-still",                 # Comando principal para capturar imágenes
            "-o", "captura.png",            # Archivo de salida
            "--width", "640",               # Ancho de la imagen
            "--height", "480",              # Alto de la imagen
            "--nopreview",                  # Sin vista previa
            "-t", "500",                    # Tiempo antes de la captura (en milisegundos)
            "--vflip",                      # flip vertical a la cámara
            "--hflip"                       # flip horizontal a la cámara
        ])


        #Arrays para guardar los datos de las imágenes a predecir
        data_aux = []
        data_aux2 = []

        frame = cv2.imread("captura.png")

        if frame is None:
            print("No se pudo leer la imagen capturada.\n")
            continue

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_detection.process(frame_rgb)

            if results.detections is not None:

                detection = results.detections[0]

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

                # Se convierten las imágenes a escala de grises
                gray_eye = cv2.cvtColor(resized_eye, cv2.COLOR_BGR2GRAY)
                gray_mouth = cv2.cvtColor(resized_mouth, cv2.COLOR_BGR2GRAY)

                # Se normaliza el valor de los pixeles al rango [0, 1]
                img1 = gray_eye / 255.0
                img2 = gray_mouth / 255.0

                # Se añaden las imágenes a las listas
                data_aux.append(img1)
                data_aux2.append(img2)

                # Se obtienen las probabilidades dadas por los modelos
                prediction_prob = cnn_model.predict(np.array(data_aux))
                prediction_prob2 = cnn_model2.predict(np.array(data_aux2))

                # Se castean las probabilidades a su estado correspondiente
                prediction = "Ojo abierto" if prediction_prob >= 0.5 else "Ojo cerrado"
                prediction2 = "Bostezo" if prediction_prob2 >= 0.5 else "Sin bostezar"

                # Se castean las probabilidades a su correspondencia númerica
                prediction_num = 1 if prediction_prob >= 0.5 else 0
                prediction_num2 = 1 if prediction_prob2 >= 0.5 else 0

                eye_states.append(prediction_num)
                mouth_states.append(prediction_num2)

                print(f"\nEstado de ojo detectado: {prediction}")
                print(f'Estados del ojo: {eye_states} \n')

                print(f"Estado de boca detectado: {prediction2}")
                print(f'Estados de la boca: {mouth_states}\n')

                if len(eye_states) > 60:
                    eye_states.pop(0)
                if len(mouth_states) > 60:
                    mouth_states.pop(0)

                alert = False

                if count_yawns(mouth_states) >= 3:
                    print("Alerta: Bostezos frecuentes\n")
                    alert = True
                    mouth_states.clear()

                if detect_microsleep(eye_states):
                    print("Alerta: Posible microsueño\n")
                    alert = True
                    eye_states.clear()

                if alert:
                    print("Ejecutando alerta sonora de Mario Bros")
                    play_mario_theme(pwm)

            else:
                print("No se detectó ninguna cara.\n")

        except Exception as e:
            if "!ssize.empty()" in str(e):
                print("Error: Asegurarse que el rostro se encuentre dentro los límites de detección de la cámara\n")
            else:
                print("Error al procesar la cara detectada:", str(e))
        
        current_time = time.time()
        exec_time = current_time - start_time
        print(f"Tiempo de ejecucion: {exec_time} s\n")
        start_time = current_time

except KeyboardInterrupt:
    print("Interrupción del usuario. Finalizando...\n")

finally:
    pwm.stop()
    GPIO.cleanup()
