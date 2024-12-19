import cv2
import mediapipe as mp

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Función para calcular distancia euclidiana
def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

# Función para detectar sonrisas
def is_smiling(landmarks, img_width, img_height):
    # Obtener coordenadas de la boca usando índices específicos de MediaPipe
    left_corner = landmarks[61]   # Esquina izquierda de la boca
    right_corner = landmarks[291] # Esquina derecha de la boca
    upper_lip = landmarks[13]     # Parte superior del labio
    lower_lip = landmarks[14]     # Parte inferior del labio

    # Convertir puntos normalizados a píxeles
    left_corner = (int(left_corner.x * img_width), int(left_corner.y * img_height))
    right_corner = (int(right_corner.x * img_width), int(right_corner.y * img_height))
    upper_lip = (int(upper_lip.x * img_width), int(upper_lip.y * img_height))
    lower_lip = (int(lower_lip.x * img_width), int(lower_lip.y * img_height))

    # Calcular distancias
    mouth_width = euclidean_distance(left_corner, right_corner)
    mouth_height = euclidean_distance(upper_lip, lower_lip)

    # Relación de aspecto de la boca
    ratio = mouth_height / mouth_width

    # Detectar sonrisa si la relación supera un umbral
    return ratio > 0.05  # Este umbral puede ajustarse

# Inicializar la cámara
cap = cv2.VideoCapture(0)

print("Presiona 'q' para salir...")

while True:
    # Capturar el frame
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener dimensiones de la imagen
    img_height, img_width, _ = frame.shape

    # Convertir a RGB (requerido por MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar los landmarks faciales
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibujar landmarks en el rostro
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
            )

            # Detectar si hay una sonrisa
            if is_smiling(face_landmarks.landmark, img_width, img_height):
                cv2.putText(frame, "Sonrisa Detectada :)", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar el frame
    cv2.imshow("Deteccion de Sonrisa", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
