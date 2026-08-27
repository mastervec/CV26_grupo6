import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import plotly.express as px
import matplotlib.pyplot as plt 
import time
import math

# Função auxiliar para desenhar landmarks em um frame
def check_posture(landmarks):
    if not landmarks:
        return "Nenhuma pose detectada"

    for lm in landmarks:
        print(f"Landmark: {lm}")

        # Indices chave 
        nose = lm[0]  # Nariz
        left_shoulder = lm[11]  # Ombro esquerdo
        right_shoulder = lm[12]  # Ombro direito
        left_hip = lm[23]  # Quadril esquerdo
        right_hip = lm[24]  # Quadril direito
        mid_shoulder = get_midpoint(left_shoulder,right_shoulder)
        # Verificar se os ombros estão alinhados horizontalmente
        nose_angle = angle_between_points(mid_shoulder,(nose.x,nose.y))
        if abs(left_shoulder.x - right_shoulder.x) < 0.05  and (115<nose_angle<140 or 40<nose_angle<75):
            return f"Postura Correta {nose_angle}"
        else:
            return "Postura Incorreta"

def get_midpoint(p1, p2):
    # Unpack x and y coordinates
    x1 = p1.x
    y1 = p1.y
    x2 = p2.x
    y2 = p2.y
    
    # Calculate the average of x and y
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    return (mid_x, mid_y)

def angle_between_points(p1, p2):
    # Calculate differences
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # Calculate angle in radians
    # Note: 'dy' must be the first argument
    radians = math.atan2(dy, dx)
    
    # Convert to degrees
    degrees = math.degrees(radians)
    
    return radians, degrees

def process_image_mp_mask(path, output_path="resultado.jpg"):
    image = cv2.imread(path)
    if image is None:
        print(f"Erro ao carregar '{path}'")
        return
    h, w, _ = image.shape

    # Configuração do MediaPipe - AGORA HABILITANDO SEGMENTAÇÃO!
    model_path = 'pose_landmarker_lite.task'
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, model_path)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        # --- ATENÇÃO: ESSA É A CHAVE ---
        output_segmentation_masks=True 
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # Tenta detecção na original
    image_rgb_orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image_orig = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb_orig)
    result = detector.detect(mp_image_orig)

    if result.pose_landmarks and result.segmentation_masks:
        # 1. Obtém a Máscara de Segmentação
        segmentation_mask = result.segmentation_masks[0].numpy_view()
        
        # 2. Converte para uma máscara binária (Transformação de Formato)
        # O MediaPipe gera valores flutuantes entre 0 (fundo) e 1 (corpo).
        # Normalizamos e binarizamos.
        mask = (segmentation_mask > 0.1).astype(np.uint8) * 255 # Ajuste suave de threshold

        # 3. PDI: Aplica a Máscara na Original
        # A máscara é apenas [h, w], precisamos do 3º canal.
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        image_segmented = cv2.bitwise_and(image, mask_3ch) # Aplica a máscara na imagem original

        annotated_image = image.copy()
        annotated_image = draw_keypoints(annotated_image, result.pose_landmarks)
        
        # Painel: Original | Máscara | Analisada
        combined = np.hstack((image, mask_3ch, annotated_image))
        
        cv2.imwrite(output_path, combined)
        #plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        print(f"Sucesso Total! Resultado salvo em {output_path}")

        # retorna os lanndimarks para análise posterior (se necessário)
        return result.pose_landmarks, mask
    else:
        print("Infelizmente, nenhuma pose foi detectada.")
        return None, None

def draw_keypoints(image, pose_landmarks_list):
    if not pose_landmarks_list:
        return image

    landmarks = pose_landmarks_list[0] # Assuming one person

    h, w, _ = image.shape

    # Using integer indices directly as mediapipe.solutions is not available
    pontos = {
        "nose": 0,  # Corresponds to mp.solutions.pose.PoseLandmark.NOSE
        "left_shoulder": 11, # Corresponds to mp.solutions.pose.PoseLandmark.LEFT_SHOULDER
        "right_shoulder": 12, # Corresponds to mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
        "left_hip": 23, # Corresponds to mp.solutions.pose.PoseLandmark.LEFT_HIP
        "right_hip": 24 # Corresponds to mp.solutions.pose.PoseLandmark.RIGHT_HIP
    }

    coords = {}

    for nome, idx in pontos.items():
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        coords[nome] = (x, y)

        cv2.circle(image, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(image, nome, (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # linhas principais
    cv2.line(image, coords["left_shoulder"], coords["left_hip"], (255,0,0), 2)
    cv2.line(image, coords["right_shoulder"], coords["right_hip"], (255,0,0), 2)

    ombro_medio = (
        int((coords["left_shoulder"][0] + coords["right_shoulder"][0]) / 2),
        int((coords["left_shoulder"][1] + coords["right_shoulder"][1]) / 2)
    )

    cv2.line(image, coords["nose"], ombro_medio, (0,0,255), 2)

    return image

def annotate_frame_with_landmarks(frame, landmarks, mask=None):
    annotated = frame.copy()
    if landmarks:
        annotated = draw_keypoints(annotated, landmarks)
    if mask is not None:
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        annotated = cv2.addWeighted(annotated, 0.75, mask_rgb, 0.25, 0)
    return annotated
    
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar webcam")
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

last_analysis_time = 0.0
analysis_interval = 1.0  # analisar postura a cada 1 segundo
last_landmarks = None
last_mask = None
posture_message = "Aguardando análise..."

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame")
        break

    current_time = time.time()
    if current_time - last_analysis_time >= analysis_interval:
        last_analysis_time = current_time
        cv2.imwrite("frame_entrada.png", frame)
        landmarks, mask = process_image_mp_mask("frame_entrada.png", "frame_saida.png")

        if landmarks:
            posture_message = check_posture(landmarks)
            last_landmarks = landmarks
            last_mask = mask
        else:
            posture_message = "Nenhuma pose detectada no frame atual"
            last_landmarks = None
            last_mask = None

        print(f"[{time.strftime('%H:%M:%S')}] {posture_message}")

    annotated_frame = annotate_frame_with_landmarks(frame, last_landmarks, last_mask)
    combined = np.hstack((frame, annotated_frame))

    if out is None:
        height, width = combined.shape[:2]
        out = cv2.VideoWriter('video_analisado.avi', fourcc, 20.0, (width, height))

    out.write(combined)

    cv2.putText(combined, posture_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.imshow('Vídeo - Pressione "q" para interromper gravação', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Gravação interrompida pelo usuário.")
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
