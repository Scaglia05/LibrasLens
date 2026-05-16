import cv2
import numpy as np
import os
import winsound
import mediapipe as mp
import time
from datetime import datetime

# --- CONFIGURAÇÕES PADRÃO (COMPATÍVEL COM LUCAS LACERDA) ---
DIR_DATASET = 'dataset_imagens'
AMOSTRAS_POR_LETRA = 1700 
IMG_SIZE = 64 

# Inicialização MediaPipe (Apenas para auxílio visual)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def iniciar_coleta():
    # Criar estrutura de pastas limpa
    if not os.path.exists(DIR_DATASET):
        os.makedirs(os.path.join(DIR_DATASET, 'training'))

    cap = cv2.VideoCapture(0)
    # Alfabeto seguindo a ordem do dicionário que você possui
    alfabeto = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["ESPACO", "APAGAR"]
    indice_letra = 0
    
    print("🚀 COLETOR FOCADO EM CNN (ESTILO LUCAS LACERDA)")
    print("Objetivo: Eliminar o vício no 'J' coletando apenas imagens estáticas.")

    while indice_letra < len(alfabeto):
        letra = alfabeto[indice_letra]
        pasta_letra = os.path.join(DIR_DATASET, 'training', letra)
        os.makedirs(pasta_letra, exist_ok=True)

        contagem = len(os.listdir(pasta_letra))
        gravando = False

        while True:
            sucesso, frame = cap.read()
            if not sucesso: break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # --- ROI (REGIÃO DE INTERESSE) ---
            # Definindo o quadrado de 200x200 centralizado
            x1, y1, x2, y2 = w//2 - 100, h//2 - 100, w//2 + 100, h//2 + 100
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Recorte da mão
            roi = frame[y1:y2, x1:x2]

            # MediaPipe processa o frame total para desenhar o guia
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.multi_hand_landmarks:
                # Desenha os pontos apenas para você saber que a mão está centrada
                hand_lms = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                if gravando and contagem < AMOSTRAS_POR_LETRA:
                    # Gerar nome único para a imagem
                    timestamp = int(time.time()*1000)
                    img_name = f"{letra}_{timestamp}.png"
                    save_path = os.path.join(pasta_letra, img_name)
                    
                    # --- PROCESSAMENTO CRÍTICO ---
                    # 1. Redimensiona para o tamanho que a CNN espera (64x64)
                    img_final = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                    
                    # 2. Salva a imagem colorida (RGB) conforme o Script 5 do Lucas
                    cv2.imwrite(save_path, img_final)
                    
                    contagem += 1
                    
                    # Feedback sonoro rápido
                    if contagem % 50 == 0:
                        winsound.Beep(1000, 50)
                    
                    if contagem >= AMOSTRAS_POR_LETRA:
                        gravando = False
                        winsound.Beep(1500, 300)

            # --- INTERFACE VISUAL ---
            cv2.rectangle(frame, (0, 0), (w, 110), (20, 20, 20), -1)
            status_txt = "GRAVANDO..." if gravando else "PAUSADO"
            cor_status = (0, 0, 255) if gravando else (0, 255, 0)
            
            cv2.putText(frame, f"LETRA: {letra} | FOTOS: {contagem}/{AMOSTRAS_POR_LETRA}", (15, 35), 1, 1.8, (255, 255, 0), 2)
            cv2.putText(frame, f"STATUS: {status_txt}", (15, 70), 1, 1.5, cor_status, 2)
            cv2.putText(frame, "[C] Iniciar | [P] Proxima | [V] Voltar | [ESC] Sair", (15, 100), 1, 1, (255, 255, 255), 1)

            cv2.imshow("COLETOR CORRIGIDO - SEM VÍCIO NO J", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'): # Iniciar/Parar
                gravando = not gravando
            elif key == ord('p'): # Próxima letra
                indice_letra += 1
                break 
            elif key == ord('v'): # Voltar letra
                if indice_letra > 0:
                    indice_letra -= 1
                    break 
            elif key == 27: # Sair
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_coleta()