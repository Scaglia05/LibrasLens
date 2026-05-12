import cv2
import pandas as pd
import numpy as np
import os
import winsound
import mediapipe as mp
import time
from datetime import datetime

# --- CONFIGURAÇÕES ---
ARQUIVO_CSV = 'banco_dados_libras.csv'
DIR_IMAGENS = 'dataset_imagens/Train'
AMOSTRAS_POR_LETRA = 1700 
IMG_SIZE = 64

# Inicialização MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, 
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def iniciar_coleta():
    if not os.path.exists(DIR_IMAGENS):
        os.makedirs(DIR_IMAGENS)

    cap = cv2.VideoCapture(0)
    alfabeto = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["ESPACO", "APAGAR"]
    indice_letra = 0
    
    print("🚀 COLETOR 2026: Interface Topo + Navegação")

    while indice_letra < len(alfabeto):
        letra = alfabeto[indice_letra]
        pasta_letra = os.path.join(DIR_IMAGENS, letra)
        os.makedirs(pasta_letra, exist_ok=True)

        contagem = len(os.listdir(pasta_letra))
        gravando = False
        dados_sessao = []

        while True: # Loop de controle por letra
            sucesso, frame = cap.read()
            if not sucesso: break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # --- ROI CENTRALIZADA ---
            x1, y1, x2, y2 = w//2 - 100, h//2 - 100, w//2 + 100, h//2 + 100
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            roi = frame[y1:y2, x1:x2]

            # Processamento MediaPipe
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.multi_hand_landmarks:
                hand_lms = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                if gravando and contagem < AMOSTRAS_POR_LETRA:
                    # NOME DO ARQUIVO: [letra_sobreposição_datadodia]
                    data_atual = datetime.now().strftime("%Y-%m-%d")
                    timestamp = int(time.time()*1000)
                    img_name = f"{letra}_{timestamp}_{data_atual}.png"
                    
                    save_path = os.path.join(pasta_letra, img_name)
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(save_path, cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE)))

                    # Coordenadas Normalizadas (Pulso Zero)
                    landmarks = hand_lms.landmark
                    p_ref = landmarks[0]
                    coords_norm = []
                    for lm in landmarks:
                        coords_norm.extend([lm.x - p_ref.x, lm.y - p_ref.y, lm.z - p_ref.z])
                    
                    dados_sessao.append([letra] + coords_norm)
                    contagem += 1
                    
                    if contagem % 10 == 0:
                        winsound.Beep(1200, 30)
                    
                    if contagem >= AMOSTRAS_POR_LETRA:
                        gravando = False
                        winsound.Beep(1500, 400)

            # --- UI NO TOPO (COMANDOS VISÍVEIS) ---
            # Tarja preta no topo para leitura clara
            cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
            
            status_txt = "GRAVANDO..." if gravando else "PAUSADO (C para iniciar)"
            cor_status = (0, 0, 255) if gravando else (0, 255, 0)
            
            cv2.putText(frame, f"LETRA ATUAL: {letra} | {contagem}/{AMOSTRAS_POR_LETRA}", (10, 30), 1, 1.5, (255, 255, 0), 2)
            cv2.putText(frame, f"STATUS: {status_txt}", (10, 60), 1, 1.2, cor_status, 2)
            cv2.putText(frame, "TECLAS: [C] Gravar | [P] Proxima | [V] Voltar | [ESC] Sair", (10, 90), 1, 1, (255, 255, 255), 1)

            cv2.imshow("COLETOR HIBRIDO 2026", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'): # Iniciar/Pausar
                gravando = not gravando
                winsound.Beep(1000 if gravando else 500, 200)
            
            elif key == ord('p'): # Próxima Letra
                if dados_sessao:
                    salvar_dados(dados_sessao)
                indice_letra += 1
                winsound.Beep(800, 100)
                break # Sai do loop interno (vai para a próxima letra)
                
            elif key == ord('v'): # Voltar Letra
                if indice_letra > 0:
                    indice_letra -= 1
                    winsound.Beep(400, 100)
                    break # Sai do loop interno (volta a letra)

            elif key == 27: # ESC para fechar
                if dados_sessao: salvar_dados(dados_sessao)
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

def salvar_dados(dados):
    if not dados: return
    df = pd.DataFrame(dados)
    header = not os.path.exists(ARQUIVO_CSV)
    df.to_csv(ARQUIVO_CSV, mode='a', index=False, header=header)
    print(f"✅ Sessão salva no CSV.")

if __name__ == "__main__":
    iniciar_coleta()