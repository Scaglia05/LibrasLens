import cv2
import numpy as np
import os
import winsound
import mediapipe as mp
import pandas as pd

# --- CONFIGURAÇÕES DO PLANO 2.0 ---
DIR_DATASET = 'dataset_pontos'
# Landmarks são tão precisos que você precisa de BEM MENOS amostras que imagens
AMOSTRAS_POR_LETRA = 500  
COLUNAS = [f'p{i}_{e}' for i in range(21) for e in ['x', 'y', 'z']] + ['label']

# Inicialização MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def iniciar_coleta_pontos():
    if not os.path.exists(DIR_DATASET):
        os.makedirs(DIR_DATASET)

    cap = cv2.VideoCapture(0)
    alfabeto = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["ESPACO", "APAGAR"]
    indice_letra = 0
    dados_totais = [] 

    print("🚀 COLETOR DE LANDMARKS (FOCO EM GEOMETRIA)")
    print("Comandos: [C] Gravar | [P] Próxima Letra | [ESC] Sair e Salvar")

    while indice_letra < len(alfabeto):
        letra = alfabeto[indice_letra]
        contagem = 0
        gravando = False

        while contagem < AMOSTRAS_POR_LETRA:
            sucesso, frame = cap.read()
            if not sucesso: break
            
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)

            if res.multi_hand_landmarks:
                hand_lms = res.multi_hand_landmarks[0]
                # Desenha para você ver se o MediaPipe está estável
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                if gravando:
                    # EXTRAÇÃO MATEMÁTICA
                    lista_pontos = []
                    for lm in hand_lms.landmark:
                        # Pegamos X, Y e Z (coordenadas normalizadas)
                        lista_pontos.extend([lm.x, lm.y, lm.z])
                    
                    lista_pontos.append(letra) # A etiqueta/letra
                    dados_totais.append(lista_pontos)
                    contagem += 1

                    if contagem % 100 == 0: winsound.Beep(1000, 50)

            # Interface Visual
            cv2.rectangle(frame, (0, 0), (640, 80), (20, 20, 20), -1)
            status = "🔴 GRAVANDO" if gravando else "⚪ PAUSADO (Aperte 'C')"
            cv2.putText(frame, f"LETRA: {letra} | AMOSTRAS: {contagem}/{AMOSTRAS_POR_LETRA}", (15, 30), 1, 1.5, (255, 255, 0), 2)
            cv2.putText(frame, status, (15, 60), 1, 1.2, (0, 255, 0), 2)
            
            cv2.imshow("COLETOR DE LANDMARKS - IA 2026", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'): gravando = True
            elif key == ord('p'): break 
            elif key == 27: # ESC
                salvar_dados(dados_totais)
                cap.release()
                cv2.destroyAllWindows()
                return

        indice_letra += 1
        winsound.Beep(1500, 200)

    salvar_dados(dados_totais)
    cap.release()
    cv2.destroyAllWindows()

def salvar_dados(dados):
    if dados:
        df = pd.DataFrame(dados, columns=COLUNAS)
        caminho = os.path.join(DIR_DATASET, 'dados_libras.csv')
        df.to_csv(caminho, index=False)
        print(f"✅ Base de dados salva: {caminho}")

if __name__ == "__main__":
    iniciar_coleta_pontos()