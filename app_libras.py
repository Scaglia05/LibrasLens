import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import torch
import time
import os
import torch.nn as nn
from collections import deque
from gtts import gTTS
import pygame
from threading import Thread

# --- CONFIGURAÇÕES DO MODELO ---
HIDDEN_SIZE = 128
NUM_LAYERS = 2
SEQ_LENGTH = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModeloLibrasLSTM(nn.Module):
    def __init__(self, num_classes):
        super(ModeloLibrasLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=63, hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LAYERS, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(HIDDEN_SIZE, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


@st.cache_resource
def carregar_modelo_e_classes():
    classes = np.load('classes_encoder.npy', allow_pickle=True)
    modelo = ModeloLibrasLSTM(len(classes))
    modelo.load_state_dict(torch.load(
        'modelo_libras_lstm.pth', map_location=DEVICE))
    modelo.to(DEVICE)
    modelo.eval()
    return modelo, classes


def falar_texto(texto):
    if not texto.strip():
        return
    try:
        tts = gTTS(text=texto, lang='pt')
        temp_audio = f"fala_temp_{int(time.time())}.mp3"
        tts.save(temp_audio)

        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue

        pygame.mixer.music.unload()  # Libera o arquivo
        pygame.mixer.quit()

        if os.path.exists(temp_audio):
            os.remove(temp_audio)
    except Exception as e:
        print(f"Erro no áudio: {e}")


# --- INTERFACE ---
st.set_page_config(page_title="IA LIBRAS 2026", layout="wide")
st.title("🤟 Tradução de LIBRAS em Tempo Real")

col_cam, col_info = st.columns([2, 1])
run = col_cam.checkbox('Ativar Webcam')
FRAME_WINDOW = col_cam.image([])

with col_info:
    st.subheader("Tradução")
    letra_atual_display = st.empty()
    texto_acumulado = st.empty()
    st.markdown("---")
    st.caption("Dica: Mantenha o sinal estável por 1 segundo.")

if run:
    if not os.path.exists('modelo_libras_lstm.pth'):
        st.error("Modelo não encontrado! Treine a rede neural primeiro.")
    else:
        modelo, classes = carregar_modelo_e_classes()
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        cap = cv2.VideoCapture(0)

        frase = ""
        ultimo_tempo_letra = time.time()
        buffer_frames = deque(maxlen=SEQ_LENGTH)

        while run:
            sucesso, frame = cap.read()
            if not sucesso:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = hands.process(rgb)

            if resultado.multi_hand_landmarks:
                for hand_landmarks in resultado.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])

                    buffer_frames.append(coords)

                    if len(buffer_frames) == SEQ_LENGTH:
                        tensor_seq = torch.tensor(
                            [list(buffer_frames)], dtype=torch.float32).to(DEVICE)
                        with torch.no_grad():
                            previsao = modelo(tensor_seq)
                            indice = torch.argmax(previsao).item()
                            letra_detectada = classes[indice]

                        letra_atual_display.info(
                            f"Detectado: **{letra_detectada}**")

                        # Debounce de 2 segundos para registro na frase
                        if time.time() - ultimo_tempo_letra > 2:
                            if letra_detectada == 'ESPACO':
                                frase += " "
                            elif letra_detectada == 'APAGAR':
                                frase = frase[:-1]
                            else:
                                frase += letra_detectada

                            ultimo_tempo_letra = time.time()
                            texto_acumulado.success(f"Frase: {frase}")
                            buffer_frames.clear()  # Reset após detecção bem sucedida
            else:
                # Se a mão sumir, limpa o buffer para não misturar gestos
                buffer_frames.clear()

            # Fala a frase se houver 5 segundos de inatividade
            if frase.strip() != "" and (time.time() - ultimo_tempo_letra > 5):
                Thread(target=falar_texto, args=(frase,)).start()
                frase = ""
                # Mantém o texto visual por um tempo ou limpa
                # texto_acumulado.empty()

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
