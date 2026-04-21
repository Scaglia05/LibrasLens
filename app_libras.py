import streamlit as st
import cv2
import numpy as np
import torch
import time
import os
import torch.nn as nn
from collections import deque, Counter
from gtts import gTTS
import pygame
from threading import Thread

# --- IMPORTAÇÃO RESILIENTE DO CONTEXTO DO STREAMLIT ---
try:
    from streamlit.runtime.scriptrunner import add_script_run_context
except ImportError:
    try:
        from streamlit.runtime.scriptrunner.script_run_context import add_script_run_context
    except ImportError:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_context as add_script_run_context
        except ImportError:
            def add_script_run_context(t): return t

# --- IMPORTAÇÃO MEDIAPIPE ---
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

# --- INICIALIZAÇÃO DE ESTADO ---
if 'frase' not in st.session_state:
    st.session_state.frase = ""

# --- CONFIGURAÇÕES DO MODELO ---
HIDDEN_SIZE = 128
NUM_LAYERS = 2
SEQ_LENGTH = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModeloLibrasLSTM(nn.Module):
    def __init__(self, num_classes):
        super(ModeloLibrasLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=63, hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LAYERS, batch_first=True, dropout=0.3)
        self.bn = nn.BatchNorm1d(HIDDEN_SIZE)
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.bn(out)
        return self.fc(out)

def normalizar_frame(coords_brutas):
    coords = np.array(coords_brutas).reshape(21, 3)
    pulso = coords[0]
    ponto_ref = coords[9] 
    dist_palma = np.linalg.norm(ponto_ref - pulso)
    if dist_palma == 0: dist_palma = 1.0
    norm_coords = (coords - pulso) / dist_palma
    return norm_coords.flatten()

@st.cache_resource
def carregar_recursos():
    classes = np.load('classes_encoder.npy', allow_pickle=True)
    modelo = ModeloLibrasLSTM(len(classes))
    modelo.load_state_dict(torch.load('modelo_libras_lstm.pth', map_location=DEVICE))
    modelo.to(DEVICE)
    modelo.eval()
    return modelo, classes

def falar_texto(texto):
    if not texto.strip(): return
    try:
        tts = gTTS(text=texto, lang='pt')
        filename = "fala_temp.mp3"
        tts.save(filename)
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): continue
        pygame.mixer.music.unload()
        pygame.mixer.quit()
        if os.path.exists(filename): os.remove(filename)
    except: pass

# --- INTERFACE ---
st.set_page_config(page_title="IA LIBRAS 2026", layout="wide")
st.title("🤟 Tradutor LIBRAS Inteligente v2")

col_cam, col_info = st.columns([2, 1])

# O Toggle serve como o interruptor principal
run = col_cam.toggle('Ligar Câmera', value=False)
FRAME_WINDOW = col_cam.image([])

with col_info:
    st.subheader("Painel de Controle")
    display_letra = st.empty()
    display_frase = st.empty()
    
    st.write("---")
    st.write("**Status do Áudio Automático**")
    timer_progresso = st.progress(0)
    timer_texto = st.empty()
    
    if st.button("Limpar Tudo (Reset)"):
        st.session_state.frase = ""
        st.rerun()

# --- LOOP PRINCIPAL ---
if run:
    modelo, classes = carregar_recursos()
    # Usamos o context manager 'with' para garantir que o MediaPipe feche corretamente
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        cap = cv2.VideoCapture(0)
        
        buffer_coords = deque(maxlen=SEQ_LENGTH)
        historico_predicoes = deque(maxlen=20)
        ultimo_registro_tempo = time.time()
        tempo_parado = time.time()

        while run:
            success, frame = cap.read()
            if not success:
                st.error("Falha ao acessar a câmera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                tempo_parado = time.time() 
                for hand_landmarks in result.multi_hand_landmarks:
                    raw_coords = []
                    for lm in hand_landmarks.landmark:
                        raw_coords.extend([lm.x, lm.y, lm.z])
                    
                    buffer_coords.append(normalizar_frame(raw_coords))
                    frequencia = 0
                    letra_final = ""

                    if len(buffer_coords) == SEQ_LENGTH:
                        input_tensor = torch.tensor([list(buffer_coords)], dtype=torch.float32).to(DEVICE)
                        with torch.no_grad():
                            output = modelo(input_tensor)
                            pred = torch.argmax(output).item()
                            historico_predicoes.append(classes[pred])

                        contagem = Counter(historico_predicoes).most_common(1)
                        letra_final = contagem[0][0]
                        frequencia = contagem[0][1]

                    cor_ponto = (255, 255, 255)
                    cor_conexao = (200, 200, 200)

                    if frequencia > 8:
                        cor_ponto = (0, 255, 255)
                        cor_conexao = (0, 255, 255)
                        display_letra.warning(f"Estabilizando: {letra_final}")
                    
                    if frequencia > 15:
                        cor_ponto = (0, 255, 0)
                        cor_conexao = (0, 255, 0)
                        display_letra.info(f"Confirmado: **{letra_final}**")

                        if time.time() - ultimo_registro_tempo > 2.5:
                            if letra_final == 'ESPACO': st.session_state.frase += " "
                            elif letra_final == 'APAGAR': st.session_state.frase = st.session_state.frase[:-1]
                            else: st.session_state.frase += letra_final
                            
                            buffer_coords.clear()
                            historico_predicoes.clear()
                            ultimo_registro_tempo = time.time()

                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=cor_ponto, thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=cor_conexao, thickness=2, circle_radius=2)
                    )
            else:
                display_letra.error("Mão fora de alcance")

            tempo_sem_sinal = time.time() - tempo_parado
            if st.session_state.frase.strip():
                progresso = min(1.0, tempo_sem_sinal / 10)
                timer_progresso.progress(progresso)
                tempo_restante = max(0, 10 - int(tempo_sem_sinal))
                timer_texto.write(f"🎤 Áudio em: {tempo_restante}s")

                if tempo_sem_sinal > 10.0:
                    texto_fala = st.session_state.frase
                    st.session_state.frase = ""
                    t = Thread(target=falar_texto, args=(texto_fala,))
                    add_script_run_context(t)
                    t.start()
                    tempo_parado = time.time()
            else:
                timer_progresso.progress(0)
                timer_texto.write("Aguardando sinal para iniciar...")

            display_frase.success(f"**Frase Atual:** {st.session_state.frase}")
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # --- MELHORIA: VERIFICAÇÃO DE PARADA ---
            # Se o usuário desligar o toggle, saímos do loop imediatamente
            if not run:
                break

        # Limpeza final ao sair do loop
        cap.release()
        cv2.destroyAllWindows()
    
    # Força o Streamlit a parar a execução desta parte do script
    st.rerun() 

else:
    FRAME_WINDOW.info("Câmera desligada. Ative o interruptor para começar.")
    # Garante que o áudio e processos paralelos parem
    pygame.mixer.quit() if pygame.mixer.get_init() else None