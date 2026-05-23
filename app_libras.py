import sys
import os
import time
import base64
from types import ModuleType
from gtts import gTTS
import streamlit as st
import cv2
import numpy as np
from tensorflow import keras

# 1. BLINDAGEM DE AMBIENTE
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

try:
    import google.protobuf.runtime_version as rv
except ImportError:
    mock_rv = ModuleType('runtime_version')
    mock_rv.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
    sys.modules['google.protobuf.runtime_version'] = mock_rv

# --- 1. FUNÇÕES AUXILIARES ---

@st.cache_resource
def carregar_recursos():
    import tensorflow as tf
    # ATENÇÃO: Atualize para o nome do seu novo arquivo .h5 de pontos!
    # modelo = tf.keras.models.load_model('models/modelo_libras_pontos_16_05_2026_17_32.h5') 
    modelo = keras.models.load_model('models/modelo_libras_pontos_16_05_2026_17_32.h5')
    classes = np.load('classes.npy', allow_pickle=True)
    return modelo, classes

@st.cache_resource
def configurar_mediapipe():
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    return mp_hands, hands, mp_drawing

def falar_frase(texto):
    if texto.strip():
        try:
            tts = gTTS(text=texto, lang='pt-br')
            tts.save("frase.mp3")
            with open("frase.mp3", "rb") as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()
                md = f"""<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
                return md
        except: return ""
    return ""

# --- 2. INTERFACE STREAMLIT ---
st.set_page_config(page_title="Libras Vision 2026", layout="wide")
st.title("🤟 Tradutor Geométrico Inteligente")

if 'frase' not in st.session_state:
    st.session_state.frase = ""

col_cam, col_info = st.columns([2, 1])
run = col_cam.toggle('LIGAR SISTEMA', value=False)
FRAME_WINDOW = col_cam.image([])
placeholder_audio = st.empty() 

with col_info:
    st.subheader("📝 Texto Montado")
    area_frase = st.empty()
    st.markdown("---")
    if st.button("Limpar Tudo"):
        st.session_state.frase = ""
        st.rerun()

# --- 3. ENGINE ---
if run:
    modelo, classes = carregar_recursos()
    mp_hands, hands, mp_drawing = configurar_mediapipe()
    
    cap = cv2.VideoCapture(0)
    ultima_letra = ""
    timer_confirmacao = 0
    tempo_ultima_mao = time.time()
    ja_leu = True 

    while run:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            tempo_ultima_mao = time.time() 
            ja_leu = False
            
            hand_lms = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # --- LÓGICA DE EXTRAÇÃO E PREDIÇÃO (IGUAL AO TREINO) ---
            lista_pontos = []
            for lm in hand_lms.landmark:
                lista_pontos.extend([lm.x, lm.y, lm.z])
            
            # Centralização no pulso (Ponto 0)
            pontos_array = np.array(lista_pontos).reshape(21, 3)
            pulso = pontos_array[0]
            pontos_centralizados = pontos_array - pulso
            input_ia = pontos_centralizados.flatten().reshape(1, 63)

            preds = modelo.predict(input_ia, verbose=0)
            idx = np.argmax(preds)
            confianca = preds[0][idx]
            letra = classes[idx]

            if confianca > 0.98: # Exigimos confiança alta pela precisão do modelo
                cv2.putText(frame, f"LETRA: {letra} ({confianca:.2%})", (50, h-50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                if letra == ultima_letra:
                    timer_confirmacao += 1
                    if timer_confirmacao >= 15:
                        if letra == "ESPACO": st.session_state.frase += " "
                        elif letra == "APAGAR": st.session_state.frase = st.session_state.frase[:-1]
                        else: st.session_state.frase += letra
                        
                        timer_confirmacao = 0 
                        ultima_letra = "" # Reset para a próxima letra
                else:
                    ultima_letra = letra
                    timer_confirmacao = 0
        else:
            tempo_ocioso = time.time() - tempo_ultima_mao
            if tempo_ocioso > 7 and not ja_leu and st.session_state.frase:
                html_audio = falar_frase(st.session_state.frase)
                placeholder_audio.markdown(html_audio, unsafe_allow_html=True)
                ja_leu = True
            
            if not ja_leu and st.session_state.frase:
                cv2.putText(frame, f"Lendo em: {int(7 - tempo_ocioso)}s", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        area_frase.info(f"### {st.session_state.frase}")
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    cap.release()
else:
    st.info("Sistema desligado. Ative o seletor acima para começar.")