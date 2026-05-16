import sys
import os
import time
import base64
from types import ModuleType
from gtts import gTTS
import streamlit as st
import cv2
import numpy as np

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
    # Certifique-se que o caminho do modelo está correto
    modelo = tf.keras.models.load_model('models/modelo_libras_cnn_13_05_2026_01_19.h5') 
    classes = np.load('classes.npy', allow_pickle=True)
    return modelo, classes

@st.cache_resource
def configurar_mediapipe():
    try:
        import mediapipe.python.solutions.hands as mp_hands
        import mediapipe.python.solutions.drawing_utils as mp_drawing
    except ImportError:
        from mediapipe.solutions import hands as mp_hands
        from mediapipe.solutions import drawing_utils as mp_drawing
    
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    return mp_hands, hands, mp_drawing

def falar_frase(texto):
    """Gera áudio e retorna a tag HTML para reprodução automática"""
    if texto.strip():
        try:
            tts = gTTS(text=texto, lang='pt-br')
            tts.save("frase.mp3")
            with open("frase.mp3", "rb") as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()
                md = f"""
                    <audio autoplay="true">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>
                    """
                return md
        except Exception as e:
            return f""
    return ""

# --- 2. INTERFACE STREAMLIT ---
st.set_page_config(page_title="Libras Vision 2026", layout="wide")
st.title("🤟 Tradutor Inteligente com Voz")

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
        # ROI centralizada
        x1, y1, x2, y2 = w//2 - 125, h//2 - 125, w//2 + 125, h//2 + 125
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            tempo_ultima_mao = time.time() 
            ja_leu = False
            
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            roi = frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi_resize = cv2.resize(roi_rgb, (64, 64)) / 255.0
                img_tensor = np.expand_dims(roi_resize, axis=0)

                preds = modelo.predict(img_tensor, verbose=0)
                idx = np.argmax(preds)
                letra = classes[idx]

                if preds[0][idx] > 0.95:
                    cv2.putText(frame, f"SINAL: {letra}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                    if letra == ultima_letra:
                        timer_confirmacao += 1
                        # Barra de progresso visual
                        cv2.rectangle(frame, (x1, y2+10), (x1 + (timer_confirmacao*16), y2+20), (0, 255, 255), -1)
                        
                        if timer_confirmacao >= 15:
                            if letra == "ESPACO": st.session_state.frase += " "
                            elif letra == "APAGAR": st.session_state.frase = st.session_state.frase[:-1]
                            else: st.session_state.frase += letra
                            timer_confirmacao = 0 
                            ultima_letra = "" # Evita repetir a mesma letra sem tirar a mão
                    else:
                        ultima_letra = letra
                        timer_confirmacao = 0
        else:
            # LÓGICA DE LEITURA (10 segundos sem mão)
            tempo_ocioso = time.time() - tempo_ultima_mao
            if tempo_ocioso > 10 and not ja_leu and st.session_state.frase:
                html_audio = falar_frase(st.session_state.frase)
                placeholder_audio.markdown(html_audio, unsafe_allow_html=True)
                ja_leu = True
            
            # Contador regressivo visual
            if not ja_leu and st.session_state.frase:
                cv2.putText(frame, f"Lendo em: {int(10 - tempo_ocioso)}s", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Atualização da UI (dentro do While, mas fora do if hand)
        area_frase.info(f"### {st.session_state.frase}")
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    cap.release()
else:
    st.info("Sistema desligado. Ative o seletor acima para começar.")