import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import time
from collections import deque

# --- 1. ARQUITETURA HÍBRIDA (Deve ser 100% igual ao script de treino) ---
class ModeloHibridoLibras(nn.Module):
    def __init__(self, num_classes):
        super(ModeloHibridoLibras, self).__init__()
        # Ramo CNN (Analisa a imagem capturada)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        # Ramo Bi-LSTM (Analisa o movimento/pontos)
        # AJUSTE: Se der erro de "l2", use num_layers=3. Se der erro de chaves faltando, use 2.
        self.lstm = nn.LSTM(63, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.bn_lstm = nn.BatchNorm1d(512)
        
        # Cabeça de Classificação
        self.classifier = nn.Sequential(
            nn.Linear(256 + 512, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, pts):
        x_img = self.cnn(img)
        out_lstm, _ = self.lstm(pts)
        x_pts = torch.mean(out_lstm, dim=1)
        x_pts = self.bn_lstm(x_pts)
        combined = torch.cat((x_img, x_pts), dim=1)
        return self.classifier(combined)

# --- 2. FUNÇÕES AUXILIARES ---
def normalizar_pontos(coords_brutas):
    coords = np.array(coords_brutas).reshape(21, 3)
    pulso = coords[0]
    # Normalização simples por distância da palma
    dist_ref = np.linalg.norm(coords[9] - pulso)
    if dist_ref == 0: dist_ref = 1.0
    return ((coords - pulso) / dist_ref).flatten()

@st.cache_resource
def carregar_recursos():
    classes = np.load('classes.npy', allow_pickle=True)
    modelo = ModeloHibridoLibras(len(classes))
    # Carrega os pesos salvos
    modelo.load_state_dict(torch.load('modelo_hibrido_final.pth', map_location='cpu'))
    modelo.eval()
    return modelo, classes

# --- 3. INTERFACE STREAMLIT ---
st.set_page_config(page_title="Libras Vision 2026", layout="wide")
st.title("🤟 Tradutor de Libras Híbrido (CNN + Bi-LSTM)")

if 'frase' not in st.session_state:
    st.session_state.frase = ""

col_cam, col_info = st.columns([2, 1])
run = col_cam.toggle('LIGAR TRADUTOR', value=False)
FRAME_WINDOW = col_cam.image([])

with col_info:
    st.subheader("📝 Tradução em Tempo Real")
    area_frase = st.empty()
    st.markdown("---")
    st.write("**Instruções:**")
    st.write("1. Coloque sua mão dentro do quadrado verde.")
    st.write("2. Mantenha o sinal por 1 segundo para confirmar a letra.")
    if st.button("Limpar Frase"):
        st.session_state.frase = ""
        st.rerun()

# --- 4. ENGINE DE EXECUÇÃO ---
if run:
    modelo, classes = carregar_recursos()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)
    
    # Buffer para a LSTM (precisa de 15 frames)
    buffer_pts = deque(maxlen=15)
    ultima_letra = ""
    tempo_confirmacao = time.time()

    while run:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Define a ROI (Região de Interesse) para a CNN
        roi_x, roi_y, roi_w, roi_h = w//2, 100, 250, 250
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

        # Processamento MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

            # 1. Coleta pontos para a LSTM
            pts_brutos = []
            for lm in lms.landmark:
                pts_brutos.extend([lm.x, lm.y, lm.z])
            buffer_pts.append(normalizar_pontos(pts_brutos))

            # 2. Coleta imagem para a CNN
            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            if roi.size > 0:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_resize = cv2.resize(roi_gray, (64, 64))
                
                # Prepara tensores
                img_tensor = torch.tensor(roi_resize, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
                
                # Só prediz se o buffer da LSTM estiver cheio (15 frames)
                if len(buffer_pts) == 15:
                    pts_np = np.array([list(buffer_pts)], dtype=np.float32)
                    pts_tensor = torch.from_numpy(pts_np)
                    
                    with torch.no_grad():
                        output = modelo(img_tensor, pts_tensor)
                        prob = F.softmax(output, dim=1)
                        conf, pred = torch.max(prob, 1)
                        letra_detectada = classes[pred.item()]

                        # Lógica de exibição e estabilização
                        if conf.item() > 0.7:
                            cv2.putText(frame, f"{letra_detectada} ({conf.item():.0%})", 
                                        (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                            if letra_detectada == ultima_letra:
                                if time.time() - tempo_confirmacao > 1.2:
                                    st.session_state.frase += letra_detectada
                                    tempo_confirmacao = time.time() # Reset do timer
                                    buffer_pts.clear() # Limpa para a próxima letra
                            else:
                                ultima_letra = letra_detectada
                                tempo_confirmacao = time.time()

        # Atualiza Interface
        area_frase.info(f"### {st.session_state.frase}")
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    cap.release()
else:
    st.info("Câmera desligada. Clique no botão acima para iniciar.")