import cv2
import pandas as pd
import numpy as np
import os
import sys
import winsound
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
from sklearn.neighbors import KNeighborsClassifier

# --- CONFIGURAÇÕES ---
ARQUIVO_BANCO_DADOS = 'banco_dados_libras.csv'
ARQUIVO_MESTRE = 'dataset_mestre.csv' 
AMOSTRAS_POR_SESSAO = 50
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def normalizar_coordenadas(coords_brutas):
    if len(coords_brutas) < 63: return coords_brutas
    px, py, pz = coords_brutas[0], coords_brutas[1], coords_brutas[2]
    norm = []
    for i in range(0, len(coords_brutas), 3):
        norm.extend([coords_brutas[i] - px, coords_brutas[i+1] - py, coords_brutas[i+2] - pz])
    return norm

def treinar_juiz():
    if not os.path.exists(ARQUIVO_MESTRE):
        print("⚠️ [AVISO] dataset_mestre.csv não encontrado.")
        return None
    try:
        df = pd.read_csv(ARQUIVO_MESTRE)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df_limpo = df.dropna()
        if df_limpo.empty: df_limpo = df.fillna(0)
        X_dados_brutos = df_limpo.drop('label', axis=1).iloc[:, :63].values.astype(np.float32)
        y = df_limpo['label'].values
        X_norm = np.array([normalizar_coordenadas(linha) for linha in X_dados_brutos])
        
        knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
        knn.fit(X_norm, y)
        print(f"✅ JUIZ PRONTO!")
        return knn
    except Exception as e:
        print(f"❌ Erro no Juiz: {e}")
        return None

def iniciar_coleta():
    juiz = treinar_juiz()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    alfabeto_completo = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["ESPACO", "APAGAR"]
    cache_dados = {}
    
    if os.path.exists(ARQUIVO_BANCO_DADOS):
        try:
            df_existente = pd.read_csv(ARQUIVO_BANCO_DADOS)
            for l_ex in df_existente['label'].unique():
                cache_dados[l_ex] = df_existente[df_existente['label'] == l_ex].values.tolist()
        except: pass

    indice_letra = 0
    while indice_letra < len(alfabeto_completo):
        letra = alfabeto_completo[indice_letra]
        amostras_letra = cache_dados.get(letra, [])
        precisa_decidir = len(amostras_letra) >= AMOSTRAS_POR_SESSAO
        
        gravando = False
        modo_forcado = False 

        while True:
            sucesso, frame = cap.read()
            if not sucesso: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            if precisa_decidir:
                cv2.rectangle(frame, (50, h//2 - 60), (w-50, h//2 + 60), (0, 0, 0), -1)
                cv2.putText(frame, f"SINAL '{letra}' JA POSSUI DADOS.", (70, h//2 - 10), 1, 1.2, (255, 255, 255), 2)
                cv2.putText(frame, "[M] MANTER | [R] RESETAR", (70, h//2 + 30), 1, 1.0, (0, 255, 255), 1)
                cv2.imshow("COLETOR LIBRAS", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('m'):
                    indice_letra += 1; precisa_decidir = False; break
                if key == ord('r'):
                    amostras_letra = []; precisa_decidir = False
                if key == 27: sys.exit()
                continue

            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            status_txt = f"SINAL: {letra} | {len(amostras_letra)}/{AMOSTRAS_POR_SESSAO}"
            cor_status = (255, 255, 255)
            juiz_ok = False 

            if res.multi_hand_landmarks:
                hand_lms = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                bruto = []
                for lm in hand_lms.landmark: bruto.extend([lm.x, lm.y, lm.z])
                norm = normalizar_coordenadas(bruto)
                
                if juiz:
                    pred = juiz.predict([norm])[0]
                    if pred == letra:
                        juiz_ok = True
                        cor_status = (0, 255, 0)
                        status_txt = f"JUIZ: OK! ({letra})"
                    else:
                        juiz_ok = False
                        cor_status = (0, 0, 255)
                        status_txt = f"JUIZ: ERRADO ({pred})"

                # Lógica de gravação
                if (gravando and juiz_ok) or modo_forcado:
                    if len(amostras_letra) < AMOSTRAS_POR_SESSAO:
                        amostras_letra.append([letra] + bruto)
                        cor_status = (255, 165, 0) if gravando else (0, 255, 255)
                        status_txt = f"GRAVANDO: {len(amostras_letra)}/{AMOSTRAS_POR_SESSAO}"

            # --- UI ---
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(frame, status_txt, (10, 40), 1, 1.5, cor_status, 2)
            cv2.rectangle(frame, (0, h-40), (w, h), (40, 40, 40), -1)
            cv2.putText(frame, "[C] GRAVAR | [V] FORCAR | [N] PROXIMA | [B] VOLTAR", (10, h-12), 1, 1.0, (255, 255, 255), 1)
            
            if len(amostras_letra) >= AMOSTRAS_POR_SESSAO:
                cv2.rectangle(frame, (w//4, h//2-25), (3*w//4, h//2+25), (0, 200, 0), -1)
                cv2.putText(frame, "COMPLETO! APERTE [N]", (w//4+20, h//2+10), 1, 1.2, (0,0,0), 2)
                gravando = modo_forcado = False

            cv2.imshow("COLETOR LIBRAS", frame)
            key = cv2.waitKey(1) & 0xFF

            # --- TECLAS DE COMANDO ---
            if key == ord('c') and juiz_ok: 
                gravando = not gravando
                modo_forcado = False
            
            if key == ord('v'): 
                modo_forcado = not modo_forcado
                gravando = False
                if modo_forcado: winsound.Beep(800, 100)

            if key == ord('n'):
                # LISTA DE EXCEÇÃO: Sinais que o Juiz não precisa validar
                sinais_livres = ['X', 'J', 'Z', 'K', 'ESPACO', 'APAGAR']
                
                if len(amostras_letra) >= AMOSTRAS_POR_SESSAO:
                    if juiz_ok or letra in sinais_livres:
                        cache_dados[letra] = amostras_letra
                        indice_letra += 1
                        winsound.Beep(1200, 100)
                        break
                    else:
                        print(f"⚠️ Juiz não validou {letra}. Use [V] para gravar.")
                winsound.Beep(600, 200) 

            if key == ord('b'): 
                indice_letra = max(0, indice_letra - 1); break
            
            if key == 27: indice_letra = 9999; break

    cap.release()
    cv2.destroyAllWindows()
    
    # --- SALVAMENTO FINAL ---
    if cache_dados:
        dados_finais = []
        for l in alfabeto_completo:
            if l in cache_dados: dados_finais.extend(cache_dados[l])
        if dados_finais:
            colunas = ['label'] + [f'{e}{i}' for i in range(21) for e in ['x', 'y', 'z']]
            pd.DataFrame(dados_finais, columns=colunas).to_csv(ARQUIVO_BANCO_DADOS, index=False)
            print(f"✨ {len(dados_finais)} amostras salvas em {ARQUIVO_BANCO_DADOS}!")
    sys.exit()

if __name__ == "__main__":
    iniciar_coleta()