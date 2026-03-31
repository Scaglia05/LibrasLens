import cv2
import pandas as pd
import numpy as np
import os
import sys
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
from sklearn.neighbors import KNeighborsClassifier

# --- CONFIGURAÇÕES ---
ARQUIVO_BANCO_DADOS = 'banco_dados_libras.csv'
ARQUIVO_MESTRE = 'dataset_mestre.csv' 
AMOSTRAS_POR_SESSAO = 50
MARGEM_QUALIDADE_MINIMA = 0.80 

alfabeto_completo = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["ESPACO", "APAGAR"]

# --- INICIALIZAÇÃO MEDIAPIPE ---
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# =====================================================================
# SISTEMA DE VALIDAÇÃO (O JUIZ)
# =====================================================================

def normalizar_coordenadas(coords_brutas):
    """Normaliza as coordenadas baseando-se no pulso (ponto 0)"""
    if len(coords_brutas) < 63: return coords_brutas
    px, py, pz = coords_brutas[0], coords_brutas[1], coords_brutas[2]
    norm = []
    for i in range(0, len(coords_brutas), 3):
        norm.extend([coords_brutas[i] - px, coords_brutas[i+1] - py, coords_brutas[i+2] - pz])
    return norm

def treinar_juiz():
    caminho_fixo = r"C:\Users\guisc\Documents\IA_LIBRAS_2026\dataset_mestre.csv"
    print(f"🔍 Lendo banco mestre: {caminho_fixo}")

    try:
        df = pd.read_csv(caminho_fixo)
        print(f"📊 Linhas originais: {len(df)}")
        
        # 1. Remove colunas de índice que o pandas às vezes cria
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # 2. EM VEZ DE DROPNA, VAMOS USAR FILLNA
        # Isso preenche buracos com 0 para o KNN não reclamar, mas sem apagar a linha
        df_limpo = df.fillna(0) 
        
        # 3. Garante que a coluna 'label' existe
        if 'label' not in df_limpo.columns:
            print("❌ [ERRO] A coluna 'label' não foi encontrada no CSV!")
            return None

        # 4. Separa dados e labels
        X_bruto = df_limpo.drop('label', axis=1).values
        y = df_limpo['label'].values
        
        # 5. Normalização (Ponto 0 no pulso)
        print("🧠 Normalizando dados para o Juiz...")
        X_norm = [normalizar_coordenadas(linha) for linha in X_bruto]
            
        # 6. Treino do KNN
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_norm, y)
        
        print(f"✅ JUIZ ATIVADO! Treinado com {len(df_limpo)} amostras.")
        return knn

    except Exception as e:
        print(f"❌ Erro crítico no Juiz: {e}")
        return None

# =====================================================================
# LÓGICA DE NAVEGAÇÃO E COLETA
# =====================================================================

def iniciar_coleta():
    juiz = treinar_juiz()
    cap = cv2.VideoCapture(0)
    
    indice_letra = 0
    cache_dados = {} 

    print("\n" + "="*40)
    print("SISTEMA DE COLETA COM NAVEGAÇÃO")
    print("Controles:")
    print(" [C] Gravar | [R] Resetar Letra")
    print(" [B] Voltar Letra | [N] Pular Letra")
    print(" [ESC] SALVAR E FECHAR TUDO")
    print("="*40 + "\n")

    while indice_letra < len(alfabeto_completo):
        letra = alfabeto_completo[indice_letra]
        amostras_letra = cache_dados.get(letra, [])
        gravando = False
        
        while True:
            sucesso, frame = cap.read()
            if not sucesso: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            status_txt = f"LETRA: {letra} | {len(amostras_letra)}/{AMOSTRAS_POR_SESSAO}"
            cor_status = (255, 255, 255)

            if res.multi_hand_landmarks:
                hand_lms = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                bruto = []
                for lm in hand_lms.landmark: bruto.extend([lm.x, lm.y, lm.z])
                
                if not np.isnan(bruto).any():
                    norm = normalizar_coordenadas(bruto)
                    score = res.multi_handedness[0].classification[0].score
                    
                    if juiz and score > MARGEM_QUALIDADE_MINIMA:
                        pred = juiz.predict([norm])[0]
                        if pred == letra:
                            cor_status = (0, 255, 0)
                            status_txt = f"OK! GRAVANDO {letra}..." if gravando else f"SINAL {letra} OK"
                            if gravando and len(amostras_letra) < AMOSTRAS_POR_SESSAO:
                                amostras_letra.append([letra] + bruto)
                        else:
                            cor_status = (0, 165, 255)
                            status_txt = f"PARECE {pred}! CORRIJA"
                    elif gravando and len(amostras_letra) < AMOSTRAS_POR_SESSAO:
                        amostras_letra.append([letra] + bruto)

            # Interface
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(frame, status_txt, (10, 40), 1, 1.5, cor_status, 2)
            cv2.putText(frame, "[C] Gravar | [R] Reset | [B] Voltar | [N] Pular | [ESC] Sair", (10, h-20), 1, 1, (255,255,255), 1)
            
            cv2.imshow("Coleta LIBRAS Otimizada", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'): gravando = True
            if key == ord('r'): 
                amostras_letra = []
                gravando = False
            if key == ord('b'): 
                cache_dados[letra] = amostras_letra
                indice_letra = max(0, indice_letra - 1)
                break
            if key == ord('n') or len(amostras_letra) >= AMOSTRAS_POR_SESSAO: 
                cache_dados[letra] = amostras_letra
                indice_letra += 1
                break
            if key == 27: # ESC
                cache_dados[letra] = amostras_letra # Salva a letra atual antes de sair
                indice_letra = 9999 
                break

    # --- FINALIZAÇÃO ---
    print("\n💾 Finalizando sessão e salvando progresso...")
    dados_finais = []
    for l in alfabeto_completo:
        if l in cache_dados: dados_finais.extend(cache_dados[l])
    
    if dados_finais:
        colunas = ['label'] + [f'{e}{i}' for i in range(21) for e in ['x', 'y', 'z']]
        pd.DataFrame(dados_finais, columns=colunas).to_csv(ARQUIVO_BANCO_DADOS, index=False)
        print(f"✅ Arquivo '{ARQUIVO_BANCO_DADOS}' atualizado com sucesso!")

    # Libera tudo e fecha o processo
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Sistema encerrado.")
    sys.exit() # Garante que o terminal pare de rodar o script

if __name__ == "__main__":
    iniciar_coleta()