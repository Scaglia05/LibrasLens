import cv2
import pandas as pd
import numpy as np
import os
import sys

# --- MACHINE LEARNING PARA O JUIZ ---
from sklearn.neighbors import KNeighborsClassifier

# --- CORREÇÃO DO MEDIAPIPE ---
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

# --- SETUP INICIAL ---
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

ARQUIVO_BANCO_DADOS = 'banco_dados_libras.csv'
ARQUIVO_MESTRE = 'dataset_mestre.csv' # Seu banco base para validação!

alfabeto_completo = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                     'ESPACO', 'APAGAR']

VOLTAS_TOTAIS = 3
AMOSTRAS_POR_SESSAO = 50
MARGEM_QUALIDADE_MINIMA = 0.90 

# =====================================================================
# SISTEMA DE VALIDAÇÃO (O JUIZ)
# =====================================================================

def normalizar_coordenadas_frame(coordenadas_brutas):
    """Normaliza 1 único frame em tempo real (Ponto 0 no Pulso)"""
    pulso_x, pulso_y, pulso_z = coordenadas_brutas[0], coordenadas_brutas[1], coordenadas_brutas[2]
    norm = []
    for i in range(0, len(coordenadas_brutas), 3):
        norm.extend([
            coordenadas_brutas[i] - pulso_x,
            coordenadas_brutas[i+1] - pulso_y,
            coordenadas_brutas[i+2] - pulso_z
        ])
    return norm

def treinar_juiz():
    """Lê o dataset mestre e cria uma IA fiscalizadora rápida"""
    if not os.path.exists(ARQUIVO_MESTRE):
        print(f"[AVISO] '{ARQUIVO_MESTRE}' não encontrado. A gravação ocorrerá SEM validação de formato.")
        return None
        
    print(f"[INFO] Treinando o Juiz Validador com o '{ARQUIVO_MESTRE}'...")
    try:
        df = pd.read_csv(ARQUIVO_MESTRE)
        X_bruto = df.drop('label', axis=1).values
        y = df['label'].values
        
        # O juiz precisa aprender com os dados normalizados!
        X_norm = []
        for linha in X_bruto:
            X_norm.append(normalizar_coordenadas_frame(linha))
            
        juiz_knn = KNeighborsClassifier(n_neighbors=3)
        juiz_knn.fit(X_norm, y)
        print("[SUCESSO] Juiz treinado e pronto para fiscalizar!")
        return juiz_knn
    except Exception as e:
        print(f"[ERRO] Falha ao treinar juiz: {e}. Coletando sem fiscalização.")
        return None

# =====================================================================
# FUNÇÕES NORMAIS DE COLETA
# =====================================================================

def avaliar_qualidade(hand_landmarks, handedness, letra_alvo, juiz):
    """Verifica qualidade técnica e o formato do sinal"""
    score = handedness.classification[0].score
    
    # 1. Verifica Borda da Tela
    for lm in hand_landmarks.landmark:
        if lm.x < 0.01 or lm.x > 0.99 or lm.y < 0.01 or lm.y > 0.99:
            return False, score, "MAO CORTADA", (0, 0, 255)
            
    # 2. Verifica Confiança MediaPipe
    if score < MARGEM_QUALIDADE_MINIMA:
        return False, score, "BAIXA QUALIDADE", (0, 0, 255)
        
    # 3. O JUIZ ENTRA EM AÇÃO AQUI
    if juiz is not None:
        coordenadas_brutas = []
        for lm in hand_landmarks.landmark:
            coordenadas_brutas.extend([lm.x, lm.y, lm.z])
            
        coords_norm = normalizar_coordenadas_frame(coordenadas_brutas)
        previsao_do_juiz = juiz.predict([coords_norm])[0]
        
        # Se o juiz achar que a letra é diferente da que estamos tentando gravar
        if previsao_do_juiz != letra_alvo:
            # Ex: "VC ESTA FAZENDO C!"
            return False, score, f"PARECE {previsao_do_juiz}!", (0, 165, 255) # Laranja para sinal errado
            
    return True, score, "OK", (0, 255, 0) # Verde

# (A função verificar_progresso, salvar_no_banco e encontrar_camera continuam iguais)
def verificar_progresso():
    contagem_letras = {letra: 0 for letra in alfabeto_completo}
    if os.path.exists(ARQUIVO_BANCO_DADOS):
        try:
            df = pd.read_csv(ARQUIVO_BANCO_DADOS)
            if 'label' in df.columns:
                contagem_letras.update(df['label'].value_counts().to_dict())
        except: pass
    fila = []
    for volta in range(1, VOLTAS_TOTAIS + 1):
        for letra in alfabeto_completo:
            if contagem_letras[letra] < volta * AMOSTRAS_POR_SESSAO:
                fila.append((letra, volta))
                contagem_letras[letra] += AMOSTRAS_POR_SESSAO
    return fila

def salvar_no_banco(novos_dados):
    colunas = ['label'] + [f'{e}{i}' for i in range(21) for e in ['x', 'y', 'z']]
    df = pd.DataFrame(novos_dados, columns=colunas)
    if not os.path.exists(ARQUIVO_BANCO_DADOS): df.to_csv(ARQUIVO_BANCO_DADOS, index=False)
    else: df.to_csv(ARQUIVO_BANCO_DADOS, mode='a', header=False, index=False)
    print(f"-> +{len(novos_dados)} amostras salvas.")

def iniciar_coleta():
    juiz = treinar_juiz() # Inicia o juiz antes de tudo
    fila_para_gravar = verificar_progresso()
    
    if not fila_para_gravar:
        print("\n🎉 PARABÉNS! Você já completou as voltas!")
        return

    cap = cv2.VideoCapture(0)
    
    for letra, volta in fila_para_gravar:
        dados_da_letra = []
        
        # Loop de Preview
        while True:
            sucesso, frame = cap.read()
            if not sucesso: break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"LETRA: {letra} | VOLTA {volta}/3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "'C' p/ Gravar | 'S' p/ Pausar", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Coleta LIBRAS", frame)
            
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('c'): break 
            if tecla == ord('s'): return cap.release()

        # Loop de Gravação
        amostras = 0
        while amostras < AMOSTRAS_POR_SESSAO:
            sucesso, frame = cap.read()
            if not sucesso: break
            frame = cv2.flip(frame, 1)
            resultado = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if resultado.multi_hand_landmarks and resultado.multi_handedness:
                hand_landmarks = resultado.multi_hand_landmarks[0]
                handedness = resultado.multi_handedness[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # AVALIAÇÃO RIGOROSA
                aprovado, score, motivo, cor = avaliar_qualidade(hand_landmarks, handedness, letra, juiz)
                
                if aprovado:
                    linha = [letra]
                    for lm in hand_landmarks.landmark: linha.extend([lm.x, lm.y, lm.z])
                    dados_da_letra.append(linha)
                    amostras += 1
                    texto = f"GRAVANDO: {letra} | {amostras}/50"
                else:
                    texto = f"RECUSADO: {motivo}"
                    
                cv2.putText(frame, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
            else:
                cv2.putText(frame, "CADE A MAO?", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            cv2.imshow("Coleta LIBRAS", frame)
            cv2.waitKey(1)

        salvar_no_banco(dados_da_letra)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_coleta()