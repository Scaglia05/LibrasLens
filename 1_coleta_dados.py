import cv2
import pandas as pd
import os
import sys

# --- CORREÇÃO DO MEDIAPIPE ---
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

# --- SETUP INICIAL ---
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

ARQUIVO_BANCO_DADOS = 'banco_dados_libras.csv'

alfabeto_completo = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                     'ESPACO', 'APAGAR']

# Configurações das Voltas
VOLTAS_TOTAIS = 3
AMOSTRAS_POR_SESSAO = 50

def verificar_progresso():
    """Analisa o CSV e gera a fila exata de voltas e letras que faltam."""
    contagem_letras = {letra: 0 for letra in alfabeto_completo}
    
    if os.path.exists(ARQUIVO_BANCO_DADOS):
        try:
            df = pd.read_csv(ARQUIVO_BANCO_DADOS)
            if 'label' in df.columns:
                # Conta quantas amostras de cada letra já existem no CSV
                contagens_salvas = df['label'].value_counts().to_dict()
                contagem_letras.update(contagens_salvas)
        except Exception as e:
            print(f"[AVISO] Erro ao ler progresso: {e}. Começando do zero.")

    # Constrói a fila de gravação organizando pelas 3 voltas
    fila = []
    for volta in range(1, VOLTAS_TOTAIS + 1):
        for letra in alfabeto_completo:
            alvo_parcial = volta * AMOSTRAS_POR_SESSAO
            # Se a contagem está abaixo do alvo dessa volta, adiciona na fila
            if contagem_letras[letra] < alvo_parcial:
                fila.append((letra, volta))
                # Simula que gravou para poder alocar corretamente nas próximas voltas
                contagem_letras[letra] += AMOSTRAS_POR_SESSAO
                
    return fila

def salvar_no_banco(novos_dados):
    colunas = ['label'] + [f'{e}{i}' for i in range(21) for e in ['x', 'y', 'z']]
    
    if not os.path.exists(ARQUIVO_BANCO_DADOS):
        df = pd.DataFrame(novos_dados, columns=colunas)
        df.to_csv(ARQUIVO_BANCO_DADOS, index=False)
    else:
        df = pd.DataFrame(novos_dados, columns=colunas)
        df.to_csv(ARQUIVO_BANCO_DADOS, mode='a', header=False, index=False)
    print(f"-> Dados salvos! ( +{len(novos_dados)} amostras )")

def encontrar_camera():
    print("[INFO] Procurando câmera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("!"*50)
        print("[ERRO] Não foi possível acessar a câmera 0.")
        sys.exit()
        
    return cap

def iniciar_coleta():
    fila_para_gravar = verificar_progresso()
    
    if not fila_para_gravar:
        print(f"\n🎉 PARABÉNS! Você já completou as {VOLTAS_TOTAIS} voltas!")
        print(f"Seu Dataset está pronto no arquivo '{ARQUIVO_BANCO_DADOS}'.")
        return

    print(f"[INFO] Restam {len(fila_para_gravar)} gravações para fechar o ciclo de {VOLTAS_TOTAIS} voltas.")
    cap = encontrar_camera()
    print("[SUCESSO] Câmera ativada!\n")
    
    for letra, volta in fila_para_gravar:
        dados_da_letra = []
        print(f"--- PREPARANDO: Letra {letra} (Volta {volta}/{VOLTAS_TOTAIS}) ---")
        
        # Loop de Espera/Preview
        while True:
            sucesso, frame = cap.read()
            if not sucesso: break

            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"LETRA: {letra} | VOLTA {volta}/3", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "'C' p/ Gravar | 'S' p/ Pausar", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Coleta LIBRAS", frame)
            
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('c'): break 
            if tecla == ord('s'): 
                print("\n[INFO] Coleta pausada. Progresso salvo de forma segura!")
                cap.release()
                cv2.destroyAllWindows()
                return

        # Loop de Gravação
        amostras = 0
        while amostras < AMOSTRAS_POR_SESSAO:
            sucesso, frame = cap.read()
            if not sucesso: break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = hands.process(rgb)
            
            if resultado.multi_hand_landmarks:
                for hand_landmarks in resultado.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    linha = [letra]
                    for lm in hand_landmarks.landmark:
                        linha.extend([lm.x, lm.y, lm.z])
                    
                    dados_da_letra.append(linha)
                    amostras += 1
            
            cv2.putText(frame, f"GRAVANDO {letra} (v{volta}) | {amostras}/50", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow("Coleta LIBRAS", frame)
            cv2.waitKey(1)

        salvar_no_banco(dados_da_letra)

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Todas as voltas finalizadas com sucesso! Banco de dados concluído.")

if __name__ == "__main__":
    iniciar_coleta()