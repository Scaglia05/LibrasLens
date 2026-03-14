import cv2
import mediapipe as mp
import pandas as pd
import os

# --- SETUP INICIAL ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# O seu "Banco de Dados" será este arquivo
ARQUIVO_BANCO_DADOS = 'banco_dados_libras.csv'

alfabeto = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
            'ESPACO', 'APAGAR']

def salvar_no_banco(novos_dados):
    """Função que gerencia o Banco de Dados CSV"""
    colunas = ['label'] + [f'{e}{i}' for i in range(21) for e in ['x', 'y', 'z']]
    
    # Se o arquivo já existe, adiciona sem o cabeçalho. Se não existe, cria com cabeçalho.
    if not os.path.exists(ARQUIVO_BANCO_DADOS):
        df = pd.DataFrame(novos_dados, columns=colunas)
        df.to_csv(ARQUIVO_BANCO_DADOS, index=False)
    else:
        df = pd.DataFrame(novos_dados, columns=colunas)
        df.to_csv(ARQUIVO_BANCO_DADOS, mode='a', header=False, index=False)
    print(f"-> Dados salvos com sucesso no banco! Total de linhas: {len(novos_dados)}")

def iniciar_coleta():
    cap = cv2.VideoCapture(0)
    
    for letra in alfabeto:
        dados_da_letra = []
        print(f"\n--- PRÓXIMA LETRA: {letra} ---")
        print("Pressione 'C' para COMEÇAR a gravar esta letra ou 'S' para PARAR tudo.")
        
        # Loop de Espera/Preview
        while True:
            sucesso, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"LETRA: {letra} | 'C' p/ Gravar | 'S' p/ Sair", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Coleta LIBRAS", frame)
            
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('c'): break # Começa a gravar
            if tecla == ord('s'): # Para o programa
                cap.release()
                cv2.destroyAllWindows()
                return

        # Loop de Gravação (Grava 50 amostras da letra)
        print(f"Gravando {letra}... Mova a mão levemente!")
        amostras = 0
        while amostras < 50:
            sucesso, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = hands.process(rgb)
            
            if resultado.multi_hand_landmarks:
                for hand_landmarks in resultado.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Coleta as coordenadas
                    linha = [letra]
                    for lm in hand_landmarks.landmark:
                        linha.extend([lm.x, lm.y, lm.z])
                    
                    dados_da_letra.append(linha)
                    amostras += 1
            
            cv2.putText(frame, f"Gravando {letra}: {amostras}/50", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Coleta LIBRAS", frame)
            cv2.waitKey(1)

        # Ao terminar a letra, salva no Banco de Dados (CSV)
        salvar_no_banco(dados_da_letra)
        print(f"Letra {letra} concluída!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_coleta()