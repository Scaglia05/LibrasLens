import pandas as pd
import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- CONFIGURAÇÕES (Devem ser iguais ao Coletor) ---
ARQUIVO_CSV = 'banco_dados_libras.csv'
DIR_IMAGENS = 'dataset_imagens/Train'
IMG_SIZE = 64
SEQUENCIA_TAMANHO = 15 # Quantos frames a LSTM vai olhar por vez

def preparar_dados_hibridos():
    print("Step 1: Carregando CSV e tratando colunas...")
    # Carregamos o CSV. Se não tiver cabeçalho, nós definimos aqui
    try:
        df = pd.read_csv(ARQUIVO_CSV)
        # Se a primeira coluna não se chamar 'label', vamos renomear
        if 'label' not in df.columns:
            # Assume que a primeira coluna é a letra e o resto são coordenadas
            colunas = ['label'] + [f'coord_{i}' for i in range(df.shape[1] - 1)]
            df = pd.read_csv(ARQUIVO_CSV, names=colunas)
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
        return

    # --- 1. TRATAMENTO DAS LABELS ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['label'])
    np.save('classes.npy', le.classes_)
    num_classes = len(le.classes_)
    print(f"✅ Classes detectadas: {le.classes_}")

    # --- 2. PREPARAÇÃO DOS PONTOS (LSTM) ---
    print("Step 2: Preparando sequências para LSTM...")
    X_pts = df.drop('label', axis=1).values
    
    # Criamos sequências de 15 frames
    X_seq, y_seq = [], []
    for i in range(SEQUENCIA_TAMANHO, len(X_pts)):
        # Só cria sequência se todos os 15 frames forem da mesma letra
        if len(set(df['label'].iloc[i-SEQUENCIA_TAMANHO:i])) == 1:
            X_seq.append(X_pts[i-SEQUENCIA_TAMANHO:i])
            y_seq.append(y_encoded[i-1])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # --- 3. PREPARAÇÃO DAS IMAGENS (CNN) ---
    print("Step 3: Carregando e processando imagens...")
    X_img = []
    # Usamos o caminho das imagens baseado no CSV para manter sincronia
    # Para simplificar, vamos carregar as imagens que batem com as sequências
    for i in range(SEQUENCIA_TAMANHO, len(df)):
        if len(set(df['label'].iloc[i-SEQUENCIA_TAMANHO:i])) == 1:
            letra = df['label'].iloc[i-1]
            pasta = os.path.join(DIR_IMAGENS, letra)
            
            # Pega a última imagem salva na pasta daquela letra (ou correspondente)
            lista_fotos = sorted(os.listdir(pasta))
            # Tenta pegar uma foto proporcional ao índice atual
            idx_foto = min(i, len(lista_fotos)-1)
            img_path = os.path.join(pasta, lista_fotos[idx_foto])
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X_img.append(img)
            else:
                # Se falhar, cria uma imagem preta para não quebrar o array
                X_img.append(np.zeros((IMG_SIZE, IMG_SIZE)))

    X_img = np.array(X_img)
    X_img = X_img.reshape(-1, 1, IMG_SIZE, IMG_SIZE) / 255.0 # Normaliza para CNN

    print(f"✅ Dados processados: Pontos {X_seq.shape} | Imagens {X_img.shape}")

    # --- 4. SALVAMENTO ---
    np.save('X_hibrido_pts.npy', X_seq)
    np.save('X_hibrido_img.npy', X_img)
    np.save('y_hibrido_labels.npy', y_seq)
    
    print("✨ TUDO PRONTO! Pode rodar o 3_treinamento.py")

if __name__ == "__main__":
    preparar_dados_hibridos()