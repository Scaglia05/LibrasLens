import pandas as pd
import numpy as np
import glob
import sys
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- CONFIGURAÇÕES ---
SEQ_LENGTH = 15  # Janela de 15 frames (aprox. 0.5s a 30fps)
ARQUIVO_ENTRADA = "banco_dados_libras.csv"

def normalizar_coordenadas_avancado(X):
    """
    1. Translação: Move o pulso para a origem (0,0,0).
    2. Escala: Redimensiona a mão para um tamanho padrão baseado na palma.
    """
    X_norm = np.copy(X)
    
    for linha in range(X_norm.shape[0]):
        # Ponto 0: Pulso (x, y, z)
        pulso = X_norm[linha, 0:3]
        
        # Ponto 9: Base do dedo médio (usado para calcular a escala da palma)
        ponto_referencia = X_norm[linha, 27:30] 
        
        # Cálculo da distância (Norma Euclidiana) para normalização de escala
        distancia_palma = np.linalg.norm(ponto_referencia - pulso)
        
        # Evita divisão por zero caso o MediaPipe falhe
        if distancia_palma == 0:
            distancia_palma = 1.0

        for i in range(0, 63, 3):
            # Subtrai o pulso (Translação) e divide pela escala
            X_norm[linha, i : i+3] = (X_norm[linha, i : i+3] - pulso) / distancia_palma
            
    return X_norm

def criar_sequencias(X, y):
    X_seq, y_seq = [], []
    for i in range(len(X) - SEQ_LENGTH):
        # Verifica se todos os frames na janela pertencem à mesma classe
        if len(set(y[i : i + SEQ_LENGTH])) == 1: 
            X_seq.append(X[i : i + SEQ_LENGTH])
            y_seq.append(y[i + SEQ_LENGTH - 1])
    return np.array(X_seq), np.array(y_seq)

def preparar_dados():
    print("="*60)
    print("🛠️  INICIANDO PRÉ-PROCESSAMENTO (PADRÃO LSTM 2026)")
    print("="*60)

    if not os.path.exists(ARQUIVO_ENTRADA):
        print(f"❌ [ERRO] {ARQUIVO_ENTRADA} não encontrado!")
        sys.exit()

    df = pd.read_csv(ARQUIVO_ENTRADA)
    
    # Separação de Features e Labels
    X_bruto = df.drop('label', axis=1).values.astype(np.float32)
    y_texto = df['label'].values
    
    print(f"-> Frames brutos: {len(X_bruto)}")

    # 1. Normalização (Translação + Escala)
    print("-> Aplicando Invariância de Translação e Escala...")
    X_norm = normalizar_coordenadas_avancado(X_bruto)

    # 2. Encoding das Labels
    encoder = LabelEncoder()
    y_numerico = encoder.fit_transform(y_texto)
    print(f"-> Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")

    # 3. Criação de Janelas Temporais para LSTM
    print(f"-> Gerando sequências de {SEQ_LENGTH} frames...")
    X_seq, y_seq = criar_sequencias(X_norm, y_numerico)
    
    if len(X_seq) == 0:
        print("❌ [ERRO] Dados insuficientes para criar sequências!")
        sys.exit()

    # 4. Divisão do Dataset (80% Treino/Val, 20% Teste Final)
    # Usamos stratify para manter o equilíbrio entre as letras
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_seq, y_seq, test_size=0.20, stratify=y_seq, random_state=42
    )
    
    # Divide os 80% em Treino (64% total) e Validação (16% total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=42
    )

    # 5. Salvamento dos arquivos binários
    print("-> Salvando arquivos .npy para o treinamento...")
    formatos = {
        'X_train.npy': X_train, 'y_train.npy': y_train,
        'X_val.npy': X_val,   'y_val.npy': y_val,
        'X_test.npy': X_test,  'y_test.npy': y_test,
        'classes_encoder.npy': encoder.classes_
    }
    
    for nome, dado in formatos.items():
        np.save(nome, dado)

    print("="*60)
    print(f"✅ SUCESSO! Dataset pronto para a LSTM.")
    print(f"📦 Treino: {X_train.shape} | Val: {X_val.shape} | Teste: {X_test.shape}")
    print("="*60)

if __name__ == "__main__":
    preparar_dados()