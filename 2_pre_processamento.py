import pandas as pd
import numpy as np
import glob
import sys
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

SEQ_LENGTH = 15 # A LSTM vai analisar 15 frames (meio segundo) por vez

def normalizar_coordenadas(X):
    """
    O GRANDE TRUQUE: Faz a rede aprender o formato da mão, e não a posição dela na tela!
    Transforma o pulso (Ponto 0) na coordenada (0,0,0) e ajusta o resto dos dedos.
    """
    X_norm = np.copy(X)
    for linha in range(X_norm.shape[0]):
        # O pulso é o primeiro ponto (índices 0, 1, 2 correspondem a x, y, z do Ponto 0)
        pulso_x = X_norm[linha, 0]
        pulso_y = X_norm[linha, 1]
        pulso_z = X_norm[linha, 2]
        
        # Subtrai as coordenadas do pulso de todos os 21 pontos (de 3 em 3 colunas)
        for i in range(0, 63, 3):
            X_norm[linha, i] -= pulso_x     # Normaliza o Eixo X
            X_norm[linha, i+1] -= pulso_y   # Normaliza o Eixo Y
            X_norm[linha, i+2] -= pulso_z   # Normaliza o Eixo Z
            
    return X_norm

def criar_sequencias(X, y):
    X_seq, y_seq = [], []
    for i in range(len(X) - SEQ_LENGTH):
        # Garante que os 15 frames pertencem à mesma letra
        if len(set(y[i : i + SEQ_LENGTH])) == 1: 
            X_seq.append(X[i : i + SEQ_LENGTH])
            y_seq.append(y[i + SEQ_LENGTH - 1]) # O rótulo é o do último frame da sequência
    return np.array(X_seq), np.array(y_seq)

def preparar_dados():
    print("="*50)
    print("[1/5] Lendo o Banco de Dados...")
    # Busca apenas o nosso banco principal
    arquivos_csv = glob.glob("banco_dados_libras.csv")
    
    if not arquivos_csv:
        print("\n[ERRO] Arquivo 'banco_dados_libras.csv' não encontrado!")
        print("Certifique-se de terminar a coleta no script '1_coleta_dados.py'.")
        sys.exit()

    df_final = pd.read_csv(arquivos_csv[0])
    
    if df_final.empty:
        print("[ERRO] O arquivo CSV está vazio!")
        sys.exit()

    X_bruto = df_final.drop('label', axis=1).values
    y_texto = df_final['label'].values
    print(f"-> Total de frames capturados: {len(X_bruto)}")

    print("\n[2/5] Normalizando coordenadas (O Segredo da IA)...")
    # Aplica a nossa nova função que zera o pulso
    X_norm = normalizar_coordenadas(X_bruto)

    print("\n[3/5] Convertendo Letras para Números...")
    encoder = LabelEncoder()
    y_numerico = encoder.fit_transform(y_texto)
    print(f"-> Classes detectadas ({len(encoder.classes_)}): {encoder.classes_}")

    print("\n[4/5] Criando sequências temporais (Janelas de 15 frames)...")
    X_seq, y_seq = criar_sequencias(X_norm, y_numerico)
    
    if len(X_seq) == 0:
        print("[ERRO] Dados insuficientes para criar sequências.")
        sys.exit()
    print(f"-> Total de sequências válidas geradas: {len(X_seq)}")

    print("\n[5/5] Dividindo pacotes para Treino, Validação e Teste...")
    # Separa 20% para Teste Final
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_seq, y_seq, test_size=0.20, stratify=y_seq, random_state=42
    )
    # Pega os 80% que sobraram e separa em Treino e Validação
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=42
    )

    print("\n[SALVANDO ARQUIVOS]")
    np.save('X_train.npy', X_train); np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val);     np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test);   np.save('y_test.npy', y_test)
    np.save('classes_encoder.npy', encoder.classes_)
    
    print("="*50)
    print("✅ PREPARAÇÃO CONCLUÍDA COM SUCESSO!")
    print(f"📊 TREINO:     {len(X_train)} sequências")
    print(f"📈 VALIDAÇÃO:  {len(X_val)} sequências")
    print(f"🎯 TESTE:      {len(X_test)} sequências")
    print("Você já pode rodar o script '3_treinamento.py'!")
    print("="*50)

if __name__ == "__main__":
    preparar_dados()