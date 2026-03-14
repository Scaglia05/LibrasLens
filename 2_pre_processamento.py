import pandas as pd
import numpy as np
import glob
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

SEQ_LENGTH = 15 # A LSTM vai analisar 15 frames (meio segundo) por vez

def criar_sequencias(X, y):
    X_seq, y_seq = [], []
    for i in range(len(X) - SEQ_LENGTH):
        # Garante que a sequência toda pertence à mesma letra
        if len(set(y[i : i + SEQ_LENGTH])) == 1: 
            X_seq.append(X[i : i + SEQ_LENGTH])
            y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def preparar_dados():
    print("[INFO] Carregando CSVs...")
    
    # Busca por arquivos que seguem o padrão de nomenclatura
    arquivos_csv = glob.glob("dataset_libras_*.csv") + glob.glob("banco_dados_libras.csv")
    
    # --- VERIFICAÇÃO DE SEGURANÇA ---
    if not arquivos_csv:
        print("\n" + "!"*50)
        print("[ERRO] Nenhum arquivo de dados encontrado!")
        print("Certifique-se de que você rodou o script '1_coleta_dados.py' primeiro.")
        print("O arquivo .csv deve estar na mesma pasta que este script.")
        print("!"*50 + "\n")
        return # Encerra a função
    # --------------------------------

    print(f"[INFO] Arquivos encontrados: {arquivos_csv}")
    df_final = pd.concat([pd.read_csv(f) for f in arquivos_csv], ignore_index=True)

    # 1. Separação básica
    X_bruto = df_final.drop('label', axis=1).values
    y_texto = df_final['label'].values

    # 2. Converte Letras para Números
    encoder = LabelEncoder()
    y_numerico = encoder.fit_transform(y_texto)

    # 3. Cria as sequências temporais para a LSTM
    print("[INFO] Criando sequências temporais...")
    X_seq, y_seq = criar_sequencias(X_bruto, y_numerico)
    
    if len(X_seq) == 0:
        print("[AVISO] Dados insuficientes para criar sequências. Grave mais frames no script 1!")
        return

    # 4. Divisão de Dados (80/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_seq, y_seq, test_size=0.20, stratify=y_seq, random_state=42
    )
    
    # Divisão de Validação (dentro do treino)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=42
    )

    # Salva os pacotes para o Script 3
    np.save('X_train.npy', X_train); np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val);     np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test);   np.save('y_test.npy', y_test)
    np.save('classes_encoder.npy', encoder.classes_)
    
    print(f"[SUCESSO] Dados divididos! Treino: {len(X_train)}, Validação: {len(X_val)}, Teste: {len(X_test)}")

if __name__ == "__main__":
    preparar_dados()