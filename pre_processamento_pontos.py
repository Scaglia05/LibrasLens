import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURAÇÕES ---
ARQUIVO_CSV = 'dataset_pontos/dados_libras.csv'

def preparar_dados_pontos():
    print("🔍 Iniciando pré-processamento de Landmarks...")
    
    if not os.path.exists(ARQUIVO_CSV):
        print(f"❌ Erro: O arquivo {ARQUIVO_CSV} não existe. Execute o novo coletor primeiro.")
        return

    # 1. CARREGAR DADOS
    df = pd.read_csv(ARQUIVO_CSV)
    
    # 2. SEPARAR CARACTERÍSTICAS (X) E LABELS (y)
    # X são as 63 coordenadas (p0_x até p20_z), y é a coluna 'label'
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # 3. CODIFICAR LABELS (A, B, C... para 0, 1, 2...)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Salva as classes para o App usar na tradução
    np.save('classes.npy', le.classes_)
    print(f"✅ Classes mapeadas: {le.classes_}")

    # 4. NORMALIZAÇÃO (Ponto fundamental para o Artigo)
    # O MediaPipe já entrega entre 0 e 1, mas centralizar os pontos melhora a precisão
    # Subtraímos o ponto do pulso (p0) de todos os outros para tornar o sinal
    # independente da posição da mão na tela.
    X_normalized = []
    for row in X:
        row_reshaped = row.reshape(21, 3) # Transforma em 21 pontos (x,y,z)
        pulso = row_reshaped[0] # Ponto 0 é o pulso
        row_centralized = row_reshaped - pulso # Todos os pontos em relação ao pulso
        X_normalized.append(row_centralized.flatten())
    
    X_normalized = np.array(X_normalized)

    # 5. DIVISÃO TREINO E TESTE (Regra de Ouro da IA)
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 6. SALVAR DADOS PRONTOS PARA O TREINO
    if not os.path.exists('data_ready'): os.makedirs('data_ready')
    
    np.save('data_ready/X_train.npy', X_train)
    np.save('data_ready/X_test.npy', X_test)
    np.save('data_ready/y_train.npy', y_train)
    np.save('data_ready/y_test.npy', y_test)

    print(f"✅ Pré-processamento concluído!")
    print(f"📊 Total de amostras: {len(X)}")
    print(f"🚀 Dados salvos em 'data_ready/'. Pronto para o Treinamento!")

if __name__ == "__main__":
    preparar_dados_pontos()