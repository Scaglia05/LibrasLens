import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURAÇÕES ---
DIR_TRAIN = 'dataset_imagens/train'
IMG_SIZE = 64

def pre_processamento_cnn():
    print("🔍 Iniciando pré-processamento para modelo CNN...")
    
    if not os.path.exists(DIR_TRAIN):
        print(f"❌ Erro: A pasta {DIR_TRAIN} não existe. Execute o coletor primeiro.")
        return

    # 1. MAPEAMENTO DE CLASSES
    # O Lucas usa a ordem alfabética das pastas
    classes = sorted(os.listdir(DIR_TRAIN))
    le = LabelEncoder()
    le.fit(classes)
    
    # Salva as classes para o App usar depois na predição
    np.save('classes.npy', le.classes_)
    print(f"✅ Classes detectadas e salvas: {le.classes_}")

    # 2. VERIFICAÇÃO DE INTEGRIDADE (Opcional, mas recomendado)
    print("📸 Verificando qualidade das imagens...")
    total_imagens = 0
    for letra in classes:
        pasta = os.path.join(DIR_TRAIN, letra)
        fotos = os.listdir(pasta)
        total_imagens += len(fotos)
        
        # Verifica se as fotos abrem corretamente
        for foto in fotos[:10]: # Checa as 10 primeiras de cada letra
            caminho = os.path.join(pasta, foto)
            img = cv2.imread(caminho)
            if img is None:
                print(f"⚠️ Alerta: Imagem corrompida em {caminho}")
            elif img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                # Se não estiver no tamanho certo, o script do Lucas (Script 2) deve ser rodado
                pass

    print(f"✅ Verificação concluída. Total de imagens: {total_imagens}")
    print("🚀 DICA: Agora você não precisa de arquivos .npy gigantes.")
    print("O script de Treino do Lucas lerá as pastas diretamente usando ImageDataGenerator.")

if __name__ == "__main__":
    pre_processamento_cnn()