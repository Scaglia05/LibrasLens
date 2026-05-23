import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def treinar_modelo_mlp():
    print(f"🚀 Iniciando Treinamento de Landmarks - {datetime.datetime.now()}")

    # 1. CARREGAR DADOS PRÉ-PROCESSADOS
    try:
        X_train = np.load('data_ready/X_train.npy')
        X_test = np.load('data_ready/X_test.npy')
        y_train = np.load('data_ready/y_train.npy')
        y_test = np.load('data_ready/y_test.npy')
        classes = np.load('classes.npy', allow_pickle=True)
    except FileNotFoundError:
        print("❌ Erro: Arquivos em 'data_ready/' não encontrados. Rode o pré-processamento primeiro.")
        return

    num_classes = len(classes)

    # 2. ARQUITETURA MLP (Multi-Layer Perceptron)
    # Ideal para processar coordenadas geométricas
    model = Sequential([
        # Camada de entrada: 63 neurônios (21 pontos x 3 coordenadas)
        Dense(128, activation='relu', input_shape=(63,)),
        BatchNormalization(), # Normaliza os pesos para treino mais rápido
        Dropout(0.2),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),
        
        # Camada de saída: Uma probabilidade para cada letra
        Dense(num_classes, activation='softmax')
    ])

    # 3. COMPILAÇÃO
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy', # Usamos sparse pois o y é inteiro
        metrics=['accuracy']
    )

    # 4. CALLBACKS ACADÊMICOS
    # Early stop para evitar overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Reduz o aprendizado se o modelo estagnar (Técnica avançada de IA II)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    # 5. EXECUÇÃO
    print("[INFO] Treinando a rede de geometria...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100, # Pontos permitem mais épocas pois são leves
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # 6. SALVAMENTO
    if not os.path.exists('models'): os.makedirs('models')
    data_str = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
    model_name = f'models/modelo_libras_pontos_{data_str}.h5'
    model.save(model_name)
    
    print(f"✅ Modelo de Landmarks salvo: {model_name}")

    # 7. GRÁFICOS DE PERFORMANCE (Essencial para o Artigo)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia por Época')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Perda (Loss) por Época')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    treinar_modelo_mlp()