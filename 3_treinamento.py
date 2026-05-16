import os
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Importações corrigidas e padronizadas
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. CONFIGURAÇÕES ---
IMG_SIZE = 64
EPOCHS = 50
BATCH_SIZE = 32
# Certifique-se de que o caminho abaixo existe exatamente assim
DIR_DATASET = 'dataset_imagens/Train' 

def treinar_modelo_v2():
    print(f"🚀 Iniciando Treinamento Estilo Lucas Lacerda - {datetime.datetime.now()}")

    # --- 2. DATA AUGMENTATION ---
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DIR_DATASET,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        color_mode='rgb'
    )

    val_generator = train_datagen.flow_from_directory(
        DIR_DATASET,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        color_mode='rgb'
    )

    num_classes = train_generator.num_classes

    # --- 3. ARQUITETURA DA CNN (Corrigida: Conv2D com D maiúsculo) ---
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
# No model.compile, mude o learning_rate:
model.compile(optimizer=Adam(learning_rate=0.0001), # Era 0.001
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

    # --- 4. CALLBACKS ---
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # --- 5. EXECUÇÃO ---
    print("[INFO] Treinando a rede...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1
    )

    # --- 6. SALVAMENTO ---
    if not os.path.exists('models'): os.makedirs('models')
    
    data_str = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
    model_name = f'models/modelo_libras_cnn_{data_str}.h5'
    model.save(model_name)
    
    print(f"✅ Modelo salvo com sucesso: {model_name}")
    
    # Gráfico de Performance
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia do Modelo')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    treinar_modelo_v2()