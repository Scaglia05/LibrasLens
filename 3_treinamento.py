import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, f1_score

# --- HIPERPARÂMETROS ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCAS = 150  
HIDDEN_SIZE = 128
NUM_LAYERS = 2
PATIENCE = 15  # Early Stopping: para se não melhorar em 15 épocas
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ARQUITETURA DO MODELO ---
class ModeloLibrasLSTM(nn.Module):
    def __init__(self, num_classes):
        super(ModeloLibrasLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=63, hidden_size=HIDDEN_SIZE, 
                            num_layers=NUM_LAYERS, batch_first=True, dropout=0.3)
        
        # Batch Normalization para estabilizar a saída da LSTM
        self.bn = nn.BatchNorm1d(HIDDEN_SIZE)
        
        # Cabeça de classificação mais densa para padrões complexos
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Pegamos apenas a saída do último passo temporal (last hidden state)
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.bn(out)
        return self.fc(out)

def avaliar_modelo(modelo, loader, criterio):
    modelo.eval()
    perda_total = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            saidas = modelo(X_batch)
            perda = criterio(saidas, y_batch)
            perda_total += perda.item()
            _, previstos = torch.max(saidas, 1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(previstos.cpu().numpy())
    
    return (
        perda_total / len(loader),
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average='weighted', zero_division=0)
    )

def treinar():
    # --- CARREGAMENTO DE DADOS ---
    try:
        X_train = torch.tensor(np.load('X_train.npy'), dtype=torch.float32)
        y_train = torch.tensor(np.load('y_train.npy'), dtype=torch.long)
        X_val = torch.tensor(np.load('X_val.npy'), dtype=torch.float32)
        y_val = torch.tensor(np.load('y_val.npy'), dtype=torch.long)
        X_test = torch.tensor(np.load('X_test.npy'), dtype=torch.float32)
        y_test = torch.tensor(np.load('y_test.npy'), dtype=torch.long)
    except FileNotFoundError:
        print("❌ Arquivos .npy não encontrados. Rode o script de pré-processamento!")
        return

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

    num_classes = len(np.unique(y_train.numpy()))
    modelo = ModeloLibrasLSTM(num_classes).to(DEVICE)
    criterio = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Scheduler: Reduz LR se a perda de validação parar de cair
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(otimizador, 'min', patience=5, factor=0.5)

    melhor_perda_val = float('inf')
    contador_patience = 0

    print(f"\n🚀 Iniciando Treino em: {DEVICE} | Classes: {num_classes}")
    print("-" * 50)

    for epoca in range(EPOCAS):
        modelo.train()
        erro_treino = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            otimizador.zero_grad()
            saida = modelo(X_batch)
            erro = criterio(saida, y_batch)
            erro.backward()
            otimizador.step()
            erro_treino += erro.item()

        # Validação
        perda_val, acc_val, f1_val = avaliar_modelo(modelo, val_loader, criterio)
        scheduler.step(perda_val)

        if (epoca + 1) % 5 == 0 or epoca == 0:
            print(f"Época [{epoca+1:03d}] | Erro Tr: {erro_treino/len(train_loader):.4f} | Acc Val: {acc_val:.4f} | LR: {otimizador.param_groups[0]['lr']:.6f}")

        # Lógica de Early Stopping e Checkpoint
        if perda_val < melhor_perda_val:
            melhor_perda_val = perda_val
            torch.save(modelo.state_dict(), 'modelo_libras_lstm.pth')
            contador_patience = 0
        else:
            contador_patience += 1
            if contador_patience >= PATIENCE:
                print(f"\n🛑 Early Stopping ativado na época {epoca+1}!")
                break

    # --- TESTE FINAL ---
    print("\n" + "="*40)
    print("🎯 TESTE FINAL COM O MELHOR MODELO")
    modelo.load_state_dict(torch.load('modelo_libras_lstm.pth'))
    _, t_acc, t_f1 = avaliar_modelo(modelo, test_loader, criterio)
    print(f"Acurácia: {t_acc:.4f} | F1-Score: {t_f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    treinar()