import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- HIPERPARÂMETROS ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCAS = 100
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ARQUITETURA DO MODELO ---
class ModeloLibrasLSTM(nn.Module):
    def __init__(self, num_classes):
        super(ModeloLibrasLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=63, hidden_size=HIDDEN_SIZE, 
                            num_layers=NUM_LAYERS, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(HIDDEN_SIZE, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def avaliar_modelo(modelo, loader):
    modelo.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            saidas = modelo(X_batch)
            _, previstos = torch.max(saidas, 1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(previstos.cpu().numpy())
    
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='weighted', zero_division=0),
        recall_score(y_true, y_pred, average='weighted', zero_division=0),
        f1_score(y_true, y_pred, average='weighted', zero_division=0)
    )

def treinar():
    # --- VERIFICAÇÃO DE ARQUIVOS ---
    arquivos_necessarios = ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy', 'X_test.npy', 'y_test.npy']
    faltando = [f for f in arquivos_necessarios if not os.path.exists(f)]
    
    if faltando:
        print("\n" + "!"*50)
        print(f"[ERRO] Arquivos ausentes: {faltando}")
        print("Execute o script '2_pre_processamento.py' antes de treinar!")
        print("!"*50 + "\n")
        return

    try:
        # Carregando os dados
        X_train = torch.tensor(np.load('X_train.npy'), dtype=torch.float32)
        y_train = torch.tensor(np.load('y_train.npy'), dtype=torch.long)
        X_val = torch.tensor(np.load('X_val.npy'), dtype=torch.float32)
        y_val = torch.tensor(np.load('y_val.npy'), dtype=torch.long)
        X_test = torch.tensor(np.load('X_test.npy'), dtype=torch.float32)
        y_test = torch.tensor(np.load('y_test.npy'), dtype=torch.long)

        # Verificação de consistência
        if X_train.shape[2] != 63:
            print(f"[ERRO] O modelo espera 63 coordenadas, mas os dados têm {X_train.shape[2]}.")
            return

        # DataLoaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

        num_classes = len(np.unique(y_train.numpy()))
        if num_classes < 2:
            print("[ERRO] Você precisa de pelo menos 2 classes (letras) diferentes para treinar.")
            return

        modelo = ModeloLibrasLSTM(num_classes).to(DEVICE)
        criterio = nn.CrossEntropyLoss()
        otimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

        print(f"\n[INFO] Iniciando Treino em: {DEVICE}")
        print(f"[INFO] Total de Classes: {num_classes}")
        print("-" * 30)

        for epoca in range(EPOCAS):
            modelo.train()
            erro_acumulado = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                
                otimizador.zero_grad()
                saida = modelo(X_batch)
                erro = criterio(saida, y_batch)
                erro.backward()
                otimizador.step()
                erro_acumulado += erro.item()
            
            if (epoca+1) % 10 == 0 or epoca == 0:
                acc, _, _, f1 = avaliar_modelo(modelo, val_loader)
                print(f"Época [{epoca+1:03d}/{EPOCAS}] | Erro: {erro_acumulado/len(train_loader):.4f} | Acc Val: {acc:.4f} | F1 Val: {f1:.4f}")

        # --- AVALIAÇÃO FINAL ---
        print("\n" + "="*40)
        print("   AVALIAÇÃO FINAL (CONJUNTO DE TESTE)")
        print("="*40)
        t_acc, t_prec, t_rec, t_f1 = avaliar_modelo(modelo, test_loader)
        print(f"Acurácia: {t_acc:.4f}")
        print(f"Precisão: {t_prec:.4f}")
        print(f"Recall:   {t_rec:.4f}")
        print(f"F1-Score: {t_f1:.4f}")
        print("="*40)

        # Salva o arquivo final
        torch.save(modelo.state_dict(), 'modelo_libras_lstm.pth')
        print("\n[SUCESSO] Modelo salvo como 'modelo_libras_lstm.pth'!")

    except Exception as e:
        print(f"\n[ERRO CRÍTICO] Ocorreu uma falha durante o treino: {e}")

if __name__ == "__main__":
    treinar()