import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURAÇÕES ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 # Reduzi para 16 para maior estabilidade
EPOCAS = 100
LEARNING_RATE = 0.001

# --- 2. ARQUITETURA HÍBRIDA ---
class ModeloHibridoLibras(nn.Module):
    def __init__(self, num_classes):
        super(ModeloHibridoLibras, self).__init__()
        # Ramo CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        # Ramo LSTM
        self.lstm = nn.LSTM(63, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.bn_lstm = nn.BatchNorm1d(512)
        # Classificador
        self.classifier = nn.Sequential(
            nn.Linear(256 + 512, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, pts):
        x_img = self.cnn(img)
        out_lstm, _ = self.lstm(pts)
        x_pts = out_lstm[:, -1, :]
        x_pts = self.bn_lstm(x_pts)
        return self.classifier(torch.cat((x_img, x_pts), dim=1))

# --- 3. DATASET COM CORREÇÃO DE TAMANHO ---
class LibrasDataset(Dataset):
    def __init__(self, imgs, pts, labels):
        # Correção da dimensão da imagem
        temp_imgs = torch.tensor(imgs, dtype=torch.float32)
        if temp_imgs.ndimension() == 5:
            temp_imgs = temp_imgs.squeeze(2)
        
        self.imgs = temp_imgs / 255.0
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        # Lógica de alinhamento dos pontos (LSTM)
        total_coordenadas = pts.flatten().shape[0]
        tamanho_sequencia = 15 * 63 
        num_sequencias_possiveis = total_coordenadas // tamanho_sequencia
        
        pts_formatados = torch.tensor(pts.flatten()[:num_sequencias_possiveis * tamanho_sequencia], dtype=torch.float32)
        self.pts = pts_formatados.view(-1, 15, 63)
        
        # Sincronização final
        self.val_len = min(len(self.imgs), len(self.pts), len(self.labels))
        print(f"📏 Dataset alinhado: {self.val_len} amostras prontas para o treino.")

    def __len__(self): 
        return self.val_len
        
    def __getitem__(self, idx): 
        return self.imgs[idx], self.pts[idx], self.labels[idx]

# --- 4. FUNÇÃO DE TREINO ---
def treinar():
    print("📂 Carregando e Alinhando arquivos...")
    X_img = np.load('X_hibrido_img.npy')
    X_pts = np.load('X_hibrido_pts.npy')
    y = np.load('y_hibrido_labels.npy')
    classes = np.load('classes.npy', allow_pickle=True)

    # CORREÇÃO CRÍTICA: Alinha todos os arrays pelo menor tamanho encontrado
    min_samples = min(len(X_img), len(X_pts), len(y))
    X_img = X_img[:min_samples]
    X_pts = X_pts[:min_samples]
    y = y[:min_samples]
    
    print(f"✅ Dados alinhados: {min_samples} amostras encontradas.")

    img_train, img_val, pts_train, pts_val, y_train, y_val = train_test_split(
        X_img, X_pts, y, test_size=0.2, stratify=y, random_state=42
    )

    train_loader = DataLoader(LibrasDataset(img_train, pts_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(LibrasDataset(img_val, pts_val, y_val), batch_size=BATCH_SIZE)

    modelo = ModeloHibridoLibras(len(classes)).to(DEVICE)
    criterio = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

    print(f"🚀 Treinando em {DEVICE}...")
    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        loss_total = 0
        for imgs, pts, labels in train_loader:
            imgs, pts, labels = imgs.to(DEVICE), pts.to(DEVICE), labels.to(DEVICE)
            
            otimizador.zero_grad()
            saida = modelo(imgs, pts)
            loss = criterio(saida, labels)
            loss.backward()
            otimizador.step()
            loss_total += loss.item()
        
        if epoca % 10 == 0 or epoca == 1:
            print(f"Época {epoca}/{EPOCAS} | Perda: {loss_total/len(train_loader):.4f}")

    torch.save(modelo.state_dict(), 'modelo_hibrido_final.pth')
    print("✨ Modelo salvo: modelo_hibrido_final.pth")

if __name__ == "__main__":
    treinar()