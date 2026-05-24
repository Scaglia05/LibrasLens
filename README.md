# 🤟 LibrasLens: Reconhecimento de LIBRAS em Tempo Real (MLP)

O **LibrasLens** é uma solução de acessibilidade de alto desempenho desenvolvida para traduzir o alfabeto da Língua Brasileira de Sinais (LIBRAS) em tempo real. Utilizando Visão Computacional de ponta e Redes Neurais Artificiais, o sistema converte gestos capturados por webcam em texto e síntese de voz.

Diferente de abordagens tradicionais baseadas em sequências temporais (LSTM), este projeto utiliza uma arquitetura **Multilayer Perceptron (MLP)** focada na integridade geométrica instantânea, atingindo uma acurácia de **99,9%** com baixíssima latência.

---

## 👥 Equipe e Orientação

* **Integrantes:** Guilherme Augusto Scaglia, João Pedro Denardo e Pedro Henrique Oliveira de Souza.
* **Instituição:** Engenharia da Computação — Fundação Hermínio Ometto (FHO).
* **Orientador:** Prof. Renato Cagnin.

---

## 🚀 Como o Projeto Funciona (Pipeline)

O LibrasLens opera através de um pipeline modular de quatro estágios:

1.  **Extração de Landmarks:** Utiliza o framework *MediaPipe Hands* para mapear 21 marcos (pontos) tridimensionais da mão, gerando um vetor de 63 características ($21 \text{ pontos} \times 3 \text{ coordenadas } [x, y, z]$).
2.  **Normalização Cinemática:** Aplica uma translação de eixos para definir o pulso (ponto 0) como a origem $(0,0,0)$. Isto torna o sistema invariante à distância da câmara ou posição da mão no ecrã.
3.  **Classificação Neural:** Uma rede MLP processa a geometria 3D. A arquitetura inclui camadas densas (128, 64, 32), *BatchNormalization* para estabilidade e *Dropout* para evitar overfitting.
4.  **Acessibilidade (TTS):** A predição é exibida numa interface web e convertida em áudio via `gTTS` (Google Text-to-Speech).

---

## 📄 Documentação Científica

O desenvolvimento deste projeto foi rigorosamente documentado num artigo científico seguindo os padrões da IEEE, detalhando a fundamentação teórica, a arquitetura da rede neural MLP e a análise estatística dos resultados.

| Arquivo | Formato | Link de Acesso |
| :--- | :---: | :--- |
| **Artigo Científico - LibrasLens** | `PDF` | [📥 Clique aqui para baixar/visualizar](./Reconhecimento_de_LIBRAS_em_Tempo_Real_via_MLP.pdf) |

---

## 📦 Download do Projeto Completo

Para facilitar os testes, avaliação e a replicação dos resultados, disponibilizamos o repositório completo em formato compactado (`.zip`). O arquivo contém todos os códigos-fonte, o banco de dados autoral extraído (`dataset.csv`), os modelos neurais pré-treinados (`.h5`) e a estrutura de pastas configurada.

🔗 **[Clique aqui para baixar o projeto completo (Google Drive)](https://drive.google.com/file/d/1BEnWo_EjBWr_MwHuhXDqhUMb7cmvtFQx/view?usp=sharing)**

---

## 🛠️ Configuração e Instalação

### 1. Pré-requisitos
* **Python 3.11** (Versão recomendada para garantir compatibilidade com MediaPipe e TensorFlow).
* **Webcam** funcional.

### 2. Instalação do Ambiente
É altamente recomendável utilizar um ambiente virtual (`venv`):

```powershell
# Criar o ambiente virtual
python -m venv venv

# Ativar o ambiente (Windows)
.\venv\Scripts\activate

# Instalar as dependências
pip install -r requirements.txt

```

---

## 📂 Guia de Execução

Siga a sequência lógica do projeto para replicar os resultados:

### Passo 1: Coleta de Dados (`1_coleta_dados.py`)

Abre a webcam para capturar as coordenadas dos sinais e gerar o dataset autoral.

* **Saída:** Arquivo `dataset.csv`.

### Passo 2: Pré-processamento (`2_pre_processamento.py`)

Executa a limpeza dos dados e a normalização invariante centrada no pulso.

```powershell
python 2_pre_processamento.py

```

* **Saída:** Pasta `data_ready/` com dados prontos para o treino.

### Passo 3: Treinamento (`3_treinamento.py`)

Treina a rede neural MLP. Utiliza *callbacks* como `EarlyStopping` e `ReduceLROnPlateau` para otimização.

```powershell
python 3_treinamento.py

```

* **Saída:** Modelo treinado (`modelo_libras.h5`) e gráficos de performance.

### Passo 4: Aplicação Final (`app_libras.py`)

Executa a interface de tradução em tempo real via browser.

```powershell
streamlit run app_libras.py

```

---

## 📊 Resultados e Performance

O modelo LibrasLens demonstrou resultados superiores quando comparado com arquiteturas da literatura recente:

| Trabalho | Método | Dados | Acurácia |
| --- | --- | --- | --- |
| Alves et al. (2024) | CNN | Skeleton 2D | 92.1% |
| Kamble (2025) | LSTM | Temporal | 86.7% |
| **LibrasLens** | **MLP** | **Geometria 3D** | **99.9%** |

---

## 📝 Notas de Implementação

* **Estabilidade de Predição:** Implementámos um filtro de "Persistência de Confirmação". Uma letra só é adicionada à frase se a confiança do modelo for superior a 98% durante 15 frames consecutivos.
* **Resiliência:** O uso do otimizador *Adam* aliado à normalização permite que o sistema funcione com diferentes biótipos de mãos e condições de iluminação.

---

**Projeto de Inteligência Artificial II — FHO 2026**

```
