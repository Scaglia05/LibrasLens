# 🤟 LibrasLens: Tradutor de LIBRAS com IA (LSTM)

O **LibrasLens** é uma solução de acessibilidade que utiliza Visão Computacional e Redes Neurais Recorrentes (**LSTM**) para reconhecer sinais da Língua Brasileira de Sinais em tempo real. O sistema captura movimentos via webcam, processa a sequência de gestos e converte-os em texto e voz.

---

## 👥 Integrantes do Grupo

| Nome | RA | E-mail |
| :--- | :--- | :--- |
| **Guilherme Augusto Scaglia** | 111598 | scaglia@alunos.fho.edu.br |
| **João Pedro Denardo** | 113036 | denardo749@alunos.fho.edu.br |
| **Pedro Henrique Oliveira de Souza** | 113364 | pedro1204@alunos.fho.edu.br |

**Orientador:** Prof. Renato Cagnin

---

## 📋 Pré-requisitos

Para rodar este projeto, certifique-se de ter:
* **Python 3.11** (Versão estritamente recomendada para garantir a estabilidade do MediaPipe e OpenCV).
* **Webcam** funcional.
* **Git** instalado para controle de versão.

---

## 🛠️ Configuração do Ambiente

É altamente recomendado o uso de um ambiente virtual (`venv`) para isolar as dependências e evitar conflitos de versão (como erros de incompatibilidade do NumPy 2.x). No terminal do projeto, siga a ordem:

### 1. Criar e Ativar a Venv
```powershell
# Criar o ambiente virtual
python -m venv venv

# Ativar no Windows (PowerShell)
.\venv\Scripts\activate
```

### 2. Instalar Dependências
Com a venv ativada (você verá `(venv)` no início da linha do terminal), instale todas as bibliotecas a partir do arquivo de requisitos. Isso instalará as versões exatas e blindadas para o projeto:

```powershell
pip install -r requirements.txt
```

---

## 📂 Estrutura do Projeto e Ordem de Execução

O projeto é dividido em 4 etapas obrigatórias. Siga esta sequência:

| Ordem | Arquivo | Função | Saída Gerada |
| --- | --- | --- | --- |
| **1º** | `1_coleta_dados.py` | Captura 21 pontos da mão (X, Y, Z) via MediaPipe. | `dataset_libras_LETRA.csv` |
| **2º** | `2_pre_processamento.py` | Limpa dados e cria sequências de 15 frames. | Arquivos `.npy` e `classes_encoder.npy` |
| **3º** | `3_treinamento.py` | Treina a Rede Neural LSTM usando PyTorch. | `modelo_libras_lstm.pth` |
| **4º** | `app_libras.py` | Interface final em Streamlit com tradução e voz. | Interface Web (Dashboard) |

---

## 📄 Documentação Científica

O desenvolvimento deste projeto foi documentado em um artigo científico que detalha a fundamentação teórica, a arquitetura da rede neural e a análise dos resultados.

* **Artigo Completo (PDF):** [Baixar Artigo Científico](https://www.google.com/search?q=./artigo_libras_lens.pdf)

---

## 🚀 Como Rodar o Aplicativo Final

Após realizar a coleta e o treinamento (ou possuir os arquivos `.pth` e `.npy`), inicie a interface garantindo que sua `venv` está ativada:

```powershell
streamlit run app_libras.py
```

---

## 🧠 Detalhes Técnicos

* **Arquitetura LSTM:** Diferente de modelos estáticos, a **Long Short-Term Memory** analisa a "janela temporal". O sistema observa os últimos 15 frames para classificar o movimento.
* **Vetor de Entrada (63):** Cada frame captura 21 marcos (landmarks). Como cada ponto possui eixos $X, Y, Z$, temos $21 \times 3 = 63$ valores por frame.
* **Multimodalidade:** O sistema exibe o texto e utiliza a biblioteca `gTTS` (Google Text-to-Speech) para síntese de voz, promovendo maior acessibilidade.

---

## 📝 Notas de Versão

* **Interface:** Otimizada com `@st.cache_resource` para evitar sobrecarga de memória RAM.
* **Segurança:** Scripts estruturados com blocos `try-except` para evitar interrupções caso a mão saia do campo de visão da câmera.
* **Estabilidade Core:** Importações do MediaPipe ajustadas para contornar falhas de inicialização nativas no Windows (`mediapipe.python.solutions`).

---

## 🙏 Agradecimentos

Gostaríamos de expressar nossa profunda gratidão ao nosso orientador e professor, **Renato Cagnin**, pelas orientações fundamentais, paciência e pelo compartilhamento de conhecimento técnico que tornaram a realização deste projeto de Inteligência Artificial possível.

---

Desenvolvido como projeto de Inteligência Artificial Aplicada — FHO 2026.
