import pandas as pd
import glob
import os
from pathlib import Path

def preparar_dataset_heitor():
    print("="*60)
    print("🚀 UNIFICADOR DE DATABASE PROFISSIONAL (PADRÃO HEITOR)")
    print("="*60)
    
    # Caminho configurado
    diretorio_base = Path(r"C:\Users\guisc\Documents\IA_LIBRAS_2026\dataset_imagens")
    arquivos = list(diretorio_base.glob("*.csv"))
    
    # Filtros de segurança
    ignorar = ['banco_dados_libras.csv', 'dataset_mestre.csv']
    arquivos = [f for f in arquivos if f.name not in ignorar]
    
    if not arquivos:
        print(f"❌ [ERRO] Nenhum CSV encontrado em: {diretorio_base}")
        return

    lista_para_unificar = []
    total_linhas = 0

    for arquivo in arquivos:
        try:
            # Carrega o CSV ignorando colunas de índice fantasmas (comuns no banco do Heitor)
            df_temp = pd.read_csv(arquivo)
            
            # 1. Remove colunas 'Unnamed' (lixo de exportação do Pandas)
            df_temp = df_temp.loc[:, ~df_temp.columns.str.contains('^Unnamed')]

            # 2. Extrai a Letra do nome do arquivo
            nome_letra = arquivo.stem.upper() # arquivo.stem pega 'a' de 'a.csv'
            
            # 3. Garante a coluna Label
            if 'label' not in df_temp.columns:
                df_temp.insert(0, 'label', nome_letra)
            else:
                # Garante que a label esteja correta mesmo se o CSV interno estiver errado
                df_temp['label'] = nome_letra
            
            # 4. Verificação de Integridade (21 pontos x 3 coordenadas + label = 64 colunas)
            if df_temp.shape[1] < 64:
                print(f"⚠️ [AVISO] Letra {nome_letra} parece incompleta (colunas: {df_temp.shape[1]}). Pulando...")
                continue

            lista_para_unificar.append(df_temp)
            total_linhas += len(df_temp)
            print(f"✅ {nome_letra.ljust(3)} | Amostras: {len(df_temp):<5} | Status: OK")
            
        except Exception as e:
            print(f"❌ [ERRO] Falha ao processar {arquivo.name}: {e}")

    if lista_para_unificar:
        print("-" * 60)
        print("📦 Consolidando arquivos... Por favor, aguarde.")
        
        # Junta tudo e remove duplicatas acidentais
        df_mestre = pd.concat(lista_para_unificar, ignore_index=True).drop_duplicates()
        
        # Salva o arquivo final
        caminho_saida = Path.cwd() / 'dataset_mestre.csv'
        df_mestre.to_csv(caminho_saida, index=False)
        
        print("="*60)
        print(f"✨ SUCESSO ABSOLUTO!")
        print(f"📂 Arquivo gerado: {caminho_saida}")
        print(f"📊 Total de letras: {len(lista_para_unificar)}")
        print(f"📈 Total de amostras: {len(df_mestre)}")
        print("="*60)
        print("👉 Agora você já pode rodar o '1_coleta_dados.py' com o Juiz!")
    else:
        print("💀 [ERRO] Nenhum dado válido foi processado.")

if __name__ == "__main__":
    preparar_dataset_heitor()