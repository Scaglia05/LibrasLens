import pandas as pd
import glob
import os
from pathlib import Path

def preparar_dataset_heitor():
    print("="*60)
    print("🚀 UNIFICADOR DE DATABASE PROFISSIONAL (PADRÃO HEITOR - CLEAN)")
    print("="*60)
    
    diretorio_base = Path(r"C:\Users\guisc\Documents\IA_LIBRAS_2026\dataset_imagens")
    arquivos = list(diretorio_base.glob("*.csv"))
    
    ignorar = ['banco_dados_libras.csv', 'dataset_mestre.csv']
    arquivos = [f for f in arquivos if f.name not in ignorar]
    
    if not arquivos:
        print(f"❌ [ERRO] Nenhum CSV encontrado em: {diretorio_base}")
        return

    lista_para_unificar = []

    for arquivo in arquivos:
        try:
            df_temp = pd.read_csv(arquivo)
            nome_letra = arquivo.stem.upper()
            
            # 1. Remove colunas 'Unnamed'
            df_temp = df_temp.loc[:, ~df_temp.columns.str.contains('^Unnamed')]

            # 2. LIMPEZA DE NaNs (AQUI ESTÁ A MÁGICA)
            # Remove a linha se houver QUALQUER buraco nos dados
            antes = len(df_temp)
            df_temp = df_temp.dropna()
            depois = len(df_temp)
            
            # 3. REDUÇÃO PARA O JUIZ (63 coordenadas + Label)
            # Se o arquivo do Heitor tem 1638 colunas, pegamos só o primeiro frame
            if 'label' in df_temp.columns:
                col_label = df_temp['label']
                # Pega as primeiras 63 colunas de dados (excluindo a label original)
                df_dados = df_temp.drop('label', axis=1).iloc[:, :63]
            else:
                df_dados = df_temp.iloc[:, :63]
            
            # Reconstrói o DataFrame com a Label na frente
            df_final_letra = pd.DataFrame(df_dados.values)
            df_final_letra.insert(0, 'label', nome_letra)
            
            if depois > 0:
                lista_para_unificar.append(df_final_letra)
                status_limpeza = f"Limpas: {antes - depois}" if antes - depois > 0 else "Integro"
                print(f"✅ {nome_letra.ljust(3)} | Amostras: {depois:<5} | {status_limpeza}")
            
        except Exception as e:
            print(f"❌ [ERRO] Falha ao processar {arquivo.name}: {e}")

    if lista_para_unificar:
        print("-" * 60)
        print("📦 Consolidando dados puros... Por favor, aguarde.")
        
        df_mestre = pd.concat(lista_para_unificar, ignore_index=True).drop_duplicates()
        
        caminho_saida = Path.cwd() / 'dataset_mestre.csv'
        df_mestre.to_csv(caminho_saida, index=False)
        
        print("="*60)
        print(f"✨ SUCESSO! O Juiz agora tem apenas dados de alta qualidade.")
        print(f"📂 Arquivo: {caminho_saida}")
        print(f"📊 Colunas finais: {df_mestre.shape[1]} (Label + 63 Coords)")
        print(f"📈 Total de amostras: {len(df_mestre)}")
        print("="*60)
    else:
        print("💀 [ERRO] Nenhum dado restou após a limpeza.")

if __name__ == "__main__":
    preparar_dataset_heitor()