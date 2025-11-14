# Salve como: corrigir_labels.py

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def main():
    # parser = argparse.ArgumentParser(description="Corrige os rótulos nos arquivos .label, subtraindo 1 de cada valor.")
    # parser.add_argument(
    #     "dataset_path",
    #     type=str,
    #     help="Caminho para a pasta do dataset processado (a que contém a pasta 'sequences')."
    # )
    # args = parser.parse_args()

    dataset_path = Path('E:\MATEUS\CNN_MODEL\ECLAIR\dataset\datasetorg')

    if not dataset_path.is_dir():
        print(f"Erro: O diretório especificado не existe: '{dataset_path}'")
        return

    # Procura por todos os arquivos .label recursivamente
    label_files = list(dataset_path.glob('**/*.label'))

    if not label_files:
        print(f"Nenhum arquivo .label encontrado em '{dataset_path}'")
        return

    print(f"Encontrados {len(label_files)} arquivos .label para corrigir...")
    
    modified_count = 0
    for file_path in tqdm(label_files, desc="Corrigindo rótulos"):
        try:
            # Carrega os rótulos do arquivo
            labels = np.fromfile(file_path, dtype=np.uint32)

            # Verifica se há algo a fazer
            if labels.size > 0:
                # A MÁGICA ACONTECE AQUI: subtrai 1 de todos os rótulos de uma vez
                corrected_labels = labels - 1
                
                # Salva o arquivo de volta, sobrescrevendo o original
                corrected_labels.astype(np.uint32).tofile(file_path)
                modified_count += 1

        except Exception as e:
            print(f"\nErro ao processar o arquivo {file_path.name}: {e}")
            
    print(f"\nCorreção concluída! {modified_count} arquivos foram modificados.")

if __name__ == '__main__':
    main()