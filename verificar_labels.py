# Salve como: verificar_labels.py

import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def analyze_label_file(file_path):
    """
    Lê um único arquivo .label e imprime um resumo das classes encontradas.
    """
    try:
        print("-" * 50)
        print(f"Analisando o arquivo: {file_path.name}")

        # 1. Lê o arquivo binário como um array de inteiros de 32 bits
        labels = np.fromfile(file_path, dtype=np.uint32)

        if labels.size == 0:
            print("  -> Arquivo está vazio ou não pôde ser lido.")
            return

        # 2. Encontra as classes únicas e conta a ocorrência de cada uma
        unique_classes, counts = np.unique(labels, return_counts=True)
        
        # 3. Imprime o resumo
        print(f"  Total de pontos: {len(labels):,}")
        print(f"  Quantidade de classes distintas: {len(unique_classes)}")
        print(f"  IDs das classes encontradas: {unique_classes}")
        print(f"  Rótulo Mínimo: {np.min(labels)}, Rótulo Máximo: {np.max(labels)}")
        print("  Distribuição de pontos por classe:")
        
        for class_id, count in zip(unique_classes, counts):
            percentage = (count / len(labels)) * 100
            print(f"    - Classe {class_id}: {count:>10,} pontos ({percentage:.2f}%)")

    except Exception as e:
        print(f"  -> Erro ao processar o arquivo {file_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Verifica o conteúdo de arquivos .label binários.")
    parser.add_argument(
        "input_path",
        type=str,
        help="Caminho para um único arquivo .label ou para um diretório contendo arquivos .label."
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"Erro: O caminho especificado não existe: '{input_path}'")
        return

    if input_path.is_file():
        files_to_analyze = [input_path]
    elif input_path.is_dir():
        # Procura por arquivos .label recursivamente dentro do diretório
        files_to_analyze = list(input_path.glob('**/*.label'))
        if not files_to_analyze:
            print(f"Nenhum arquivo .label encontrado no diretório: '{input_path}'")
            return
    else:
        print(f"Erro: O caminho especificado не é um arquivo nem um diretório válido.")
        return
        
    for file_path in tqdm(files_to_analyze, desc="Analisando arquivos"):
        analyze_label_file(file_path)

if __name__ == '__main__':
    main()