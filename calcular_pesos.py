import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# --- 1. CONFIGURE AS VARIÁVEIS ABAIXO ---

# O ID da "sequência falsa" que você designou para os dados de TREINAMENTO
TRAIN_SEQUENCE_ID = '00'

# O número total de classes no seu dataset
NUM_CLASSES = 3

# --- FIM DA CONFIGURAÇÃO ---


def main():
    """
    Este script calcula os pesos para cada classe com base na sua frequência
    no conjunto de treinamento. Classes com menos pontos receberão um peso maior.
    """
    
    parser = argparse.ArgumentParser(description="Este script calcula os pesos para cada classe com base na sua frequência no conjunto de treinamento.")
    parser.add_argument(
        "input_path",
        type=str,
        help="Caminho para um único arquivo .label ou para um diretório contendo arquivos .label."
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    
    
    # Monta o caminho para a pasta de rótulos do conjunto de treino
    labels_dir = input_path / 'sequences' / TRAIN_SEQUENCE_ID / 'labels'
    
    # Encontra todos os arquivos .label na pasta
    label_files = list(labels_dir.glob('*.label'))
    
    if not label_files:
        print(f"Erro: Nenhum arquivo .label foi encontrado em '{labels_dir}'")
        print("Verifique se o caminho e o ID da sequência estão corretos.")
        return

    # Inicializa um array para contar a ocorrência de cada classe
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_points = 0

    print(f"Analisando {len(label_files)} arquivos de rótulos do conjunto de treinamento...")

    # Itera sobre cada arquivo de rótulo para contar os pontos
    for label_path in tqdm(label_files, desc="Contando pontos"):
        # Carrega os rótulos do arquivo binário
        labels = np.fromfile(label_path, dtype=np.uint32)
        
        # Encontra os rótulos únicos e suas contagens no arquivo atual
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Adiciona as contagens às contagens totais
        for label, count in zip(unique_labels, counts):
            if label < NUM_CLASSES: # Garante que o rótulo está dentro do intervalo esperado
                class_counts[label] += count
        
        total_points += len(labels)

    print("\nAnálise concluída.")
    print("-" * 30)
    print(f"Total de pontos no conjunto de treino: {total_points}")
    print("Contagem de pontos por classe:")
    for i, count in enumerate(class_counts):
        print(f"  Classe {i}: {count} pontos")
    print("-" * 30)

    # --- Cálculo dos Pesos ---
    # Fórmula de ponderação de classe inversa (inverse class frequency)
    # A ideia é que o peso seja inversamente proporcional à frequência da classe.
    class_weights = np.zeros(NUM_CLASSES, dtype=np.float32)

    for i in range(NUM_CLASSES):
        # Evita divisão por zero se uma classe não aparecer no dataset de treino
            if class_counts[i] > 0:
                class_weights[i] = total_points / (NUM_CLASSES * class_counts[i])
            else:
                # Se a classe não existe, o peso é zero
                class_weights[i] = 0.0

    print("Pesos calculados para cada classe:")
    print(class_weights)
    print("-" * 30)

    # Salva o array de pesos em um arquivo .npy
    output_path = 'meudataset_aereo_class_weights.npy'
    np.save(output_path, class_weights)
    
    print(f"Pesos salvos com sucesso em: '{output_path}'")
    print("Este arquivo agora pode ser carregado pela sua classe de Dataset.")

if __name__ == '__main__':
    main()