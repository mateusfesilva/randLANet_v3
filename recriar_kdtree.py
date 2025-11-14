# Salve como: recriar_kdtree.py

import numpy as np
import pickle
import argparse
from sklearn.neighbors import KDTree
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Recria os arquivos KD-Tree (.pkl) a partir dos arquivos .bin para resolver problemas de compatibilidade do pickle.")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Caminho para a pasta do dataset processado (a que contém a pasta 'sequences')."
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    if not dataset_path.is_dir():
        print(f"Erro: O diretório especificado não existe: '{dataset_path}'")
        return

    # Encontra todos os arquivos .bin recursivamente
    bin_files = list(dataset_path.glob('**/velodyne/*.bin'))

    if not bin_files:
        print(f"Nenhum arquivo .bin encontrado em '{dataset_path}'")
        return

    print(f"Encontrados {len(bin_files)} arquivos .bin. Recriando as KD-Trees correspondentes...")

    for bin_path in tqdm(bin_files, desc="Recriando KD-Trees"):
        try:
            # 1. Carrega os pontos do arquivo .bin
            # Formato KITTI é (X, Y, Z, Intensidade)
            points_xyzi = np.fromfile(bin_path, dtype=np.float32)
            points_xyz = points_xyzi.reshape(-1, 4)[:, :3]

            # 2. Cria uma nova KD-Tree com a versão atual do scikit-learn
            kdtree = KDTree(points_xyz)

            # 3. Determina o caminho de saída para o arquivo .pkl
            # Ex: .../velodyne/000000.bin -> .../KDTree/000000.pkl
            kdtree_path = bin_path.parent.parent / 'KDTree' / (bin_path.stem + '.pkl')

            # 4. Salva a nova KD-Tree, sobrescrevendo a antiga
            with open(kdtree_path, 'wb') as f:
                pickle.dump(kdtree, f)

        except Exception as e:
            print(f"\nErro ao processar o arquivo {bin_path.name}: {e}")

    print(f"\nProcesso concluído! {len(bin_files)} arquivos KD-Tree foram recriados/atualizados.")

if __name__ == '__main__':
    main()