import numpy as np
import laspy
import pickle
from sklearn.neighbors import KDTree
from pathlib import Path
import argparse
from tqdm import tqdm

def process_test_file(source_path, target_root_path, sequence_id, frame_id):
    """
    Processa um único arquivo .laz de teste e o salva nos formatos .bin e .pkl.
    """
    try:
        # --- Leitura do arquivo .laz ---
        las = laspy.read(source_path)
        points_xyz = las.xyz.astype(np.float32)
        # Garante que a intensidade exista, se não, cria um array de zeros
        if hasattr(las, 'intensity'):
            intensity = (np.array(las.intensity).astype(np.float32) / 65535.0)[:, np.newaxis]
        else:
            intensity = np.zeros((len(points_xyz), 1), dtype=np.float32)

        # --- 1. Salvar arquivo .bin (pontos) ---
        # Formato: [X, Y, Z, Intensidade]
        points_xyzi = np.hstack((points_xyz, intensity))
        bin_path = target_root_path / 'sequences' / sequence_id / 'velodyne' / f'{frame_id:06d}.bin'
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        points_xyzi.tofile(bin_path)

        # --- 2. Gerar e salvar KD-Tree ---
        # A KD-Tree é construída apenas com as coordenadas espaciais (XYZ)
        kdtree = KDTree(points_xyz)
        kdtree_path = target_root_path / 'sequences' / sequence_id / 'KDTree' / f'{frame_id:06d}.pkl'
        kdtree_path.parent.mkdir(parents=True, exist_ok=True)
        with open(kdtree_path, 'wb') as f:
            pickle.dump(kdtree, f)
            
        return True
    except Exception as e:
        print(f"Erro ao processar {source_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Prepara um dataset de teste no formato KITTI-like.")
    parser.add_argument('--source_dir', type=str, required=True, help='Pasta com os arquivos .laz/.las de teste originais.')
    parser.add_argument('--dest_dir', type=str, required=True, help='Pasta de destino para o dataset de teste processado.')
    parser.add_argument('--seq_id', type=str, default='99', help='ID da "sequência falsa" para o conjunto de teste.')
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)
    
    source_files = sorted(list(source_dir.glob('*.laz')) + list(source_dir.glob('*.las')))
    if not source_files:
        print(f"Nenhum arquivo .laz ou .las encontrado em {source_dir}")
        return

    print(f"Encontrados {len(source_files)} arquivos de teste. Processando para a sequência '{args.seq_id}'...")

    # Itera sobre os arquivos e os processa
    for i, file_path in enumerate(tqdm(source_files, desc=f"Processando arquivos de teste")):
        process_test_file(file_path, dest_dir, args.seq_id, i)

    print("\nProcessamento dos dados de teste concluído!")

if __name__ == '__main__':
    main()