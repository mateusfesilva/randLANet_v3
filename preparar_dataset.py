import numpy as np
import laspy
import pickle
import yaml
import argparse
from sklearn.neighbors import KDTree
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def process_file(source_path, target_root_path, sequence_id, frame_id, remap_lut):
    """
    Processa um único arquivo .laz e o salva nos formatos .bin, .label e .pkl,
    aplicando o remapeamento de rótulos.
    """
    try:
        # --- Leitura do arquivo .laz ---
        las = laspy.read(source_path)
        points_xyz = las.xyz.astype(np.float32)
        intensity = (np.array(las.intensity).astype(np.float32) / 65535.0)[:, np.newaxis]
        original_labels = np.array(las.classification).astype(np.int32)

        # --- APLICAÇÃO DO REMAPEAMENTO (PASSO CRUCIAL) ---
        # Garante que o remap_lut seja grande o suficiente
        if np.max(original_labels) >= len(remap_lut):
             print(f"Atenção: Rótulo máximo no arquivo {source_path.name} ({np.max(original_labels)}) é maior que o tamanho do mapa ({len(remap_lut)}). Verifique seu YAML.")
             # Opção para pular arquivos problemáticos
             # return False
        
        remapped_labels = remap_lut[original_labels].astype(np.uint32)

        # --- 1. Salvar arquivo .bin (pontos) ---
        points_xyzi = np.hstack((points_xyz, intensity))
        bin_path = target_root_path / 'sequences' / sequence_id / 'velodyne' / f'{frame_id:06d}.bin'
        points_xyzi.tofile(bin_path)

        # --- 2. Salvar arquivo .label (rótulos JÁ REMAPEADOS) ---
        label_path = target_root_path / 'sequences' / sequence_id / 'labels' / f'{frame_id:06d}.label'
        remapped_labels.tofile(label_path)

        # --- 3. Gerar e salvar KD-Tree ---
        kdtree = KDTree(points_xyz)
        kdtree_path = target_root_path / 'sequences' / sequence_id / 'KDTree' / f'{frame_id:06d}.pkl'
        with open(kdtree_path, 'wb') as f:
            pickle.dump(kdtree, f)
            
        return True
    except Exception as e:
        print(f"Erro ao processar {source_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Prepara um dataset customizado no formato KITTI-like, aplicando o remapeamento de classes a partir de um arquivo YAML.")
    parser.add_argument('--source_dir', type=str, required=True, help='Pasta com os arquivos .laz/.las originais.')
    parser.add_argument('--dest_dir', type=str, required=True, help='Pasta de destino para o dataset processado.')
    parser.add_argument('--yaml_config', type=str, required=True, help='Caminho para o arquivo de configuração .yaml.')
    args = parser.parse_args()

    print("Carregando o arquivo de configuração YAML...")
    DATA = yaml.safe_load(open(args.yaml_config, 'r'))
    remap_dict = DATA["learning_map"]
    max_key = max(remap_dict.keys())
    remap_lut = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut[list(remap_dict.keys())] = list(remap_dict.values())
    print("Mapa de remapeamento de classes criado.")

    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)
    
    source_files = list(source_dir.glob('*.laz')) + list(source_dir.glob('*.las'))
    if not source_files:
        print(f"Nenhum arquivo .laz ou .las encontrado em {source_dir}")
        return

    train_files, val_files = train_test_split(source_files, test_size=0.2, random_state=42)

    splits = {'00': train_files, '01': val_files}

    for seq_id, file_list in splits.items():
        print(f"\nProcessando sequência '{seq_id}' ({'treino' if seq_id == '00' else 'validação'})...")
        
        (dest_dir / 'sequences' / seq_id / 'velodyne').mkdir(parents=True, exist_ok=True)
        (dest_dir / 'sequences' / seq_id / 'labels').mkdir(parents=True, exist_ok=True)
        (dest_dir / 'sequences' / seq_id / 'KDTree').mkdir(parents=True, exist_ok=True)

        for i, file_path in enumerate(tqdm(file_list, desc=f"Sequência {seq_id}")):
            process_file(file_path, dest_dir, seq_id, i, remap_lut)

    print("\nProcessamento concluído!")

if __name__ == '__main__':
    main()