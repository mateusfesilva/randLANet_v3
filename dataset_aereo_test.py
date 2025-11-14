# Salve como: dataset/dataset_aereo_test.py

import numpy as np
import pickle
import torch
import torch.utils.data as torch_data
from pathlib import Path
from sklearn.neighbors import KDTree

# Importe sua configuração
from utils.config import ConfigSemanticKITTI as cfg
# Importe suas funções de processamento de dados
from utils.data_process import DataProcessing as DP


class datasetAereoTest(torch_data.Dataset):
    def __init__(self, test_path, config):
        """
        test_path: Caminho para a pasta da sequência de teste (ex: .../sequences/99)
        """
        self.name = 'MeuDatasetAereoTest'
        self.config = config
        
        # Encontra todos os arquivos .bin na pasta velodyne da sequência de teste
        self.file_list = sorted(list(Path(test_path).glob('velodyne/*.bin')))
        print(f"Encontrados {len(self.file_list)} arquivos de teste em '{test_path}'.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        """
        Carrega um arquivo de teste e retorna uma amostra de tamanho fixo.
        """
        # Carrega os dados e a KD-Tree
        points_and_features, tree, file_path = self.get_data(item)

        # --- CORREÇÃO PARA NUVENS PEQUENAS (ValueError) ---
        # Verifica se a nuvem de pontos carregada é menor que o tamanho da amostra
        if len(points_and_features) < self.config.num_points:
            # Se for menor, duplica pontos aleatoriamente para atingir o tamanho necessário
            indices = np.random.choice(len(points_and_features), self.config.num_points, replace=True)
            points_and_features = points_and_features[indices]

            # CRÍTICO: Recria a KD-Tree com a nuvem aumentada (apenas com XYZ)
            tree = KDTree(points_and_features[:, :3])
        # --- FIM DA CORREÇÃO ---

        # Pega um ponto aleatório como centro e recorta uma sub-nuvem
        pick_idx = np.random.choice(len(points_and_features), 1)
        selected_points, selected_idx = self.crop_pc(points_and_features, tree, pick_idx)
    
        return {
            'points': selected_points,
            'indices': selected_idx,
            'file_path': file_path
        }

    def get_data(self, item):
        # Pega o caminho do arquivo .bin
        bin_path = self.file_list[item]
        
        # Carrega os pontos do arquivo .bin (com todas as features, ex: XYZI)
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, self.config.num_features)
        
        # Carrega a KD-Tree correspondente (que foi feita apenas com XYZ)
        pkl_path = bin_path.parent.parent / 'KDTree' / (bin_path.stem + '.pkl')
        with open(pkl_path, 'rb') as f:
            search_tree = pickle.load(f)
        
        return points, search_tree, str(bin_path)

    @staticmethod
    def crop_pc(points, search_tree, pick_idx):
        # Usa apenas as 3 primeiras colunas (XYZ) do ponto central para a busca
        center_point = points[pick_idx, :3].reshape(1, -1)
        
        # Encontra os vizinhos na árvore
        select_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]
        
        # Embaralha os índices por segurança
        select_idx = DP.shuffle_idx(select_idx)
        
        # Seleciona os pontos completos (com todas as features) usando os índices
        select_points = points[select_idx]
        
        return select_points, select_idx


def tf_map_test(batch_pc):
    """
    Função auxiliar para preparar os dados geométricos para a RandLA-Net.
    É uma versão do tf_map de treino, mas sem os rótulos.
    """
    features = batch_pc
    input_points = []
    input_neighbors = []
    input_pools = []
    input_up_samples = []

    for i in range(cfg.num_layers):
        neighbour_idx = DP.knn_search(batch_pc[:, :, :3], batch_pc[:, :, :3], cfg.k_n)
        sub_points = batch_pc[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
        pool_i = neighbour_idx[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
        up_i = DP.knn_search(sub_points[:, :, :3], batch_pc[:, :, :3], 1)
        input_points.append(batch_pc)
        input_neighbors.append(neighbour_idx)
        input_pools.append(pool_i)
        input_up_samples.append(up_i)
        batch_pc = sub_points

    input_list = input_points + input_neighbors + input_pools + input_up_samples
    input_list += [features]

    return input_list


def collate_fn_test(batch):
    """
    Função collate para o teste. Agrupa as amostras em um lote e prepara
    os dados de entrada para o modelo.
    """
    selected_pc = np.stack([item['points'] for item in batch])
    
    # O resto das chaves é mais para referência e não entra no modelo diretamente
    selected_idx = [item['indices'] for item in batch]
    original_file_paths = [item['file_path'] for item in batch]

    # Prepara os inputs geométricos para a rede
    flat_inputs = tf_map_test(selected_pc)

    # Organiza em um dicionário que o modelo espera
    num_layers = cfg.num_layers
    inputs = {}
    inputs['xyz'] = [torch.from_numpy(i).float() for i in flat_inputs[:num_layers]]
    inputs['neigh_idx'] = [torch.from_numpy(i).long() for i in flat_inputs[num_layers: 2 * num_layers]]
    inputs['sub_idx'] = [torch.from_numpy(i).long() for i in flat_inputs[2 * num_layers:3 * num_layers]]
    inputs['interp_idx'] = [torch.from_numpy(i).long() for i in flat_inputs[3 * num_layers:4 * num_layers]]
    inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()

    return inputs, original_file_paths