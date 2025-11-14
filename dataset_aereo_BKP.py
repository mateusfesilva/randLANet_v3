from utils.data_process import DataProcessing as DP
from utils.config import cfg
from os.path import join
import numpy as np
import pickle
import torch.utils.data as torch_data
import torch
import os
from pathlib import Path
from scipy.spatial import KDTree

class datasetAereo(torch_data.Dataset):
    def tf_map(self, batch_pc, batch_label, batch_pc_idx, batch_cloud_idx):
        features = batch_pc
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = DP.knn_search(batch_pc, batch_pc, cfg.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            up_i = DP.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx]

        return input_list

    def collate_fn(self, batch):

        selected_pc, selected_labels, selected_idx, cloud_ind = [], [], [], []
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])

        selected_pc = np.stack(selected_pc)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)

        flat_inputs = self.tf_map(selected_pc, selected_labels, selected_idx, cloud_ind)

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()

        return inputs

    def __init__(self, mode, data_list=None):
        self.name = 'MeuDatasetAereo' # Nome atualizado
        # Caminho corrigido para Windows (usando Path para robustez)
        self.dataset_path = cfg.dataset_path # Usando o caminho do cfg
        self.num_classes = cfg.num_classes
        # Verifique se a classe 0 deve ser ignorada no seu dataset
        self.ignored_labels = np.sort([0])

        self.mode = mode
        if data_list is None:
            if mode == 'training':
                seq_list = ['00'] # Sequência de treino
            elif mode == 'validation':
                seq_list = ['01'] # Sequência de validação
            else:
                raise ValueError(f"Modo '{mode}' desconhecido.")
            self.data_list = DP.get_file_list(str(self.dataset_path), seq_list)
        else:
            self.data_list = data_list
        self.data_list = sorted(self.data_list)

    def get_class_weight(self):
        # Carrega os pesos pré-calculados de forma segura
        weights_path = 'meudataset_aereo_class_weights.npy'
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Arquivo de pesos '{weights_path}' não encontrado. "
                "Execute o script 'calcular_pesos.py' primeiro."
            )
        return np.load(weights_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        # Renomeado 'spatially_regular_gen' para '_get_one_sample' para clareza
        selected_pc, selected_labels, selected_idx, cloud_ind = self._get_one_sample(item)
        return selected_pc, selected_labels, selected_idx, cloud_ind

    @staticmethod
    def crop_pc(points, labels, search_tree, pick_idx):
        """
        Esta função recorta uma sub-nuvem de pontos de tamanho fixo
        ao redor de um ponto central.
        """
        # crop a fixed size point cloud for training
        center_point = points[pick_idx, :].reshape(1, -1)
        select_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]
        select_idx = DP.shuffle_idx(select_idx)
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        return select_points, select_labels, select_idx

    def _get_one_sample(self, item):
        cloud_ind = item
        pc_path = self.data_list[cloud_ind]
        pc, tree, labels = self.get_data(pc_path)
            # --- CORREÇÃO PARA NUVENS PEQUENAS (ValueError) ---
        # Verifica se a nuvem de pontos carregada é menor que o tamanho da amostra exigido
        if len(pc) < cfg.num_points:
            # Se for menor, duplica pontos aleatoriamente para atingir o tamanho necessário.
            # O 'replace=True' permite que os mesmos pontos sejam escolhidos mais de uma vez.
            indices = np.random.choice(len(pc), cfg.num_points, replace=True)
            pc = pc[indices]
            labels = labels[indices]
            
            # CRÍTICO: Recria a KD-Tree com a nuvem de pontos agora aumentada
            tree = KDTree(pc)
        
        # Pega um ponto aleatório como centro e recorta uma sub-nuvem
        pick_idx = np.random.choice(len(pc), 1)
        selected_pc, selected_labels, selected_idx = datasetAereo.crop_pc(pc, labels, tree, pick_idx)

        return selected_pc, selected_labels, selected_idx, np.array([cloud_ind], dtype=np.int32)

    def get_data(self, file_path):
        seq_id = file_path[0]
        frame_id = file_path[1]
        
        # Carrega a KD-Tree (que contém os pontos)
        kd_tree_path = self.dataset_path / seq_id / 'KDTree' / f'{frame_id}.pkl'
        with open(kd_tree_path, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)

        # Carrega os rótulos do arquivo .label (formato binário)
        label_path = self.dataset_path / seq_id / 'labels' / f'{frame_id}.label'
        labels = np.fromfile(label_path, dtype=np.uint32)
        
        return points, search_tree, labels

    # ... os métodos crop_pc, tf_map, e collate_fn permanecem os mesmos ...
    # (O resto do seu código já estava correto para esses métodos)