import numpy as np
import pickle
import torch.utils.data as torch_data
from pathlib import Path
from sklearn.neighbors import KDTree

class datasetAereoFullTest(torch_data.Dataset):
    def __init__(self, test_path, config):
        self.name = 'MeuDatasetAereoFullTest'
        self.config = config
        self.path = Path(test_path)
        
        # Encontra todos os arquivos .bin na pasta velodyne da sequência de teste
        self.file_list = sorted(list(self.path.glob('velodyne/*.bin')))
        print(f"Encontrados {len(self.file_list)} arquivos de teste para inferência completa.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        # Pega o caminho do arquivo .bin
        bin_path = self.file_list[item]
        
        # Carrega os pontos do arquivo .bin (com todas as features, ex: XYZI)
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, self.config.num_features).copy()
        
        # Carrega a KD-Tree correspondente (que foi feita apenas com XYZ)
        pkl_path = bin_path.parent.parent / 'KDTree' / (bin_path.stem + '.pkl')
        with open(pkl_path, 'rb') as f:
            search_tree = pickle.load(f)
                
        # Retorna a nuvem de pontos COMPLETA, a árvore e o caminho do arquivo
        return points, search_tree, str(bin_path)