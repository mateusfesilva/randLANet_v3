import torch
import numpy as np
from sklearn.metrics import confusion_matrix


class IoUCalculator:
    def __init__(self, cfg):
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg

    def add_data(self, end_points):
        # Pega os dados brutos do dicionário
        logits = end_points['logits']
        labels = end_points['labels']
        preds = torch.argmax(logits, dim=1)

        # Transfere os tensores para a CPU e converte para NumPy
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # --- LÓGICA DE FILTRAGEM ---
        # Ignora o rótulo 255 (ou qualquer outro definido na config, se houver)
        # A função de perda já usa 'ignore_index', mas para as métricas, filtramos aqui.
        # O ideal é ter a config acessível, mas vamos usar 255 como padrão de ignorar.
        ignore_idx = 255
        valid_mask = (labels_np != ignore_idx)

        labels_filtered = labels_np[valid_mask]
        preds_filtered = preds_np[valid_mask]
        # ---------------------------

        # Atualiza a matriz de confusão apenas com os dados válidos
        conf_matrix = confusion_matrix(labels_filtered, preds_filtered, labels=np.arange(0, self.cfg.num_classes, 1))

        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        # 'gt_classes' é o total de pontos verdadeiros para cada classe
        # 'positive_classes' é o total de pontos previstos para cada classe
        # 'true_positive_classes' é a intersecção (TP)
        for n in range(self.cfg.num_classes):
            # A união é a soma de todos os pontos (verdadeiros + previstos) menos a intersecção
            union = self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]
            iou = self.true_positive_classes[n] / (union + 1e-6) # Adiciona epsilon para evitar divisão por zero
            iou_list.append(iou)

        # Calcula o mean IoU apenas para as classes que não devem ser ignoradas
        # Assumindo que a classe 0 do NOVO índice é uma classe válida, como "solo"
        # Se você tivesse um ignore_index no YAML para métricas, usaria aqui.
        # Por agora, calculamos a média de todas as classes.
        mean_iou = np.mean(iou_list)

        return mean_iou, iou_list


# Esta função 'compute_acc' agora é redundante se IoUCalculator já faz o trabalho
# mas vamos corrigi-la para não dar erro.
def compute_acc(end_points):
    # Pega os dados diretamente do dicionário
    logits = end_points['logits']
    labels = end_points['labels']
    preds = torch.argmax(logits, dim=1)

    # A lógica de acurácia pode ser adicionada aqui, mas
    # o IoU é a métrica principal.
    # Vamos apenas retornar o end_points para o fluxo continuar.
    end_points['preds'] = preds
    return 0.0, end_points # Retorna uma acurácia dummy de 0.0