import abc
import torch
from torch import nn
import copy

class Fusion(abc.ABC):
    def __init__(self, num_parties):
        self.name = "fusion"
        self.num_parties = num_parties

    def average_selected_models(self, selected_parties, party_models):
        with torch.no_grad():
            sum_vec = nn.utils.parameters_to_vector(party_models[selected_parties[0]].parameters())
            if len(selected_parties) > 1:
                for i in range(1, len(selected_parties)):
                    sum_vec += nn.utils.parameters_to_vector(party_models[selected_parties[i]].parameters())
                sum_vec /= len(selected_parties)
            model = copy.deepcopy(party_models[0])
            nn.utils.vector_to_parameters(sum_vec, model.parameters())
        return model.state_dict()

    @abc.abstractmethod
    def fusion_algo(self, party_models, current_model=None):
        raise NotImplementedError


class FusionAvg(Fusion):
    def __init__(self, num_parties):
        super().__init__(num_parties)
        self.name = "Fusion-Average"

    def fusion_algo(self, party_models, current_model=None):
        selected_parties = [i for i in range(self.num_parties)]
        return super().average_selected_models(selected_parties, party_models)


class FusionRetrain(Fusion):
    def __init__(self, num_parties):
        super().__init__(num_parties)
        self.name = "Fusion-Retrain"

    def fusion_algo(self, party_models, current_model=None):
        selected_parties = [i for i in range(1, self.num_parties)]
        return super().average_selected_models(selected_parties, party_models)
