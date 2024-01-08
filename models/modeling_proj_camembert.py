from torch import nn

from transformers import CamembertModel, CamembertForMaskedLM, CamembertConfig


class ProjCamembertConfig(CamembertConfig):
    def __init__(self,
                 projection_size: int = 512,
                 **kwargs):

        model_type = "proj_camembert"

        super().__init__(**kwargs)
        self.projection_size = projection_size


class ProjCamembertModel(CamembertModel):
    config: ProjCamembertConfig

    def __init__(self, config: ProjCamembertConfig):
        super().__init__(config)

        self.projection = nn.Linear(config.hidden_size, config.projection_size)


class ProjCamembertForMaskedLM(CamembertForMaskedLM):
    config: ProjCamembertConfig

    def __init__(self, config: ProjCamembertConfig):
        super().__init__(config)

        self.projection = nn.Linear(config.hidden_size, config.projection_size)
