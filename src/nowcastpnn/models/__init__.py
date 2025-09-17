import logging

from omegaconf import DictConfig
from torch import nn


from .NowcastPNN import NowcastPNN, NowcastPNNDOW
from .other_models import NowcastLSTM, NowcastMLP

def get_model(cfg: DictConfig) -> nn.Module:
    """
    Model factory. Reads the config and returns an initialized model.
    It inspects `cfg.data.use_dow` to decide which model variant to build.
    """
    model_cfg = cfg.model.params
    model_name = cfg.model.name
    data_cfg = cfg.data

    logging.info(f"Initializing model: {model_name}")

    # Logic to select the correct model class based on the config
    if model_name == "NowcastPNN":
        if data_cfg.use_dow:
            logging.info("Data config specifies use_dow=true. Building NowcastPNNDOW.")
            model = NowcastPNNDOW(
                past_units=data_cfg.past_units,
                max_delay=data_cfg.max_delay,
                hidden_units=model_cfg.hidden_units,
                conv_channels=model_cfg.conv_channels,
                embedding_dim=model_cfg.embedding_dim,
                load_embed=model_cfg.load_embed,
                dropout_probs=model_cfg.dropout_probs
            )
        else:
            logging.info("Data config specifies use_dow=false. Building NowcastPNN.")
            model = NowcastPNNDOW(
                past_units=data_cfg.past_units,
                max_delay=data_cfg.max_delay,
                hidden_units=model_cfg.hidden_units,
                conv_channels=model_cfg.conv_channels,
                dropout_probs=model_cfg.dropout_probs
            )
    elif model_name == "NowcastLSTM":
        # You can add similar logic for your other models if they have DOW variants
        model = NowcastLSTM(

        )
    elif model_name == "NowcastMLP":
        model = NowcastMLP(

        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
