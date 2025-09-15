from omegaconf import DictConfig
from .NowcastPNN import NowcastPNN, NowcastPNNDOW
from .other_models import NowcastLSTM, NowcastMLP

def get_model(cfg: DictConfig):
    """
    Model factory. Reads the config and returns an initialized model.
    It inspects `cfg.data.use_dow` to decide which model variant to build.
    """
    model_cfg = cfg.model
    data_cfg = cfg.data

    print(f"Initializing model: {model_cfg.name}")

    # Logic to select the correct model class based on the config
    if model_cfg.name == "NowcastPNN":
        if data_cfg.use_dow:
            print("Data config specifies use_dow=true. Building NowcastPNNDOW.")
            model_class = NowcastPNNDOW
        else:
            print("Data config specifies use_dow=false. Building NowcastPNN.")
            model_class = NowcastPNN
    elif model_cfg.name == "NowcastLSTM":
        # You can add similar logic for your other models if they have DOW variants
        model_class = NowcastLSTM
    elif model_cfg.name == "NowcastMLP":
        model_class = NowcastMLP
    else:
        raise ValueError(f"Unknown model name: {model_cfg.name}")

    # Pass relevant parameters to the constructor
    # This assumes your model constructors can handle the params from the config
    model = model_class(**model_cfg.params, **data_cfg)

    return model
