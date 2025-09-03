from .NowcastPNN import NowcastPNN, NowcastPNNDOW

def get_model(cfg):
    name = cfg.model.name
    if name == "NowcastPNN":
        return NowcastPNN(**cfg.params)
    elif name == "NowcastPNNDOW":
        return NowcastPNNDOW(**cfg.params)
    else:
        raise ValueError(f"Model {name} not recognized.")
