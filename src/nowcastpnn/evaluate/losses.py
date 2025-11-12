def nll(y_true, y_pred):
    nll_loss = -y_pred.log_prob(y_true)
    return nll_loss

def hybrid_loss(y_true, y_pred):
    """ Add negative log likelihood to percentage error to encourage
    accurate predictions of smaller values
    """
    return nll(y_true=y_true, y_pred=y_pred) + abs(y_true - y_pred.mode)

def mae(y_true, y_pred):
    return abs(y_true - y_pred.mode)

def mse(y_true, y_pred):
    return (y_true - y_pred.mode).pow(2)

def get_loss(y_true, y_pred, loss_fct):
    match loss_fct:
        case "nll":
            return nll(y_true, y_pred)
        case "hybrid":
            return hybrid_loss(y_true, y_pred)
        case "mae":
            return mae(y_true, y_pred)
        case "mse":
            return mse(y_true, y_pred)
    raise ValueError(f"Loss function {loss_fct} not supported. Choose one of hybrid, nll, mse or mae.")
