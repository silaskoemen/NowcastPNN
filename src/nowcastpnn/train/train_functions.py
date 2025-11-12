import torch
import logging

from nowcastpnn.evaluate.losses import get_loss
from .early_stopper import EarlyStopper


def get_y_model_prediction(X, y, model, dow, num_obs, device):
    """ Get model prediction and process it if dow or num_obs is used """
    if num_obs:
        y, _ = y
    if dow:
        X, dow_val = X.copy()
        dist_pred = model(X.to(device), dow_val.to(device))
    else:
        dist_pred = model(X.to(device))
    return y.to(device), dist_pred


def validate_gradients(model):
    valid_gradients = True
    for _, param in model.named_parameters():
        if param.grad is not None:
            valid_gradients = not (torch.isnan(param.grad).any())
            if not valid_gradients:
                return False
    return True


def train(model, train_loader, val_loader, early_stopper: EarlyStopper, loss_fct: str = 'nll', num_epochs: int = 500, learning_rate: float = 3e-4, device = torch.device("mps"), dow: bool = False, num_obs: bool = False, return_history: bool = False):
    """
    Main training loop.
    Returns the best model and a history of training/validation losses.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)  # pyright: ignore[reportPrivateImportUsage]

    history = {"train_loss": [], "val_loss": []}  # Could also use `if` but doesn't hurt

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            y_batch, model_pred = get_y_model_prediction(X_batch, y_batch, model, dow, num_obs, device)
            loss = get_loss(y_batch, model_pred, loss_fct=loss_fct).mean()
            loss.retain_grad()
            loss.backward()

            if not validate_gradients(model):
                logging.warning("Detected inf/NaN values in gradients. Not updating model parameters.")
                optimizer.zero_grad()
                continue
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_batch, model_pred = get_y_model_prediction(X_batch, y_batch, model, dow, num_obs, device)
                loss = get_loss(y_batch, model_pred, loss_fct=loss_fct).mean()
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        if return_history:
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

        if early_stopper(avg_val_loss, model):
            logging.info("Early stopping triggered.")
            break

        logging.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.2f} - Val Loss: {avg_val_loss:.2f} - ES Count: {early_stopper.get_count()}")

    logging.info("Training finished. Loading best model weights.")
    if early_stopper.best_model_state:
        model.load_state_dict(early_stopper.best_model_state)

    if return_history:
        return model, history
    return model
