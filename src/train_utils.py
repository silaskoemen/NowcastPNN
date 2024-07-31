import torch
import numpy as np
import torch.nn as nn

def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true.type(torch.int)) # re-cast to int after multiplication with 1000

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

def train(model, num_epochs, train_loader, val_loader, early_stopper, loss_fct = "nll", device = torch.device("mps")):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0003, weight_decay=1e-3)
    early_stopper.reset() # set counter to zero if same instance used for multiple training runs
    for e in range(num_epochs):
        batch_loss = 0.
        model.train()
        for mat, y in train_loader:
            optimizer.zero_grad()
            dist_pred = model(mat)
            loss = get_loss(y, dist_pred, loss_fct=loss_fct).mean()
            loss.retain_grad()
            loss.backward()
            #nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            batch_loss += loss.item()
            #print(f"------------ Loss grad {loss.grad} | data {loss.data} -------------")
            """for name, param in model.named_parameters():
                print(name, param, param.grad)
                #print(name, torch.isfinite(param.grad).all())"""
        
        
        
        batch_loss /= len(train_loader)

        with torch.no_grad(): # performance on test/validation set
            model.eval()
            test_batch_loss = 0.
            for mat, y in val_loader:
                test_pred = model(mat)
                test_loss = get_loss(y, test_pred, loss_fct=loss_fct).mean()
                test_batch_loss += test_loss.item()
            #test_batch_loss /= len(test_loader)
            if early_stopper.early_stop(test_batch_loss, model):
                model.train() # set back to train for sampling
                break
        model.train()
        #if e % 50 == 0 or e == num_epochs-1:
        print(f"Epoch {e+1} - Train loss: {batch_loss:.3} - Val loss: {test_batch_loss:.3} - ES count: {early_stopper.get_count()}")