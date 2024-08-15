import torch
import numpy as np
import torch.nn as nn
from NegativeBinomial import NegBin

class Sampler(object):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SubsetSampler(Sampler):
    r""" Sampler for PyTorch that uses a given subset of indices to sample from. Not random, to use for reproducible samplings
    on the test set.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        for idx in self.indices:
            yield idx


    def __len__(self):
        return len(self.indices)

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
    model.float()
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
            #print(model.attn1.weight.grad)
            #nn.utils.clip_grad_value_(model.parameters(), 1.0)

            ## Check for inf or nan gradients - stop updates in that case
            valid_gradients = True
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                    valid_gradients = not (torch.isnan(param.grad).any())
                    if not valid_gradients:
                        break
            if not valid_gradients:
                print("Detected inf or nan values in gradients. Not updating model parameters.")
                optimizer.zero_grad()
        
            optimizer.step()
            batch_loss += loss.item()

            """for name, param in model.named_parameters():
                print(param)
                print("param.data",torch.isfinite(param.data).all())
                print("param.grad.data",torch.isfinite(param.grad.data).all(),"\n")"""
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



class EarlyStopper:
    """ Class implementing early stopping. Theoretically, PyTorch lightning could be used, but this might be more rubust.
    
    As seen e.g. in https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch and adapted to include 
    restoration of best weights.
    """
    def __init__(self, past_units, max_delay, weeks, future_obs, state, triangle, patience = 30):
        self.patience = patience
        self.counter = 0
        self.min_loss = float('inf')
        self.past_units = past_units
        self.max_delay = max_delay
        self.weeks = weeks
        self.future_obs = future_obs
        self.state = state
        self.triangle = triangle

    def early_stop(self, val_loss, model):
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.counter = 0
            ## Save best weights
            torch.save(model.state_dict(), f"./weights/weights-{self.past_units}-{self.max_delay}-{'week' if self.weeks else 'day'}-{('tri' if self.triangle else 'sum')}-fut{self.future_obs}-{self.state}")
        elif val_loss > self.min_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def get_count(self):
        return self.counter
    
    def get_patience(self):
        return self.patience
    
    def reset(self):
        self.counter = 0