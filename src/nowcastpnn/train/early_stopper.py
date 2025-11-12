import torch
import copy


class LegacyEarlyStopper:
    """ Class implementing early stopping. Theoretically, PyTorch lightning could be used, but this might be more rubust.

    As seen e.g. in https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch and adapted to include
    restoration of best weights.
    """
    def __init__(self, past_units, max_delay, weeks = False, future_obs = 0, state = "SP", triangle = True, patience = 30, random_split = False, dow = False, n_training = None, biggest_outbreak = False):
        self.patience = patience
        self.counter = 0
        self.min_loss = float('inf')
        self.past_units = past_units
        self.max_delay = max_delay
        self.weeks = weeks
        self.future_obs = future_obs
        self.state = state
        self.triangle = triangle
        self.random_split = random_split
        self.dow = dow
        self.n_training = n_training
        self.biggest_outbreak = biggest_outbreak

    def early_stop(self, val_loss, model):
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.counter = 0
            ## Save best weights
            if self.biggest_outbreak:
                torch.save(model.state_dict(), f"./weights/weights-{self.past_units}-{self.max_delay}-{'week' if self.weeks else 'day'}-fut{self.future_obs}-biggest{'-dow' if self.dow else ''}")
            elif self.n_training is not None:
                torch.save(model.state_dict(), f"./weights/weights-{self.past_units}-{self.max_delay}-{'week' if self.weeks else 'day'}-fut{self.future_obs}{'-rec' if not self.random_split else ''}{'-dow' if self.dow else ''}-{self.n_training}")
            else:
                torch.save(model.state_dict(), f"./weights/weights-{self.past_units}-{self.max_delay}-{'week' if self.weeks else 'day'}-fut{self.future_obs}{'-rec' if not self.random_split else ''}{'-dow' if self.dow else ''}")
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


class EarlyStopper:
    def __init__(self, patience=30, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True # Signal to stop
        return False

    def get_count(self):
        return self.counter
