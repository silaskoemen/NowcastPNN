import numpy as np
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
from scipy import stats
import torch
import pandas as pd
#plt.rcParams['font.family'] = "cmr10"
plt.rcParams['font.serif'] = "Computer Modern"

def test_plot():
    plt.plot([1, 5.4, 4, 9.2, 19])
    plt.show()

## Make plot over entire dataset for desired confidence level
def plot_entire_confints(dataset, model, n_samples = 200, levels = [0.5, 0.95], weeks = True, xlims = None, random_split = True, test_idcs = None):
    model.train()
    model = model.to("cpu")
    plotloader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
    mat, y = next(iter(plotloader))
    mat, y = mat.to("cpu"), y.to("cpu").numpy()
    preds = np.zeros((y.shape[0], n_samples))
    for i in range(n_samples):
        preds[:, i] = model(mat).sample().numpy()
    preds_mean = np.quantile(preds, 0.5, axis=1)#np.mean(preds, axis = 1)
    #preds_mean = torch.mode(preds, dim=1).values#.to_numpy()#np.mean(preds, axis = 1)#stats.mode(preds, axis=0)[0]
    
    intervals_dict = {}
    for l in levels:
        intervals_dict[l] = (np.quantile(preds, (1-l)/2, 1), np.quantile(preds, (1+l)/2, 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(y, label=r"True count", c = "black")
    #plt.plot(y_atm, label="reported on day", c = "darkgrey")
    plt.plot(preds_mean, label = r"Nowcast predictions", c = "crimson", alpha = 0.75)
    for l in levels:
        lower, upper = intervals_dict[l]
        plt.fill_between(range(len(y)), lower, upper, color = "crimson", alpha = 0.2, label = f"{int(100*l)}% CI")
    plt.grid(alpha=.2)
    if not random_split:
        if weeks:
            plt.axvline(300, color = "black", label=r"division train/test", linestyle="--")
        else:
            plt.axvline(2100, color = "black", label=r"division train/test", linestyle="--")
    if test_idcs is not None:
        plt.vlines(test_idcs, ymin=0, ymax=1000, linewidth = 0.1, colors = "red", label = "Test set")
    plt.xlabel(fr"{'EpiWeeks' if weeks else 'Days'} since start of observation")
    plt.legend()
    plt.ylabel(r"Number of cases")
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    if xlims is not None:
        plt.xlim(xlims)
        plt.savefig(fr"../outputs/figures/nowcast_{'week' if weeks else 'day'}_subset_{xlims[0]}_{xlims[1]}")
    else:
        plt.savefig(fr"../outputs/figures/nowcast_{'week' if weeks else 'day'}")
    plt.show()


def plot_distance_true_observed(df: pd.DataFrame, idx: str = 100, horizon: int = 30, future_units = 0, start_date: str = "2013-01-01", weeks = False) -> None:
    """ For specific index (specific date), calculate the difference between true counts versus observed at date
    
    Args:
        :df: [pd.DataFrame]: DataFrame containing the counts at all delay values for all dates
        :idx: [int]: Index for date to plot from, should be bigger than horizon
        :horizon: [int]: Time horizon steps to go back
        :start_date: [str]: Starting date of observations, so labels can be done with true dates
    """
    assert idx > horizon, "Observation index should be larger than horizon to go backwards"
    if isinstance(df, pd.DataFrame): df = np.array(df.values, dtype = np.float32)

    df = df[(idx-horizon+1):(idx+future_units+1), :]
    max_delay = df.shape[1]
    y_true = df.sum(axis = 1)

    mask = np.zeros((horizon+future_units, max_delay), dtype=bool)
    for p in range(max_delay):
        for d in range(max_delay):
            if p + d >= max_delay:
                mask[p+(horizon-max_delay), d] = True
    df[mask] = 0.
    df[(idx+1):-1, :] = 0.

    y_obs = df.sum(axis = 1)

    plt.figure(figsize=(9, 5))
    plt.plot(y_true, label="True count", color = "black")
    plt.plot(y_obs, label=f"Observed on day {idx}", color = "crimson") # convert with start date to day and then plot with months
    if weeks:
        plt.xlabel("EpiWeeks in the past")
    else:
        plt.xlabel("Days in the past")
    plt.xticks([*range(horizon)], [*range(horizon-1, -1, -1)])
    plt.axvline(horizon-1, color = "black", linestyle="--", label="Current day")
    plt.ylabel("Number of cases")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


def plot_past_correction(model, past_units, max_delay, future_obs, weeks, dataset, padding = "both", padding_val = 0, n_samples = 200, levels = [0.5, 0.95], state = "SP", idx = 787, test_idcs = None):
    model.eval() # sets batch norm to eval so a single entry can be passed without issues of calculating mean and std.
    model.drop1.train() # keeps dropout layers active
    model.drop2.train()
    if padding is None or padding == "none": left, right = False, False
    elif padding == "both": left, right = True, True
    elif padding == "left": left, right = True, False
    else: left, right = False, True

    model = model.to("cpu")
    if idx is None and test_idcs is not None:
        idx_current = np.random.choice(test_idcs) # plot any of the test indices
    elif idx is not None and test_idcs is None:
        idx_current = idx
    else:
        raise AssertionError("Provide just one of idx or test_idcs, not both")
    mat, y = dataset[idx_current]
    mat, y = torch.unsqueeze(mat.to("cpu"), 0), y.to("cpu").numpy()
    preds = np.zeros((future_obs, n_samples)) # 7x200
    y_vals = []

    #x_vals = [*range(idx_current-future_obs+1, idx_current+1)]
    for p in range(idx_current-future_obs+1, idx_current): # know last one from above, would add padding outside of them
        y_vals.append(dataset[p][1].cpu().numpy())
    y_vals.append(y)

    x_min, x_max = idx_current-future_obs+1, idx_current
    if left:
        x_min = idx_current - future_obs + 1 - padding_val
        for l in range(idx_current-future_obs, idx_current-future_obs-padding_val, -1):
            y_vals.insert(0, dataset[l][1].cpu().numpy())
    if right:
        x_max = idx_current + padding_val
        for r in range(idx_current+1, idx_current+padding_val+1):
            y_vals.append(dataset[r][1].cpu().numpy())
    
    for f in range(future_obs):
        model.load_state_dict(torch.load(f"./weights/weights-{past_units}-{max_delay}-{'week' if weeks else 'day'}-fut{f}-{state}"))
        for i in range(n_samples):
            preds[f, i] = model(mat).sample().numpy()
    preds_mean = np.quantile(preds, 0.5, axis=1)
    
    intervals_dict = {}
    for l in levels:
        intervals_dict[l] = (np.quantile(preds, (1-l)/2, 1), np.quantile(preds, (1+l)/2, 1))
    
    plt.figure(figsize=(12, 7))
    plt.plot([*range(x_min, x_max+1)], y_vals, label="True count", c = "black")
    plt.plot([*range(idx_current-future_obs+1, idx_current+1)], preds_mean, label = "Median nowcasted predictions", c = "crimson", alpha = 0.75)
    for l in levels:
        lower, upper = intervals_dict[l]
        plt.fill_between([*range(idx_current-future_obs+1, idx_current+1)], lower, upper, color = "crimson", alpha = 0.2, label = f"{int(100*l)}% CI")
    plt.grid(alpha=.2)
    plt.axvline(idx_current, color = "black", linestyle = "--", label = f"Current {'day' if not weeks else 'week'}")
    plt.xlabel("Days")
    plt.legend()
    plt.ylabel("Number of cases")
    plt.savefig(f"../outputs/figures/past_correction_{'week' if weeks else 'day'}_{idx_current}_fut{future_obs}")
    plt.show()