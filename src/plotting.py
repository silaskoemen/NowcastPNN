import numpy as np
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
from scipy import stats
import torch

## Make plot over entire dataset for desired confidence level
def plot_entire_confints(dataset, model, n_samples = 200, levels = [0.5, 0.95]):
    model.train()
    model = model.to("cpu")
    plotloader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
    preds = np.zeros((dataset.__len__(), n_samples))
    mat, y = next(iter(plotloader))
    mat, y = mat.to("cpu"), 1000*y.to("cpu").numpy()
    preds = np.zeros((y.shape[0], n_samples))
    #preds = torch.zeros((y.shape[0], n_samples))
    for i in range(n_samples):
        preds[:, i] = model(mat).sample().numpy()
    preds_mean = np.quantile(preds, 0.5, axis=1)#np.mean(preds, axis = 1)
    #preds_mean = torch.mode(preds, dim=1).values#.to_numpy()#np.mean(preds, axis = 1)#stats.mode(preds, axis=0)[0]
    
    intervals_dict = {}
    for l in levels:
        intervals_dict[l] = (np.quantile(preds, (1-l)/2, 1), np.quantile(preds, (1+l)/2, 1))
    
    plt.figure(figsize=(12, 7))
    plt.plot(y, label="True count", c = "black")
    #plt.plot(y_atm, label="reported on day", c = "darkgrey")
    plt.plot(preds_mean, label = "mean nowcasted predictions", c = "crimson", alpha = 0.75)
    for l in levels:
        lower, upper = intervals_dict[l]
        plt.fill_between(range(len(y)), lower, upper, color = "crimson", alpha = 0.2, label = f"{int(100*l)}% CI")
    plt.grid(alpha=.2)
    plt.axvline(300, color = "black", label="division train/test", linestyle="--")
    plt.legend()
    plt.ylabel("Number of cases")
    plt.xlabel("EpiWeeks since start of observation")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.savefig("../outputs/figures/simple_forecast")
    plt.show()