import numpy as np
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from metrics import form_predictions
from scipy import stats
import torch
from sklearn.decomposition import PCA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import torch
import pandas as pd
plt.rcParams['font.family'] = "Times New Roman" #"cmr10"
plt.rcParams.update({"axes.labelsize" : "large"}) # 'font.size': 11, 
#plt.rcParams['font.serif'] = "Computer Modern"

models = ["Epinowcast", "RIVM", "NowcastPNN"]
colors = ['dodgerblue', 'black', 'crimson']

## Make plot over entire dataset for desired confidence level
def plot_entire_confints(dataset, model, n_samples = 200, levels = [0.5, 0.95], weeks = False, xlims = None, random_split = True, test_idcs = None, total = True, dow = False):
    plotloader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
    mat, y = next(iter(plotloader))
    if dow:
        mat, dow_val = mat
        mat, dow_val, y = mat.to("cpu"), dow_val.to("cpu"), y.to("cpu").numpy()
    else:
        mat, y = mat.to("cpu"), y.to("cpu").numpy()
    model.eval() # sets batch norm to eval so a single entry can be passed without issues of calculating mean and std.
    model.drop1.train() # keeps dropout layers active
    model.drop2.train()
    model = model.to("cpu")
    preds = np.zeros((y.shape[0], n_samples))
    for i in range(n_samples):
        if total:
            preds[:, i] = np.squeeze(model(mat).sample().numpy()) if not dow else np.squeeze(model(mat, dow_val).sample().numpy())
        else:
            temp_counts = model(mat).sample().numpy()
            preds[:, i] = form_predictions(temp_counts, y, future_obs=0)    
    preds_median = np.quantile(preds, 0.5, axis=1)
    #print(preds_median[2133:2353])
    intervals_dict = {}
    for l in levels:
        intervals_dict[l] = (np.quantile(preds, (1-l)/2, 1), np.quantile(preds, (1+l)/2, 1))

    if not total:
        y = y.sum(axis = 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(y, label=r"True count", c = "black")
    #plt.plot(y_atm, label="reported on day", c = "darkgrey")
    plt.plot(preds_median, label = r"Nowcast predictions", c = "crimson", alpha = 0.75)
    for l in levels:
        lower, upper = intervals_dict[l]
        plt.fill_between(range(len(y)), lower, upper, color = "crimson", alpha = 0.2, label = f"{int(100*l)}% CI")
    plt.grid(alpha=.2)
    if not random_split:
        if weeks:
            plt.axvline(300, color = "black", label=r"division train/test", linestyle="--")
        else:
            plt.axvline(2100, color = "black", label=r"division train/test", linestyle="--")
    """if test_idcs is not None:
        plt.vlines(test_idcs, ymin=0, ymax=1000, linewidth = 0.1, colors = "red", label = "Test set")"""
    plt.xlabel(fr"{'EpiWeeks' if weeks else 'Days'} since start of observation")
    plt.legend()
    plt.ylabel(r"Number of cases")
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    if xlims is not None:
        plt.xlim(xlims)
        plt.savefig(fr"../outputs/figures/nowcast_{'week' if weeks else 'day'}_subset_{xlims[0]}_{xlims[1]}")            
    elif not random_split:
        plt.xlim(2133, 2844)
        plt.savefig(fr"../outputs/figures/nowcast_{'week' if weeks else 'day'}_recent")
    else: 
        plt.savefig(fr"../outputs/figures/nowcast_{'week' if weeks else 'day'}")
    plt.show()

def plot_coverages(epi_coverages, rivm_coverages, pnn_coverages, levels = [0.5, 0.95]):
    assert len(levels) == 2, "Only two distinct levels supported"
    assert all(ele in [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1] for ele in levels), "Levels must be in levels used: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]"
    # The models and their corresponding coverages
    #models = ["Epinowcast", "RIVM", "NowcastPNN"]
    coverages_lower =  [epi_coverages[min(levels)], rivm_coverages[min(levels)], pnn_coverages[min(levels)]]
    coverages_higher = [epi_coverages[max(levels)], rivm_coverages[max(levels)], pnn_coverages[max(levels)]]

    y_pos = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(7.5, 3.5))  # Reduced height

    colors_50 = ['dodgerblue', 'black', 'crimson']
    colors_95 = ['dodgerblue', 'black', 'crimson']

    ax.barh(y_pos, coverages_lower, color=colors_50, alpha=1.0, height=0.35, label=f'{100*min(levels)}% Coverage', zorder=3)
    ax.barh(y_pos, coverages_higher, color=colors_95, alpha=0.5, height=0.35, label=f'{100*max(levels)}% Coverage', zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize="large")
    ax.set_xlabel('Coverage', fontsize="large")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])

    ## Add minor ticks (for grid) but do not show them (empty labels)
    ax.set_xticks(np.arange(0, 1.01, 0.125), minor=True)
    ax.tick_params(which='minor', length=0)

    ## Add dashed lines at 0.5 and 0.95 with high zorder to appear above grid lines
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5, zorder=4)
    ax.axvline(0.95, color='black', linestyle='--', linewidth=1, zorder=4)

    ## Add grid for both major and minor ticks with lower zorder to be below the bars
    ax.grid(True, alpha=0.5, zorder=1)
    ax.grid(True, which='minor', alpha=0.5, zorder=1)

    ## Move the legend outside the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(f"../outputs/figures/coverages")
    plt.show()


def visualize_embeddings(dim = 8):
    embeddings = torch.load(f"./weights/embedding_weights_{dim}").cpu().detach().numpy()
    pca = PCA(n_components=3)
    vis_dims = pca.fit_transform(embeddings)

    fig = plt.figure(figsize=(11.5, 3.5))  # Adjust figure size for better spacing
    colors = ["crimson", "deepskyblue"]
    n_bins = 7  # Number of color bins in the color map
    cmap_name = 'black_blue'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Different view angles for each cube
    angles = [(20, 20), (20, 45), (20, 60)]

    # Create 3 subplots for the 3 cubes
    for j in range(3):
        ax = fig.add_subplot(1, 3, j+1, projection='3d')
        for i, day in enumerate(weekdays):
            # Assuming vis_dims is a list of 3D coordinates for each weekday
            sub_matrix = np.array(vis_dims[i])
            x = sub_matrix[0]
            y = sub_matrix[1]
            z = sub_matrix[2]
            colors = [cmap(i/len(weekdays))]
            ax.scatter(x, y, zs=z, zdir='z', c=colors, label=day, s=60)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        #ax.set_title(f'Cube {j+1}')  # Title for each cube

        # Set different viewing angles
        ax.view_init(elev=angles[j][0], azim=angles[j][1])

    # Add legend only to the first subplot to avoid repetition
    fig.legend(weekdays, bbox_to_anchor=(0.5, 0.9), loc='upper center', ncol=7)
    plt.savefig("../outputs/figures/visualization_embeddings.svg")
    plt.show()


def plot_is_decomposition(epi_scores, rivm_scores, pnn_scores):
    #models = ["Epinowcast", "RIVM", "NowcastPNN"]
    
    # Colors for each component
    colors_under = ['dodgerblue', 'black', 'crimson']
    colors_spread = ['aliceblue', 'lightgrey', 'mistyrose']  # almost white, a very bright hue of the original color
    colors_over = ['dodgerblue', 'black', 'crimson']

    # Position for each bar on y-axis
    y_pos = np.arange(len(models))

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(7.5, 3.5))

    # Plot each component of the WIS score
    for i, scores in enumerate([epi_scores, rivm_scores, pnn_scores]):
        ax.barh(y_pos[i], scores[0], color=colors_under[i], height=0.35, label='Underprediction' if i == 1 else "", zorder=3)
        ax.barh(y_pos[i], scores[1], left=scores[0], color=colors_over[i], height=0.35, label='Spread' if i == 1 else "", zorder=3, alpha = 0.25) # colors_spread
        ax.barh(y_pos[i], scores[2], left=scores[0] + scores[1], color=colors_over[i], height=0.35, label='Overprediction' if i == 1 else "", zorder=3, alpha = 0.66)

    # Set the y-ticks with the model names
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize="large")

    # Set the x-axis labels
    ax.set_xlabel('IS Decomposition', fontsize="large")

    # Add grid for both major and minor ticks with lower zorder to be below the bars
    ax.grid(True, alpha=0.5, zorder=1)

    # Move the legend outside the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    
    plt.tight_layout()
    plt.savefig(f"../outputs/figures/is_decompositions")
    plt.show()


def plot_wis(epi_scores, rivm_scores, pnn_scores):
    """ Plot vertical bar charts to visualize the WIS scores achieved by all models. """
    scores  = [epi_scores, rivm_scores, pnn_scores]
    # Positions for the bars on the x-axis
    x_pos = np.arange(len(models))

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Plot vertical bars
    ax.bar(x_pos, scores, color=colors, zorder = 2, width=0.66)

    # Set the x-ticks with the model names
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize="large")

    # Add grid lines
    ax.grid(True, axis='y', alpha=0.2, zorder=1)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"../outputs/figures/wis")
    plt.show()

def plot_coverages_pica(epi_dict, rivm_dict, pnn_dict, levels = [0.5, 0.95], save = False, random_split = True, suffix = ""):
    colors_50 = ['dodgerblue', 'black', 'crimson']
    colors_95 = ['dodgerblue', 'black', 'crimson']

    coverages_lower =  [epi_dict["coverages"][min(levels)], rivm_dict["coverages"][min(levels)], pnn_dict["coverages"][min(levels)]]
    coverages_higher = [epi_dict["coverages"][max(levels)], rivm_dict["coverages"][max(levels)], pnn_dict["coverages"][max(levels)]]

    y_pos = np.arange(len(models))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))  # Reduced height

    for i in range(len(models)):
        ax1.barh(y_pos[i], coverages_lower[i], color=colors_50[i], alpha=1.0, height=0.4, label=f'{int(100*min(levels))}% Coverage' if i == 1 else '', zorder=3)
        ax1.barh(y_pos[i], coverages_higher[i], color=colors_95[i], alpha=0.5, height=0.4, label=f'{int(100*max(levels))}% Coverage' if i == 1 else '', zorder=2)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize="x-large")
    ax1.set_xlabel('Coverage', fontsize="x-large")
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1])

    ## Add minor ticks (for grid) but do not show them (empty labels)
    ax1ticks = np.arange(0, 1.01, 0.125)
    ax1ticks = np.delete(ax1ticks, ax1ticks == 0.5)
    ax1.set_xticks(ax1ticks, minor=True)
    ax1.tick_params(which='minor', length=0)

    ## Add dashed lines at 0.5 and 0.95 with high zorder to appear above grid lines
    ax1.axvline(0.5, color='black', linestyle='--', linewidth=1.2, zorder=4)
    ax1.axvline(0.95, color='black', linestyle='--', linewidth=1.2, zorder=4)

    ## Add grid for both major and minor ticks with lower zorder to be below the bars
    ax1.grid(True, alpha=0.2, zorder=1)
    ax1.grid(True, which='minor', alpha=0.2, zorder=1)

    ## Move the legend outside the plot
    for text in ax1.legend().get_texts():
        text.set_color("black")
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)


    pica_scores  = [epi_dict["pica"], rivm_dict["pica"], pnn_dict["pica"]]
        # Positions for the bars on the x-axis
    x_pos = np.arange(len(models))

    # Plot vertical bars
    ax2.bar(x_pos, pica_scores, color=colors, zorder = 2, width=0.6)

    # Set the x-ticks with the model names
    ax2.set_xticks(x_pos)
    ax2.set_ylabel("PICA Score", fontsize="x-large")
    ax2.set_xticklabels(models, fontsize="x-large")

    # Add grid lines
    ax2.grid(True, axis='y', alpha=0.2, zorder=1)
    plt.tight_layout()
    if save:
        plt.savefig(f"../outputs/figures/coverages_pica{'_rec' if not random_split else ''}{suffix}.svg")
    plt.show()

def plot_is_decomp_wis(epi_score_dict, rivm_score_dict, pnn_score_dict, save = False, random_split = True, suffix = ""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 3.5))
    y_pos = np.arange(len(models))
    bar_width = 0.35


    ax1.barh(y_pos[0], epi_score_dict["is"][0], color="dodgerblue", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
    ax1.barh(y_pos[0], epi_score_dict["is"][1], left=epi_score_dict["is"][0], color="aliceblue", height = bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
    ax1.barh(y_pos[0], epi_score_dict["is"][2], left=epi_score_dict["is"][0] + epi_score_dict["is"][1], color="deepskyblue", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
    ax1.barh(y_pos[1], rivm_score_dict["is"][0], color="black", height=bar_width, label='Underpred.', zorder=3, edgecolor = "black", linewidth = 0.4)
    ax1.barh(y_pos[1], rivm_score_dict["is"][1], left=rivm_score_dict["is"][0], color="gainsboro", height=bar_width, label='Spread', zorder=3, edgecolor = "black", linewidth = 0.4)
    ax1.barh(y_pos[1], rivm_score_dict["is"][2], left=rivm_score_dict["is"][0] + rivm_score_dict["is"][1], color="grey", height=bar_width, label='Overpred.', zorder=3, edgecolor = "black", linewidth = 0.4)
    ax1.barh(y_pos[2], pnn_score_dict["is"][0], color="crimson", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
    ax1.barh(y_pos[2], pnn_score_dict["is"][1], left=pnn_score_dict["is"][0], color="mistyrose", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
    ax1.barh(y_pos[2], pnn_score_dict["is"][2], left=pnn_score_dict["is"][0] + pnn_score_dict["is"][1], color="#f5626e", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)


    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize="x-large")
    ax1.set_xlabel('IS Decomposition', fontsize="x-large")
    ax1.grid(True, alpha=0.4, zorder=1)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)

    wis_scores  = [epi_score_dict["wis"], rivm_score_dict["wis"], pnn_score_dict["wis"]]
    # Positions for the bars on the x-axis
    x_pos = np.arange(len(models))

    # Plot vertical bars
    ax2.bar(x_pos, wis_scores, color=colors, zorder = 2, width=0.55)

    # Set the x-ticks with the model names
    ax2.set_xticks(x_pos)
    ax2.set_ylabel("WIS", fontsize="x-large")
    ax2.set_xticklabels(models, fontsize="x-large")

    # Add grid lines
    ax2.grid(True, axis='y', alpha=0.2, zorder=1)
    plt.tight_layout()
    if save:
        plt.savefig(f"../outputs/figures/is_decomp_wis{'_rec' if not random_split else ''}{suffix}.svg")
    plt.show()

def plot_coverages_rand_rec(epi_score_dict, epi_score_dict_rec, rivm_score_dict, rivm_score_dict_rec, pnn_score_dict, pnn_score_dict_rec, levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], save = False):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    colors_covs = ["dodgerblue", "black", "crimson"]
    models = ["Epinowcast", "RIVM", "NowcastPNN"]
    # Extract values for this day
    for i, covs in enumerate([epi_score_dict["coverages"], rivm_score_dict["coverages"], pnn_score_dict["coverages"],]):
        actual_coverages = [covs[level] for level in levels]
        #colors = [cmap(i/14)]
        axes[0].plot(levels, actual_coverages, marker='o', color=colors_covs[i], label=f'{models[i]}')

        # Set title, labels, and x-ticks
    axes[0].set_xlabel('Expected Coverage', fontsize="x-large")
    axes[0].set_ylabel('Actual Coverage', fontsize="x-large")
    axes[0].set_xticks(levels)
    axes[0].set_yticks(levels)
    axes[0].set_title("Random test set", fontsize="x-large")
    axes[0].grid(True)
    axes[0].tick_params(axis='x', labelrotation = 45)
    axes[0].plot(levels, levels, linestyle = '--', color='black', linewidth = 3)  # 45° line

    axes[0].legend(title='Model', bbox_to_anchor=(1.025, 1.), loc='upper left')

    for i, covs in enumerate([epi_score_dict_rec["coverages"], rivm_score_dict_rec["coverages"], pnn_score_dict_rec["coverages"],]):
        actual_coverages = [covs[level] for level in levels]
        #colors = [cmap(i/14)]
        axes[1].plot(levels, actual_coverages, marker='o', color=colors_covs[i], label=f'{models[i]}')

        # Set title, labels, and x-ticks
    axes[1].set_xlabel('Expected Coverage', fontsize="x-large")
    #axes[1].set_ylabel('Actual Coverage', fontsize="x-large")
    axes[1].set_xticks(levels)
    axes[1].set_title("Recent test set", fontsize="x-large")
    axes[1].set_yticks(levels)
    axes[1].grid(True)
    axes[1].tick_params(axis='x', labelrotation = 45)
    axes[1].plot(levels, levels, linestyle = '--', color='black', linewidth = 3)  # 45° line

    #axes[1].legend(title='Model', bbox_to_anchor=(1.025, 1.), loc='upper left')

    plt.tight_layout()
    if save:
        plt.savefig(f"../outputs/figures/coverages_rand_rec.svg")
    plt.show()

def plot_pica(epi_scores, rivm_scores, pnn_scores):
    """ Plot vertical bar charts to visualize the PI Coverage Accuracies scores achieved by all models. """
    scores  = [epi_scores, rivm_scores, pnn_scores]
    # Positions for the bars on the x-axis
    x_pos = np.arange(len(models))

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Plot vertical bars
    ax.bar(x_pos, scores, color=colors, zorder = 2, width = 0.66)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize="large")
    ax.set_ylim((0, 0.1))
    ax.grid(True, axis='y', alpha=0.2, zorder=1)

    plt.tight_layout()
    plt.savefig(f"../outputs/figures/pica")
    plt.show()


def plot_sameday_nowcast_recent(test_loader, test_idcs, levels_pnn, levels_epi, levels_rivm, save = False, xlims = None, suffix = ""):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    if xlims is not None:
        xlims = [datetime.strptime(xlims[0], "%Y-%m-%d"), datetime.strptime(xlims[1], "%Y-%m-%d")]
    _, y = next(iter(test_loader))
    y = y.to("cpu").numpy()

    dates = [days_to_date("2013-01-01", days, past_units=40) for days in test_idcs]
        
    # Create a DataFrame
    date_df = pd.DataFrame({'Date': dates})

    # Plot PNN model results
    axs[0].plot(date_df.Date, y, label='True Count', color='black')
    axs[0].plot(date_df.Date, levels_pnn[0], label='Median of nowcasted predictions', color='crimson')

    # Fill between for the confidence intervals
    for l in [0.5, 0.95]:
        lower, upper = levels_pnn[l]
        axs[0].fill_between(date_df.Date, lower, upper, color='crimson', alpha=-l*2/9+0.4, label=f'{int(100*l)}% CI')

    axs[0].set_title('NowcastPNN', fontsize="x-large")
    axs[0].set_xlabel('Date', fontsize="x-large")
    axs[0].tick_params(axis='x', labelrotation = 25)
    axs[0].set_ylabel('Counts', fontsize="x-large")
    axs[0].legend(loc = "upper left")

    ## Epinowcast
    axs[1].plot(date_df.Date, y, label='True Count', color='black')
    axs[1].plot(date_df.Date, levels_epi[0], label='Predicted Median', color='dodgerblue')

    # Fill between for the confidence intervals
    for l in [0.5, 0.95]:
        lower, upper = levels_epi[l]
        axs[1].fill_between(date_df.Date, lower, upper, color='dodgerblue', alpha=-l*2/9+0.4, label=f'{int(100*l)}% CI')

    axs[1].set_title('Epinowcast', fontsize="x-large")
    axs[1].set_xlabel('Date', fontsize="x-large")
    axs[1].tick_params(axis='x', labelrotation = 25)

    ## RIVM
    axs[2].plot(date_df.Date, y, label='True Count', color='black')
    axs[2].plot(date_df.Date, levels_rivm[0], label='Predicted Median', color='grey')

    # Fill between for the confidence intervals
    for l in [0.5, 0.95]:
        lower, upper = levels_rivm[l]
        axs[2].fill_between(date_df.Date, lower, upper, color='grey', alpha=-l*2/9+0.4, label=f'{int(100*l)}% CI')

    axs[2].set_title('RIVM', fontsize="x-large")
    axs[2].set_xlabel('Date', fontsize="x-large")
    axs[2].tick_params(axis='x', labelrotation = 25)
    if xlims is not None:
        axs[0].set_xlim(xlims)
        axs[1].set_xlim(xlims)
        axs[2].set_xlim(xlims)

    # Adjust layout
    plt.tight_layout()
    if save:
        if xlims is not None:
            plt.savefig(f"../outputs/figures/comparison_sameday_nowcast_{xlims[0].strftime('%Y-%m-%d')}_{xlims[1].strftime('%Y-%m-%d')}{suffix}.svg")
        else:
            plt.savefig(f"../outputs/figures/comparison_sameday_nowcast{suffix}.svg")
    plt.show()


def plot_distance_true_observed(df: pd.DataFrame, idx: str = 100, horizon: int = 30, past_units = 40, future_units = 0, start_date: str = "2013-01-01", weeks = False) -> None:
    """ For specific index (specific date), calculate the difference between true counts versus observed at date
    
    Args:
        :df: [pd.DataFrame]: DataFrame containing the counts at all delay values for all dates
        :idx: [int]: Index for date to plot from, should be bigger than horizon
        :horizon: [int]: Time horizon steps to go back
        :start_date: [str]: Starting date of observations, so labels can be done with true dates
    """
    assert idx > horizon, "Observation index should be larger than horizon to go backwards"
    if isinstance(df, pd.DataFrame): df = np.array(df.values, dtype = np.float32)

    ## Add first of month as ticks, add all reported on day, add prediction with past from model

    df = df[(idx-horizon+1):(idx+future_units+2), :].copy()
    max_delay = df.shape[1]
    y_otd = df[:, 0].copy()
    y_true = df.sum(axis = 1)

    mask = np.zeros((horizon+future_units+1, max_delay), dtype=bool)
    for p in range(max_delay):
        for d in range(max_delay):
            if p + d >= max_delay:
                mask[p+(horizon-max_delay), d] = True
    df[mask] = 0.
    df[(idx+1):, :] = 0.
    df[-1, :] = 0.

    y_obs = df.sum(axis = 1)

    dates = [days_to_date(start_date, days, past_units) for days in range(idx - horizon+1, idx+future_units+2)]
    
    # Create a DataFrame
    date_df = pd.DataFrame({'Date': dates})

    plt.figure(figsize=(8, 4))
    plt.plot(date_df["Date"], y_true, label="True count", color = "black")
    plt.plot(date_df["Date"][:-1], y_obs[:-1], label=f"Observed up to {date_df.iloc[-(future_units+2), 0].strftime('%Y-%m-%d')}", color = "crimson") # convert with start date to day and then plot with months
    plt.plot(date_df["Date"], y_otd, label= "Reported on day", c = "grey")
    """if weeks:
        plt.xlabel("EpiWeeks in the past")
    else:
        plt.xlabel("Days in the past")
    plt.xticks([*range(horizon)], [*range(horizon-1, -1, -1)])"""
    plt.axvline(date_df.iloc[-(future_units+2)], color = "black", linestyle="--", label="Current day")
    plt.ylabel("Number of cases", fontsize = "x-large")
    """date_df['MonthStart'] = date_df['Date'].apply(lambda x: x.replace(day=1))
    month_starts = date_df['MonthStart'].unique()
    
    plt.xticks(month_starts, [date.strftime('%Y-%m-%d') for date in month_starts], rotation=45)"""
    plt.xticks(rotation = 30)
    plt.legend()
    plt.xlim(date_df["Date"].iloc[0], date_df["Date"].iloc[-1])
    plt.ylim(0)
    plt.grid(alpha=0.2)
    plt.savefig("../outputs/figures/nowcasting_task_true_on_day.svg")
    plt.show()

def plot_max_delay_day(df_unlimited_delay):
    unlim_delay_array = np.array(df_unlimited_delay)
    ## Could plot pdf and ecdf for both, next to each other to show
    fractions_reported = np.ndarray((365,))
    cum_reported = np.ndarray((365,))
    for d in range(365):
        fractions_reported[d] = unlim_delay_array[:, d].sum()/unlim_delay_array.sum()
        cum_reported[d] = unlim_delay_array[:, :(d+1)].sum()/unlim_delay_array.sum()

    n_days_99 = next(i for i, value in enumerate(cum_reported) if value >= 0.99)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))

    # Plot for cum_reported
    axs[0].plot(cum_reported, label='Cumulative Reported Cases', color='grey')
    axs[0].axhline(0.99, color='red', linestyle='-.', label='99% threshold')
    axs[0].axvline(n_days_99, color='black', linestyle='--', label=f'Day {n_days_99}')
    axs[0].set_xlabel('Days', fontsize="large")
    axs[0].set_ylabel('Cumulative proportion of reported cases (%)', fontsize="large")
    axs[0].legend()
    axs[0].set_ylim(0, 1.05)
    axs[0].set_xlim(-5,365)
    axs[0].grid(alpha=.2)

    # Plot for fraction_reported
    axs[1].plot(fractions_reported, label='Fraction Reported', color='dodgerblue')
    axs[1].axvline(n_days_99, color='black', linestyle='--', label=f'Day {n_days_99}')
    axs[1].set_xlabel('Days', fontsize="large")
    axs[1].set_ylabel('Proportion of reported cases (%)', fontsize="large")
    axs[1].legend()
    axs[1].set_xlim(-5,365)
    axs[1].grid(alpha=.2)
    plt.tight_layout()
    plt.savefig("../outputs/figures/day_max_delay.svg")
    plt.show()

def plot_past_correction(model, past_units, max_delay, future_obs, weeks, dataset, save = False, random_split = True, padding = "both", dow = False, padding_val = 0, n_samples = 200, levels = [0.5, 0.95], state = "SP", idx = 787, test_idcs = None):
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
    if dow:
        mat, dow_val = mat
        dow_val = dow_val.to("cpu")
    mat, y = torch.unsqueeze(mat.to("cpu"), 0), y.to("cpu").numpy()
    preds = np.zeros((future_obs+1, n_samples)) # 7x200
    y_vals = []

    #x_vals = [*range(idx_current-future_obs+1, idx_current+1)]
    for p in range(idx_current-future_obs, idx_current): # know last one from above, would add padding outside of them
        y_vals.append(dataset[p][1].cpu().numpy())
    y_vals.append(y)

    x_min, x_max = idx_current-future_obs, idx_current
    if left:
        x_min = idx_current - future_obs + 1 - padding_val
        for l in range(idx_current-future_obs, idx_current-future_obs-padding_val, -1):
            y_vals.insert(0, dataset[l][1].cpu().numpy())
    if right:
        x_max = idx_current + padding_val
        for r in range(idx_current+1, idx_current+padding_val+1):
            y_vals.append(dataset[r][1].cpu().numpy())
    

    dates = [days_to_date("2013-01-01", days, past_units) for days in range(x_min, x_max+1)]

    cur_date = days_to_date("2013-01-01", idx, past_units)
    
    # Create a DataFrame
    date_df = pd.DataFrame({'Date': dates})
    
    for f in range(future_obs+1):
        model.load_state_dict(torch.load(f"./weights/weights-{past_units}-{max_delay}-{'week' if weeks else 'day'}-fut{f}{'-rec' if not random_split else ''}{'-dow' if dow else ''}"))
        for i in range(n_samples):
            if dow:
                preds[f, i] = model(mat, dow_val).sample().numpy()
            else:
                preds[f, i] = model(mat).sample().numpy()
    preds = preds[::-1, :]
    preds_mean = np.quantile(preds, 0.5, axis=1)
    
    intervals_dict = {}
    for l in levels:
        intervals_dict[l] = (np.quantile(preds, (1-l)/2, 1), np.quantile(preds, (1+l)/2, 1))
    plt.figure(figsize=(12, 7))
    plt.plot(date_df["Date"], y_vals, label="True count", c = "black") # [*range(x_min, x_max+1)]
    plt.plot(date_df["Date"].iloc[(padding_val):(padding_val+future_obs+1)], preds_mean, label = "Median nowcasted predictions", c = "crimson", alpha = 0.75) # [*range(idx_current-future_obs+1, idx_current+1)]
    for l in levels:
        lower, upper = intervals_dict[l]
        plt.fill_between(date_df["Date"].iloc[(padding_val):(padding_val+future_obs+1)], lower, upper, color = "crimson", alpha = 0.2, label = f"{int(100*l)}% CI")
    plt.grid(alpha=.2)
    plt.axvline(cur_date, color = "black", linestyle = "--", label = f"Current {'day' if not weeks else 'week'}")
    plt.xlabel("Days")
    plt.legend()
    plt.ylabel("Number of cases")
    if save:
        plt.savefig(f"../outputs/figures/past_correction_{'week' if weeks else 'day'}_{idx_current}_fut{future_obs}")
    plt.show()

def past_correction_comparison(model, past_units, max_delay, future_obs, weeks, dataset, rivm_dict, epi_dict, save = False, random_split = True, padding = "both", dow = False, padding_val = 0, n_samples = 200, levels = [0.5, 0.95], idx = 787, number_obs = False):
    model.eval() # sets batch norm to eval so a single entry can be passed without issues of calculating mean and std.
    model.drop1.train() # keeps dropout layers active
    model.drop2.train()
    if number_obs: assert isinstance(dataset, list), "If the number of observations is given, the entire dataset collection needs to be parsed"
    if padding is None or padding == "none": left, right = False, False
    elif padding == "both": left, right = True, True
    elif padding == "left": left, right = True, False
    else: left, right = False, True

    level_keys = {0: 0, 0.5: 4, 0.95: 7}

    model = model.to("cpu")
    idx_current = idx
    if number_obs:
        mat, y = dataset[0][idx_current]
    else:
        mat, y = dataset[idx_current]
    if number_obs:
        y, num_obs = y
        num_obs = num_obs.to("cpu").numpy()
        num_obs_vals = []
    if dow:
        mat, dow_val = mat
        dow_val = dow_val.to("cpu")
    mat, y = torch.unsqueeze(mat.to("cpu"), 0), y.to("cpu").numpy()
    preds = np.zeros((future_obs+1, n_samples))
    y_vals = []

    #x_vals = [*range(idx_current-future_obs+1, idx_current+1)]
    if number_obs:
        for p in range(-future_obs, 0): # know last one from above, would add padding outside of them
            y_vals.append(dataset[np.abs(p)][idx_current][1][0].cpu().numpy())
            num_obs_vals.append(dataset[np.abs(p)][idx_current][1][1].cpu().numpy())
        num_obs_vals.append(num_obs)
    else:
        for p in range(idx_current-future_obs, idx_current): # know last one from above, would add padding outside of them
            y_vals.append(dataset[p][1].cpu().numpy())
    y_vals.append(y)
        
    x_min, x_max = idx_current-future_obs, idx_current
    if left:
        x_min -= padding_val
        for l in range(idx_current-future_obs, idx_current-future_obs-padding_val, -1):
            y_vals.insert(0, dataset[l][1].cpu().numpy())
    if right:
        x_max = idx_current + padding_val
        for r in range(idx_current+1, idx_current+padding_val+1):
            y_vals.append(dataset[r][1].cpu().numpy())

    dates = [days_to_date("2013-01-01", days, past_units) for days in range(x_min, x_max+1)]
    cur_date = days_to_date("2013-01-01", idx, past_units)
    date_df = pd.DataFrame({'Date': dates})
    
    for f in range(future_obs+1):
        model.load_state_dict(torch.load(f"./weights/weights-{past_units}-{max_delay}-{'week' if weeks else 'day'}-fut{f}{'-rec' if not random_split else ''}{'-dow' if dow else ''}"))
        if random_split:
            model.drop1.p, model.drop2.p = 0.3 * (1-f/future_obs), 0.1 * (1-f/future_obs)
        else:
            model.drop1.p, model.drop2.p = 0.15 * (1-f/future_obs), 0.05 * (1-f/future_obs) # 1st +0.05, 2nd +0.1
        for i in range(n_samples):
            if dow:
                preds[f, i] = model(mat, dow_val).sample().numpy()
            else:
                preds[f, i] = model(mat).sample().numpy()
        # Have lower bound in num_obs_vals, set all samples below that to LB
        preds[f, :][preds[f, :] < num_obs_vals[-(f+1)]] = num_obs_vals[-(f+1)]
    preds = preds[::-1, :]
    preds_median = np.quantile(preds, 0.5, axis=1)
    
    intervals_dict = {}
    for l in levels:
        intervals_dict[l] = (np.quantile(preds, (1-l)/2, 1), np.quantile(preds, (1+l)/2, 1))
    
    epi_dict = epi_dict[(cur_date).strftime('%Y-%m-%d')]
    rivm_dict = rivm_dict[(cur_date).strftime('%Y-%m-%d')]

    # Set up the figure and axes for the subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)  # 1 row, 3 columns, shared y-axis
    # Plot for Epinowcast
    ax1.plot(date_df["Date"], y_vals, label="True count", c="black")
    ax1.plot(date_df["Date"].iloc[(padding_val):padding_val+future_obs+2], epi_dict[:, 0, 0][::-1], label="Median nowcasted predictions", c="crimson", alpha=0.75)
    for l in levels:
        lower, upper = epi_dict[:, level_keys[l], 0][::-1], epi_dict[:, level_keys[l], 1][::-1]
        ax1.fill_between(date_df["Date"].iloc[(padding_val):padding_val+future_obs+2], lower, upper, color="crimson", alpha=0.2, label=f"{int(100*l)}% CI")
    ax1.grid(alpha=0.2)
    ax1.tick_params(axis='x', labelrotation = 25)
    ax1.axvline(cur_date, color="black", linestyle="--", label=f"Current {'day' if not weeks else 'week'}")
    ax1.set_title("Epinowcast", fontsize = "x-large")
    ax1.set_ylabel("Number of cases", fontsize="x-large")

    # Plot for NowcastPNN
    ax2.plot(date_df["Date"], y_vals, label="True count", c="black")
    ax2.plot(date_df["Date"], num_obs_vals, label="Already observed count", c="grey")
    ax2.plot(date_df["Date"].iloc[(padding_val):padding_val+future_obs+2], preds_median, label="Median nowcasted predictions", c="crimson", alpha=0.75)
    for l in levels:
        lower, upper = intervals_dict[l]
        ax2.fill_between(date_df["Date"].iloc[(padding_val):padding_val+future_obs+2], lower, upper, color="crimson", alpha=0.2, label=f"{int(100*l)}% CI")
    ax2.grid(alpha=0.2)
    ax2.tick_params(axis='x', labelrotation = 25)
    ax2.axvline(cur_date, color="black", linestyle="--", label=f"Current {'day' if not weeks else 'week'}")
    ax2.set_title("NowcastPNN", fontsize = "x-large")

    # Plot for RIVM
    ax3.plot(date_df["Date"], y_vals, label="True count", c="black")
    ax3.plot(date_df["Date"].iloc[(padding_val):padding_val+future_obs+2], rivm_dict[:, 0, 0][::-1], label="Median nowcasted predictions", c="crimson", alpha=0.75)
    for l in levels:
        lower, upper = rivm_dict[:, level_keys[l], 0][::-1], rivm_dict[:, level_keys[l], 1][::-1]
        ax3.fill_between(date_df["Date"].iloc[(padding_val):padding_val+future_obs+2], lower, upper, color="crimson", alpha=0.2, label=f"{int(100*l)}% CI")
    ax3.grid(alpha=0.2)
    ax3.tick_params(axis='x', labelrotation = 25)
    ax3.axvline(cur_date, color="black", linestyle="--", label=f"Current {'day' if not weeks else 'week'}")
    ax3.set_title("RIVM", fontsize = "x-large")

    # Add a legend only to the first subplot
    ax1.legend()

    # Show the plot
    plt.tight_layout()
    if save:
        plt.savefig(f"../outputs/figures/comparison_past_correction_{idx}.svg")
    plt.show()

def compare_coverages(pnn_dict, epi_dict, rivm_dict):
    """ If just single test run, make all in one plot against 45° line.
    """
    pass

def compare_coverages_future_obs(pnn_coverages, epi_coverages, rivm_coverages, save = False):
    """ 3x1 plot of coverages vs 45° per day of future obs to see how behaves.
    """
    
    models = [epi_coverages, pnn_coverages, rivm_coverages]
    model_names = ['Epinowcast', 'NowcastPNN', 'RIVM']

    # Coverage levels
    levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    # Create 1x3 subplot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Color map
    #colors = cm.plasma(np.linspace(0, 1, 14))  # Adjust for number of days (14)
    #colors = ["crimson", "deepskyblue"]
    #colors = ["deepskyblue", "darkgray", "black"]
    colors = ["gold", "red", "black"]
    n_bins = 14  # Number of color bins in the color map
    cmap_name = 'red_blue'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    for i, (model_coverage, ax) in enumerate(zip(models, axes)):
        for day_idx, day_coverage in enumerate(model_coverage):
            # Extract values for this day
            actual_coverages = [day_coverage[level] for level in levels]
            #colors = [cmap(i/14)]
            ax.plot(levels, actual_coverages, marker='o', color=cmap(day_idx)[:3], label=f'Fut. obs.: {day_idx}')
        
        # Set title, labels, and x-ticks
        ax.set_title(model_names[i], fontsize="x-large")
        ax.set_xlabel('Expected Coverage', fontsize="x-large")
        if i == 0: ax.set_ylabel('Actual Coverage', fontsize="x-large")
        ax.set_xticks(levels)
        ax.set_yticks(levels)
        ax.grid(True)
        ax.tick_params(axis='x', labelrotation = 33)
        ax.plot(levels, levels, linestyle = '--', color='black', linewidth = 3)  # 45° line

        # Add legend only for the first plot
        if i == 2:
            ax.legend(title='Days', bbox_to_anchor=(1.025, 1.05), loc='upper left')

    plt.tight_layout()
    if save:
        plt.savefig("../outputs/figures/coverages_comparison_future_obs_rec.svg")
    plt.show()

def plot_training_size_is_wis_covs(n_training, pnn_is_decomp, pnn_wis, pnn_coverages, save = False, random_split = True):
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.5))

    temp_pnn_is = np.array(pnn_is_decomp)[:, -1]
    temp_pnn_is[0] -= 8000
    # IS Plot
    axes[0].plot(n_training, temp_pnn_is, label='NowcastPNN', marker='o', c = "crimson", markersize=5)
    axes[0].set_xlabel('Number of observations in training set', fontsize="x-large")
    axes[0].set_ylabel('IS', fontsize="x-large")
    axes[0].set_xlim(450, 2200)
    axes[0].hlines(1112.1429236476092, xmin = 450, xmax = 2200, label = "Epinowcast", linestyle = "-.", color = "dodgerblue")
    axes[0].hlines(974.3626620759587, xmin = 450, xmax = 2200, label = "RIVM", linestyle = "--", color = "black")
    axes[0].set_yticks(range(0, 7000, 1000))
    axes[0].set_yticklabels(["0", "1000", "2000", "3000", "...", "13000", "14000"]) 
    axes[0].set_ylim(0)
    axes[0].legend()
    axes[0].text(x = 680, y = 3580, s = "//", color = "crimson", fontsize = 27)
    axes[0].text(x = 636, y = 3005, s = "/", color = "white", fontsize = 132, zorder = 2)
    axes[0].grid(alpha=.2, zorder = 3)

    # WIS Plot
    temp_wis = pnn_wis.copy()
    temp_wis[0] -= 3000
    axes[1].plot(n_training, temp_wis, label='NowcastPNN', marker='o', c = "crimson", markersize=5)
    axes[1].set_xlabel('Number of observations in training set', fontsize="x-large")
    axes[1].set_ylabel('WIS', fontsize="x-large")
    axes[1].set_xlim(450, 2200)
    axes[1].hlines(357.8707966239242, xmin = 450, xmax = 2200, label = "Epinowcast", linestyle = "-.", color = "dodgerblue")
    axes[1].hlines(313.4214658136002, xmin = 450, xmax = 2200, label = "RIVM", linestyle = "--", color = "black")
    axes[1].set_yticks(range(0, 3000, 500))
    axes[1].set_yticklabels(["0", "500", "1000", "...", "5000", "5500"]) 
    axes[1].text(x = 710, y = 1370, s = "//", color = "crimson", fontsize = 27)
    axes[1].text(x = 635, y = 995, s = "/", color = "white", fontsize = 132, zorder = 2)
    axes[1].set_ylim(0)
    #axes[1].legend()
    axes[1].grid(alpha=.2)

    from matplotlib.colors import LinearSegmentedColormap
    colors = ["gold", "red", "black"]
    n_bins = len(n_training)  # Number of color bins in the color map
    cmap_name = 'red_blue'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    for n_idx, n_coverage in enumerate(pnn_coverages):
        # Extract values for this day
        actual_coverages = [n_coverage[level] for level in levels]
        #colors = [cmap(i/14)]
        axes[2].plot(levels, actual_coverages, marker='o', color=cmap(n_idx)[:3], label=f'{n_training[n_idx]}')
        
        # Set title, labels, and x-ticks
        axes[2].set_xlabel('Expected Coverage', fontsize="x-large")
        axes[2].set_ylabel('Actual Coverage', fontsize="x-large")
        axes[2].set_xticks(levels)
        axes[2].set_yticks(levels)
        axes[2].grid(True)
        axes[2].tick_params(axis='x', labelrotation = 45)
        axes[2].plot(levels, levels, linestyle = '--', color='black', linewidth = 3)  # 45° line

    axes[2].legend(title='Size train set', bbox_to_anchor=(1.025, 1.), loc='upper left')

    plt.tight_layout()
    if save:
        plt.savefig(f"../outputs/figures/training_size_metrics_rec.svg")
    plt.show()

def plot_training_size_is_wis_covs_rand(n_training, pnn_is_decomp, pnn_wis, pnn_coverages, save = False, random_split = True):
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.5))

    axes[0].plot(n_training, np.array(pnn_is_decomp)[:, -1], label='NowcastPNN', marker='o', c = "crimson", markersize=5)
    axes[0].set_xlabel('Number of observations in training set', fontsize="x-large")
    axes[0].set_ylabel('IS', fontsize="x-large")
    axes[0].set_xlim(450, 2200)
    axes[0].hlines(949.9955032623753, xmin = 450, xmax = 2200, label = "Epinowcast", linestyle = "-.", color = "dodgerblue")
    axes[0].hlines(781.4161973734148, xmin = 450, xmax = 2200, label = "RIVM", linestyle = "--", color = "black")
    axes[0].set_ylim(0)
    axes[0].legend(loc = "lower left")
    axes[0].grid(alpha=.2, zorder = 3)

    # WIS Plot
    axes[1].plot(n_training, pnn_wis, label='NowcastPNN', marker='o', c = "crimson", markersize=5)
    axes[1].set_xlabel('Number of observations in training set', fontsize="x-large")
    axes[1].set_ylabel('WIS', fontsize="x-large")
    axes[1].set_xlim(450, 2200)
    axes[1].hlines(305.22665527302786, xmin = 450, xmax = 2200, label = "Epinowcast", linestyle = "-.", color = "dodgerblue")
    axes[1].hlines(250.3374279725725, xmin = 450, xmax = 2200, label = "RIVM", linestyle = "--", color = "black")
    axes[1].set_ylim(0)
    #axes[1].legend()
    axes[1].grid(alpha=.2)

    from matplotlib.colors import LinearSegmentedColormap
    colors = ["gold", "red", "black"]
    n_bins = len(n_training)  # Number of color bins in the color map
    cmap_name = 'red_blue'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    for n_idx, n_coverage in enumerate(pnn_coverages):
        # Extract values for this day
        actual_coverages = [n_coverage[level] for level in levels]
        #colors = [cmap(i/14)]
        axes[2].plot(levels, actual_coverages, marker='o', color=cmap(n_idx)[:3], label=f'{n_training[n_idx]}')
        
        # Set title, labels, and x-ticks
        axes[2].set_xlabel('Expected Coverage', fontsize="x-large")
        axes[2].set_ylabel('Actual Coverage', fontsize="x-large")
        axes[2].set_xticks(levels)
        axes[2].set_yticks(levels)
        axes[2].grid(True)
        axes[2].tick_params(axis='x', labelrotation = 45)
        axes[2].plot(levels, levels, linestyle = '--', color='black', linewidth = 3)  # 45° line

    axes[2].legend(title='Size train set', bbox_to_anchor=(1.025, 1.), loc='upper left')

    plt.tight_layout()
    if save:
        plt.savefig(f"../outputs/figures/training_size_metrics.svg")
    plt.show()

def plot_past_units_is_wis_covs(n_past_units, pnn_is_decomp, pnn_wis, pnn_coverages, save = False, random_split = True):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    # IS Plot
    axes[0].plot(n_past_units, np.array(pnn_is_decomp)[:, -1], label='NowcastPNN', marker='o', c = "crimson", markersize=5)
    axes[0].set_xlabel('Past units used for estimation', fontsize="x-large")
    axes[0].set_ylabel('IS', fontsize="x-large")
    axes[0].set_xlim(0, 42)
    if random_split:
        axes[0].hlines(949.9955032623753, xmin = -2, xmax = 42, label = "Epinowcast", linestyle = "-.", color = "dodgerblue")
        axes[0].hlines(781.4161973734148, xmin = -2, xmax = 42, label = "RIVM", linestyle = "--", color = "black")
        axes[0].set_ylim(0, 1000)
    else:
        axes[0].hlines(1112.1429236476092, xmin = -2, xmax = 42, label = "Epinowcast", linestyle = "-.", color = "dodgerblue")
        axes[0].hlines(974.3626620759587, xmin = -2, xmax = 42, label = "RIVM", linestyle = "--", color = "black")
        axes[0].set_ylim(0)
    axes[0].legend()
    axes[0].grid(alpha=.2, zorder = 3)

    # WIS Plot
    axes[1].plot(n_past_units, pnn_wis, label='NowcastPNN', marker='o', c = "crimson", markersize=5)
    axes[1].set_xlabel('Past units used for estimation', fontsize="x-large")
    axes[1].set_ylabel('WIS', fontsize="x-large")
    axes[1].set_xlim(0, 42)
    if random_split:
        axes[1].hlines(305.22665527302786, xmin = -2, xmax = 42, label = "Epinowcast", linestyle = "-.", color = "dodgerblue")
        axes[1].hlines(250.3374279725725, xmin = -2, xmax = 42, label = "RIVM", linestyle = "--", color = "black")
    else:
        axes[1].hlines(357.8707966239242, xmin = -2, xmax = 42, label = "Epinowcast", linestyle = "-.", color = "dodgerblue")
        axes[1].hlines(313.4214658136002, xmin = -2, xmax = 42, label = "RIVM", linestyle = "--", color = "black")
    axes[1].set_ylim(0)
    #axes[1].legend()
    axes[1].grid(alpha=.2)

    from matplotlib.colors import LinearSegmentedColormap
    colors = ["gold", "red", "black"]
    n_bins = len(n_past_units)  # Number of color bins in the color map
    cmap_name = 'red_blue'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    for n_idx, n_coverage in enumerate(pnn_coverages):
        # Extract values for this day
        actual_coverages = [n_coverage[level] for level in levels]
        #colors = [cmap(i/14)]
        axes[2].plot(levels, actual_coverages, marker='o', color=cmap(n_idx)[:3], label=f'{n_past_units[n_idx]}')
        
        # Set title, labels, and x-ticks
        axes[2].set_xlabel('Expected Coverage', fontsize="x-large")
        axes[2].set_ylabel('Actual Coverage', fontsize="x-large")
        axes[2].set_xticks(levels)
        axes[2].set_yticks(levels)
        axes[2].grid(True)
        axes[2].tick_params(axis='x', labelrotation = 45)
        axes[2].plot(levels, levels, linestyle = '--', color='black', linewidth = 3)  # 45° line

    axes[2].legend(title='Past units', bbox_to_anchor=(1.025, 1.), loc='upper left')

    plt.tight_layout()
    if save:
        plt.savefig(f"../outputs/figures/past_units_metrics{'_rec' if not random_split else ''}.svg")
    plt.show()

def plot_is_wis_future_obs(pnn_is_decomp, epi_is_decomp, rivm_is_decomp, pnn_wis, epi_wis, rivm_wis, save = False):
    # Create 2x1 subplot
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))

    # IS Score Plot
    axes[0].plot(np.array(pnn_is_decomp)[:, -1], label='NowcastPNN', marker='o', c = "crimson", markersize=5)
    axes[0].plot(np.array(epi_is_decomp)[:, -1], label='Epinowcast', marker='o', c= "dodgerblue", markersize=5)
    axes[0].plot(np.array(rivm_is_decomp)[:, -1], label='RIVM', marker='o', c="black", markersize=5)
    #axes[0].set_title('IS Score Over Time')
    axes[0].set_xlabel('Future observations (days)', fontsize="x-large")
    axes[0].set_xticks(range(0, 14, 2))
    axes[0].set_ylabel('IS', fontsize="x-large")
    axes[0].legend()
    axes[0].grid(alpha=.2)

    y_pos = np.arange(len(models))
    bar_width = 0.35

    for i, scores in enumerate(zip(np.array(epi_is_decomp)[(0, -1), :], np.array(rivm_is_decomp)[(0, -1), :], np.array(pnn_is_decomp)[(0, -1), :])):
        epi_scores, rivm_scores, pnn_scores = scores
        if i == 0:
            axes[1].barh(y_pos[0] + bar_width / 2, epi_scores[0], color="dodgerblue", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
            axes[1].barh(y_pos[0] + bar_width / 2, epi_scores[1], left=epi_scores[0], color="aliceblue", height = bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
            axes[1].barh(y_pos[0] + bar_width / 2, epi_scores[2], left=epi_scores[0] + epi_scores[1], color="deepskyblue", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
            axes[1].barh(y_pos[1] + bar_width / 2, rivm_scores[0], color="black", height=bar_width, label='Underpred.', zorder=3, edgecolor = "black", linewidth = 0.4)
            axes[1].barh(y_pos[1] + bar_width / 2, rivm_scores[1], left=rivm_scores[0], color="gainsboro", height=bar_width, label='Spread', zorder=3, edgecolor = "black", linewidth = 0.4)
            axes[1].barh(y_pos[1] + bar_width / 2, rivm_scores[2], left=rivm_scores[0] + rivm_scores[1], color="grey", height=bar_width, label='Overpred.', zorder=3, edgecolor = "black", linewidth = 0.4)
            axes[1].barh(y_pos[2] + bar_width / 2, pnn_scores[0], color="crimson", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
            axes[1].barh(y_pos[2] + bar_width / 2, pnn_scores[1], left=pnn_scores[0], color="mistyrose", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
            axes[1].barh(y_pos[2] + bar_width / 2, pnn_scores[2], left=pnn_scores[0] + pnn_scores[1], color="#f5626e", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
        else:
            axes[1].barh(y_pos[0] - bar_width / 2, epi_scores[0], color="dodgerblue", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
            axes[1].barh(y_pos[0] - bar_width / 2, epi_scores[1], left=epi_scores[0], color="aliceblue", height=bar_width, alpha=1, edgecolor = "black", linewidth = 0.4)
            axes[1].barh(y_pos[0] - bar_width / 2, epi_scores[2], left=epi_scores[0] + epi_scores[1], color="deepskyblue", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.5)
            axes[1].barh(y_pos[1] - bar_width / 2, rivm_scores[0], color="black", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
            axes[1].barh(y_pos[1] - bar_width / 2, rivm_scores[1], left=rivm_scores[0], color="gainsboro", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
            axes[1].barh(y_pos[1] - bar_width / 2, rivm_scores[2], left=rivm_scores[0] + rivm_scores[1], color="grey", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
            axes[1].barh(y_pos[2] - bar_width / 2, pnn_scores[0], color="crimson", height=bar_width, zorder=3, edgecolor = "black", linewidth = 0.4)
            axes[1].barh(y_pos[2] - bar_width / 2, pnn_scores[1], left=pnn_scores[0], color="mistyrose", height=bar_width, edgecolor = "black", linewidth = 0.4)
            axes[1].barh(y_pos[2] - bar_width / 2, pnn_scores[2], left=pnn_scores[0] + pnn_scores[1], color="#f5626e", height=bar_width, edgecolor = "black", linewidth = 0.4)
        
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(models, fontsize="x-large")
    axes[1].set_xlabel('IS Decomposition', fontsize="x-large")
    axes[1].grid(True, alpha=0.4, zorder=1)
    axes[1].text(x = 790,y = 2.127, s = "same-day", fontsize = 10, bbox=dict(facecolor='gainsboro', alpha=0.7, boxstyle = "larrow", lw = 0.1))
    axes[1].text(x = 390,y = 1.76, s = "13 fut. obs.", fontsize = 10, bbox=dict(facecolor='gainsboro', alpha=0.7, boxstyle = "larrow", lw = 0.1))
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)

    # WIS Score Plot
    axes[2].plot(pnn_wis, label='NowcastPNN', marker='o', c = "crimson", markersize=5)
    axes[2].plot(epi_wis, label='Epinowcast', marker='o', c = "dodgerblue", markersize=5)
    axes[2].plot(rivm_wis, label='RIVM', marker='o', c = "black", markersize=5)
    #axes[1].set_title('WIS Score Over Time')
    axes[2].set_xlabel('Future observations (days)', fontsize="x-large")
    axes[2].set_xticks(range(0, 14, 2))
    axes[2].set_ylabel('WIS', fontsize="x-large")
    #axes[2].legend()
    axes[2].grid(alpha=.2)

    plt.tight_layout()
    if save:
        plt.savefig("../outputs/figures/is_wis_future_obs.svg")
    plt.show()

def days_to_date(start_date, num_days, past_units = 1):
    """
    Converts number of days since start_date to the corresponding date.
    
    Args:
    start_date (str): The start date in 'YYYY-MM-DD' format.
    num_days (int): Number of days from the start date.
    
    Returns:
    datetime: The corresponding date.
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    return start_date + timedelta(days=int(num_days+past_units-1))