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
def plot_entire_confints(dataset, model, n_samples = 200, levels = [0.5, 0.95], weeks = True, xlims = None, random_split = True, test_idcs = None, total = True, dow = False):
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


def visualize_embeddings(path = "./weights/embedding_weights"):
    embeddings = torch.load("./weights/embedding_weights").cpu().numpy()
    pca = PCA(n_components=3)
    vis_dims = pca.fit_transform(embeddings)

    fig = plt.figure(figsize=(11.5, 3.5))  # Adjust figure size for better spacing
    colors = ["crimson", "deepskyblue"]
    n_bins = 7  # Number of color bins in the color map
    cmap_name = 'black_blue'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Different view angles for each cube
    angles = [(25, 20), (25, 45), (25, 60)]

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

    df = df[(idx-horizon+1):(idx+future_units+1), :].copy()
    max_delay = df.shape[1]
    y_otd = df[:, 0]
    y_true = df.sum(axis = 1)

    mask = np.zeros((horizon+future_units, max_delay), dtype=bool)
    for p in range(max_delay):
        for d in range(max_delay):
            if p + d >= max_delay:
                mask[p+(horizon-max_delay), d] = True
    df[mask] = 0.
    df[(idx+1):-1, :] = 0.

    y_obs = df.sum(axis = 1)

    dates = [days_to_date(start_date, days, past_units) for days in range(idx - horizon+1, idx+future_units+1)]
    
    # Create a DataFrame
    date_df = pd.DataFrame({'Date': dates})

    plt.figure(figsize=(8, 4))
    plt.plot(date_df["Date"], y_true, label="True count", color = "black")
    plt.plot(date_df["Date"], y_obs, label=f"Observed up to {date_df.iloc[-(future_units+1), 0].strftime('%Y-%m-%d')}", color = "crimson") # convert with start date to day and then plot with months
    plt.plot(date_df["Date"], y_otd, label= "Reported on day", c = "grey")
    """if weeks:
        plt.xlabel("EpiWeeks in the past")
    else:
        plt.xlabel("Days in the past")
    plt.xticks([*range(horizon)], [*range(horizon-1, -1, -1)])"""
    plt.axvline(date_df.iloc[-(future_units+1)], color = "black", linestyle="--", label="Current day")
    plt.ylabel("Number of cases")
    """date_df['MonthStart'] = date_df['Date'].apply(lambda x: x.replace(day=1))
    month_starts = date_df['MonthStart'].unique()
    
    plt.xticks(month_starts, [date.strftime('%Y-%m-%d') for date in month_starts], rotation=45)"""
    plt.xticks(rotation = 30)
    plt.legend()
    plt.xlim(date_df["Date"].iloc[0], date_df["Date"].iloc[-1])
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
    axs[0].set_xlabel('Days')
    axs[0].set_ylabel('Cumulative Reported Cases')
    axs[0].legend()
    axs[0].set_ylim(0, 1.05)
    axs[0].set_xlim(-5,365)
    axs[0].grid(alpha=.2)

    # Plot for fraction_reported
    axs[1].plot(fractions_reported, label='Fraction Reported', color='dodgerblue')
    axs[1].axvline(n_days_99, color='black', linestyle='--', label=f'Day {n_days_99}')
    axs[1].set_xlabel('Days')
    axs[1].set_ylabel('Fraction Reported')
    axs[1].legend()
    axs[1].set_xlim(-5,365)
    axs[1].grid(alpha=.2)
    plt.tight_layout()
    plt.savefig("../outputs/figures/day_max_delay.svg")
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
    return start_date + timedelta(days=num_days+past_units-1)