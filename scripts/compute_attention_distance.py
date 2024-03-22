import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


def load_attention_maps(ep_id):
    """Load attention maps from a pickle file."""
    filename = f"./visualisations/debug_attention/attention_maps_{ep_id}.pkl"
    with open(filename, 'rb') as f:
        attention_maps = pickle.load(f)
    return attention_maps


def compute_total_mean_squared_distance(attention_maps):
    """Compute the total mean squared distance between successive attention maps."""
    distances = []
    for i in range(len(attention_maps) - 1):
        map_current = attention_maps[i]
        map_next = attention_maps[i + 1]

        if map_current.shape[-2:] == map_next.shape[-2:]:
            diff = map_next - map_current
            total_distance = np.linalg.norm(diff)
            distances.append(total_distance)
    return distances


def plot_distances(distances):
    """Plot the mean squared distances as a line graph with improvements for clarity."""
    plt.figure(figsize=(10, 6))  # Larger figure size for clarity
    plt.plot(distances, marker='o', markersize=4, alpha=0.5, linestyle='-',
             linewidth=1)  # Reduced marker size and added transparency
    plt.title("Mean Squared Distances Between Successive Attention Maps")
    plt.xlabel("Step")
    plt.ylabel("Mean Squared Distance")
    plt.grid(True)
    # Set the y-axis limit to exclude outliers based on a percentile
    y_max = np.percentile(distances, 99) if distances else 0
    plt.ylim(top=y_max)  # Adjust the maximum y value to the 95th percentile to exclude outliers
    plt.tight_layout()  # Adjusts subplot params so the subplot(s) fits in to the figure area.
    plt.show()


def plot_gaussian_distributions_over_time(distances):
    """Plot the evolving means and standard deviations of two Gaussian distributions fitted at each step, along with the original distances, and Cohen's d."""
    # Initialize lists to store the means, standard deviations, and Cohen's d over time
    means_over_time = [[], []]
    stds_over_time = [[], []]
    noise_value = []

    # Set up the figure and the two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot the original distances as a scatter plot on the first subplot
    steps = np.arange(1, len(distances) + 1)
    gaus_steps = []
    ax1.scatter(steps, distances, color='black', s=10, zorder=3, label='Data Points')

    # Compute GMM parameters for each step
    for step in range(2, len(distances) + 1):
        # Fit the GMM to the data up to the current step
        current_data = np.array(distances[:step]).reshape(-1, 1)
        if gaus_steps:
            gmm1 = GaussianMixture(n_components=2, random_state=0, reg_covar=1e-6,
                                  means_init=gmm1.means_)
            gmm2 = GaussianMixture(n_components=1,means_init=gmm2.means_)
        else:
            gmm1 = GaussianMixture(n_components=2, random_state=0, reg_covar=1e-6)
            gmm2 = GaussianMixture(n_components=1)
        gmm1.fit(current_data)
        gmm2.fit(current_data)

        # Sort the components by mean for consistent coloring and plotting
        ordered_indices = np.argsort(gmm1.means_.flatten())
        means = gmm1.means_.flatten()[ordered_indices]
        stds = np.sqrt(gmm1.covariances_.flatten()[ordered_indices])

        # Store the means and std devs
        for i in range(2):
            means_over_time[i].append(means[i])
            stds_over_time[i].append(stds[i])

        # Calculate noise
        # d = 0.0 if gmm2.bic(current_data) < gmm1.bic(current_data)*1.01 or step < 60 else \
        #     min(1.0, abs(current_data[-1, 0] - means[1]) / min(stds))

        d = min(1.0, max(0, (current_data[-1, 0] - 3.5)/(2-3.5)))

        noise_value.append(d)
        gaus_steps.append(step)

    # Plot lines and continuous error bars for each Gaussian component on the first subplot
    for i, color in zip(range(2), ['blue', 'red']):
        ax1.plot(gaus_steps, means_over_time[i], color=color, label=f'Gaussian {i + 1} Mean')
        ax1.fill_between(gaus_steps,
                         np.array(means_over_time[i]) - np.array(stds_over_time[i]),
                         np.array(means_over_time[i]) + np.array(stds_over_time[i]),
                         color=color, alpha=0.1, label=f'Gaussian {i + 1} Std Dev')

    # Plot Cohen's d on the second subplot
    ax2.plot(gaus_steps[10:], noise_value[10:], color='green', label="Noise")
    ax2.axhline(0, color='grey', lw=0.5)
    ax2.set_ylabel("Noise")
    ax2.grid(True)
    ax2.legend()

    # Customize the main plot (first subplot)
    ax1.set_title("Gaussian Distributions Over Steps")
    ax1.set_ylabel("Mean Squared Distance")
    ax1.legend()
    ax1.grid(True)

    # Set the xlabel for the bottom subplot, which is shared with the top subplot
    ax2.set_xlabel("Step")
    ax2.set_title("Noise value Over Steps")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <episode_id>")
        sys.exit(1)

    episode_id = sys.argv[1]
    attention_maps = load_attention_maps(episode_id)
    distances = compute_total_mean_squared_distance(attention_maps)
    plot_gaussian_distributions_over_time(distances)
