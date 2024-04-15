import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_radar(df, bins, column,data):
    # SET DATA 
    data_counts = pd.crosstab(pd.cut(df[column], bins=bins), df[data])

    # CREATE BACKGROUND
    datas = set(pd.cut(df[column], bins=bins))

    # Angle of each axis in the plot
    angles = [(n / len(datas)) * 2 * np.pi for n in range(len(datas)+1)]

    subplot_kw = {
        'polar': True
    }

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=subplot_kw)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    plt.xticks(angles[:-1], datas)
    plt.yticks(color="grey", size=10)

    # ADD PLOTS
    for outcome in data_counts.columns:
        counts = data_counts[outcome].tolist()
        counts += counts[:1]  # Properly loops the circle back
        ax.plot(angles, counts, linewidth=1, linestyle='solid', label=outcome)
        ax.fill(angles, counts, alpha=0.1)

    plt.title(f"Counts by {column} Bins")
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.show()

# Example usage:
# plot_radar(df, bins=20, columns=["Outcome"])
