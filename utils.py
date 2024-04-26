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


def encode_blood_pressure(bp):
    if bp < 120:
        return 0  # Normal
    elif bp < 130:
        return 1  # Elevated
    elif bp < 140:
        return 2  # Hypertension Stage 1
    else:
        return 3  # Hypertension Stage 2
    
def categorize_glucose(glucose):
    if glucose < 100:
        return 0
    elif glucose < 126:
        return 1
    else:
        return 2

def calculate_feature_importance(model, data):
    # Set the model to evaluation mode
    model.eval()

    # Move model and data to the same device
    device = next(model.parameters()).device
    data = data.to(device)

    # Convert data to PyTorch tensor if it's not already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    # Ensure gradients are enabled for the input tensor
    data.requires_grad = True

    # Forward pass to get the predictions
    outputs = model(data)

    # Initialize gradients tensor
    gradients = torch.zeros_like(data)

    # Backward pass to calculate gradients for each output element
    model.zero_grad()
    outputs.backward(torch.ones_like(outputs))  # Backpropagate with respect to the scalar output

    # Get the gradients of the input with respect to the loss
    gradients = data.grad.abs()

    # Calculate mean gradient across samples
    mean_gradients = gradients.mean(dim=0)

    return mean_gradients.cpu().detach().numpy()  # Detach from computational graph and move to CPU
    