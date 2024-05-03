import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from itertools import product
from sklearn.metrics import classification_report


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
    
def categorise_glucose(glucose):
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

def regression_imputation(data, target_column, predictors):
    
    complete_data = data.dropna(subset=[target_column])
    missing_data = data[data[target_column].isnull()].copy()  # Ensure a copy is made
    
    X = complete_data[predictors]
    y = complete_data[target_column]
    
    model = LinearRegression()
    model.fit(X, y)
    
    missing_data_X = missing_data[predictors]
    
    # Use .loc to avoid SettingWithCopyWarning
    missing_data.loc[:, target_column + '_imputed'] = model.predict(missing_data_X)
    
    imputed_data = pd.concat([complete_data, missing_data], axis=0)
    
    return imputed_data

def combine_imputed_column(imputed_data, target_column):
    imputed_data.loc[:, target_column] = imputed_data.apply(lambda row: row[target_column+'_imputed'] if pd.isnull(row[target_column]) else row[target_column], axis=1)
    imputed_data.drop(columns=[target_column+'_imputed'], inplace=True)
    
    return imputed_data
    
def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, optimizer, loss_fn, n_epochs=100, batch_size=16, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []
    train_f1_scores, val_f1_scores, test_f1_scores = [], [], []
    train_precisions, val_precisions, test_precisions = [], [], []
    train_recalls, val_recalls, test_recalls = [], [], []
    train_cm, val_cm, test_cm = None, None, None

    best_val_acc, early_stop_counter, best_test_acc = 0, 0, 0

    for epoch in range(n_epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            Xbatch = X_train[i:i + batch_size]
            ybatch = y_train[i:i + batch_size]

            # Forward pass
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_losses.append(loss.item())

        # Compute training accuracy
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train)
            train_accuracy = (y_pred_train.round() == y_train).float().mean().item()
            train_accuracies.append(train_accuracy)
            train_report = classification_report(y_train.cpu().numpy(), y_pred_train.round().cpu().numpy(), output_dict=True)
            train_f1_scores.append(train_report['macro avg']['f1-score'])
            train_precisions.append(train_report['macro avg']['precision'])
            train_recalls.append(train_report['macro avg']['recall'])

            # Compute validation loss and accuracy
            y_pred_val = model(X_val)
            val_loss = loss_fn(y_pred_val, y_val)
            val_losses.append(val_loss.item())
            val_accuracy = (y_pred_val.round() == y_val).float().mean().item()
            val_accuracies.append(val_accuracy)
            val_report = classification_report(y_val.cpu().numpy(), y_pred_val.round().cpu().numpy(), output_dict=True)
            val_f1_scores.append(val_report['macro avg']['f1-score'])
            val_precisions.append(val_report['macro avg']['precision'])
            val_recalls.append(val_report['macro avg']['recall'])

            # Compute test loss and accuracy
            y_pred_test = model(X_test)
            test_loss = loss_fn(y_pred_test, y_test)
            test_losses.append(test_loss.item())
            test_accuracy = (y_pred_test.round() == y_test).float().mean().item()
            test_accuracies.append(test_accuracy)
            test_report = classification_report(y_test.cpu().numpy(), y_pred_test.round().cpu().numpy(), output_dict=True)
            test_f1_scores.append(test_report['macro avg']['f1-score'])
            test_precisions.append(test_report['macro avg']['precision'])
            test_recalls.append(test_report['macro avg']['recall'])

            # Update confusion matrices
            if epoch == n_epochs - 1:
                train_cm = confusion_matrix(y_train.cpu().numpy(), y_pred_train.round().cpu().numpy())
                val_cm = confusion_matrix(y_val.cpu().numpy(), y_pred_val.round().cpu().numpy())
                test_cm = confusion_matrix(y_test.cpu().numpy(), y_pred_test.round().cpu().numpy())

            # Check for early stopping
            if best_test_acc < test_accuracy:
                best_test_acc = test_accuracy
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1} as test accuracy has not improved for {patience} epochs.')
                train_cm = confusion_matrix(y_train.cpu().numpy(), y_pred_train.round().cpu().numpy())
                val_cm = confusion_matrix(y_val.cpu().numpy(), y_pred_val.round().cpu().numpy())
                test_cm = confusion_matrix(y_test.cpu().numpy(), y_pred_test.round().cpu().numpy())
                break

        print(f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, '
              f'Train F1 Score: {train_f1_scores[-1]:.4f}, Val F1 Score: {val_f1_scores[-1]:.4f}, Test F1 Score: {test_f1_scores[-1]:.4f}, '
              f'Train Precision: {train_precisions[-1]:.4f}, Val Precision: {val_precisions[-1]:.4f}, Test Precision: {test_precisions[-1]:.4f}, '
              f'Train Recall: {train_recalls[-1]:.4f}, Val Recall: {val_recalls[-1]:.4f}, Test Recall: {test_recalls[-1]:.4f}')

        if early_stop_counter >= patience:
            break
    

    train={"Loss": train_losses, "Accuracy": train_accuracies, "F1 Score": train_f1_scores, "Precision": train_precisions, "Recall": train_recalls, "cm": train_cm}
    validation={"Loss": val_losses, "Accuracy": val_accuracies, "F1 Score": val_f1_scores, "Precision": val_precisions, "Recall": val_recalls, "cm": val_cm}
    test={"Loss": test_losses, "Accuracy": test_accuracies, "F1 Score": test_f1_scores, "Precision": test_precisions, "Recall": test_recalls, "cm": test_cm}

    return train, validation, test


def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="rocket", cbar=False)
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

def create_model(n1, n2,num_features):
    model = nn.Sequential(
        nn.Linear(num_features, n1),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(n1, n2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(n2, 1),
        nn.Sigmoid()
    )
    return model

def grid_search(X_train, y_train, X_val, y_val, X_test, y_test, grid_params, n_epochs=100, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    best_params = None
    best_val_loss = float('inf')
    
    for params in product(*grid_params.values()):
        param = dict(zip(grid_params.keys(), params))
        model = create_model(param["n1"], param['n2'], 18)
        model.to(device)
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=param['lr'])
        batch_size = param["batch_size"]
        
        train_losses, val_losses = [], []
        early_stop_counter = 0

        for epoch in range(n_epochs):
            model.train()
            for i in range(0, len(X_train), batch_size):
                Xbatch = X_train[i:i + batch_size]
                ybatch = y_train[i:i + batch_size]

                y_pred = model(Xbatch)
                loss = loss_fn(y_pred, ybatch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            train_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                y_pred_val = model(X_val)
                val_loss = loss_fn(y_pred_val, y_val)
                val_losses.append(val_loss.item())
                y_pred_val = model(X_val)
                val_accuracy = (y_pred_val.round() == y_val).float().mean().item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch + 1} for parameter combination: {param}')
                break

        print(f'\nParameters: {param}, Best Validation Loss: {best_val_loss:.4f}, Val_acc: {val_accuracy} at epoch {best_epoch + 1}')

        if best_params is None or best_val_loss < best_params['best_val_loss']:
            best_params = {'param': param, 'best_val_loss': best_val_loss}
    return best_params