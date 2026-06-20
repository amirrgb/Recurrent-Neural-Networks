import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RESULT_DIR = BASE_DIR / "results"

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_figure(fig, relative_path):
    """Save figure to RESULT_DIR/figures/relative_path.png"""
    full_path = RESULT_DIR / "figures" / f"{relative_path}.png"
    ensure_dir(full_path.parent)
    fig.savefig(full_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

def save_table(df, filename):
    """Save DataFrame to RESULT_DIR/tables/filename.csv"""
    ensure_dir(RESULT_DIR / "tables")
    path = RESULT_DIR / "tables" / f"{filename}.csv"
    df.to_csv(path, index=False)
    return df

def plot_time_series(y, start_idx=0, end_idx=1000, title="Target Variable (y)", 
                     xlabel="Sample Index", ylabel="y (target)", 
                     save_name=None, figsize=(12, 5)):
    """Plot time series data."""
    fig = plt.figure(figsize=figsize)
    y_range = y[start_idx:end_idx]
    plt.plot(range(start_idx, end_idx), y_range, linewidth=1.5, color='darkblue')
    plt.title(f'{title} from {start_idx} to {end_idx-1}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if save_name:
        save_figure(fig, save_name)
    
    return fig

def plot_all_features(data, start_idx=0, end_idx=1000, save_name=None):
    """
    Plot all features (x1-x5 and y) as separate individual plots.
    Each feature is saved as a separate image: save_name_{feature}.png
    """
    features = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
    
    for feat in features:
        fig = plt.figure(figsize=(10, 5))
        values = data[feat][start_idx:end_idx]
        plt.plot(range(start_idx, end_idx), values, linewidth=1.5, color='darkblue')
        plt.title(f'{feat} over Time')
        plt.xlabel('Sample Index')
        plt.ylabel(feat)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        if save_name:
            # Save with format: save_name_x1.png, save_name_x2.png, etc.
            feature_save_name = f"{save_name}_{feat}"
            save_figure(fig, feature_save_name)
        
        plt.close(fig)  # Close figure to free memory
    
    return None  # Since we're not returning a single figure anymore

def plot_correlation_heatmap(data, save_name=None):
    """Plot correlation heatmap of all features."""
    features = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
    df = pd.DataFrame()
    for f in features:
        df[f] = data[f]
    
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix of Features')
    
    plt.tight_layout()
    plt.show()
    if save_name:
        save_figure(fig, save_name)
    
    return fig, corr 

def plot_training_history(histories, labels, title="Training History", 
                          ylabel="Loss (MSE)", xlabel="Epoch",
                          save_name=None, figsize=(10, 6)):
    """Plot training loss curves for multiple models."""
    fig = plt.figure(figsize=figsize)
    for history, label in zip(histories, labels):
        plt.plot(history, linewidth=1.5, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    if save_name:
        save_figure(fig, save_name)
    
    return fig

def plot_mse_vs_parameter(values, mse_list, param_counts, xlabel, 
                          title, save_name=None, figsize=(10, 6)):
    """Plot MSE vs some parameter (hidden size, seq length, etc.)."""
    fig = plt.figure(figsize=figsize)
    plt.plot(values, mse_list, 'bo-', linewidth=2, markersize=8)
    plt.xlabel(xlabel)
    plt.ylabel('MSE (Test)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Annotate with parameter counts
    for i, (v, m, p) in enumerate(zip(values, mse_list, param_counts)):
        plt.annotate(f'{p:,}', (v, m), textcoords="offset points", 
                     xytext=(5, 5), ha='center', fontsize=8)
    plt.show()
    if save_name:
        save_figure(fig, save_name)
    
    return fig

def plot_comparison_mse(values, mse_dict, title, xlabel, 
                        save_name=None, figsize=(10, 6)):
    """Plot MSE comparison for multiple models (e.g., RNN vs LSTM)."""
    fig = plt.figure(figsize=figsize)
    for model_name, mse_list in mse_dict.items():
        plt.plot(values, mse_list, 'o-', linewidth=2, markersize=8, label=model_name)
    plt.xlabel(xlabel)
    plt.ylabel('MSE (Test)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()  
    if save_name:
        save_figure(fig, save_name)

    
    return fig

def plot_hidden_cell_states(hidden_states, cell_states, save_name=None, figsize=(14, 6)):
    """Plot hidden states and cell states over time."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Hidden states
    axes[0].plot(hidden_states, linewidth=1.5, color='blue')
    axes[0].set_title('Hidden States (h_t) over Time')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Hidden State Value')
    axes[0].grid(True, alpha=0.3)
    
    # Cell states
    axes[1].plot(cell_states, linewidth=1.5, color='red')
    axes[1].set_title('Cell States (c_t) over Time')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Cell State Value')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    if save_name:
        save_figure(fig, save_name)
    
    return fig

def plot_gradient_norms(grad_norms_first, grad_norms_last, title, 
                        save_name=None, figsize=(10, 6)):
    """Plot gradient norms of first and last layers over epochs."""
    fig = plt.figure(figsize=figsize)
    epochs = range(1, len(grad_norms_first) + 1)
    plt.plot(epochs, grad_norms_first, 'b-o', label='First RNN Layer', linewidth=1.5, markersize=4)
    plt.plot(epochs, grad_norms_last, 'r-s', label='Last RNN Layer', linewidth=1.5, markersize=4)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm (log scale)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    if save_name:
        save_figure(fig, save_name)
    
    return fig

def plot_forecast_horizon(horizons, mse_results, model_names, 
                          title="Prediction Error vs Forecast Horizon",
                          save_name=None, figsize=(10, 6)):
    """Plot MSE vs forecast horizon for multiple models."""
    fig = plt.figure(figsize=figsize)
    for i, model_name in enumerate(model_names):
        plt.plot(horizons, mse_results[i], 'o-', linewidth=2, markersize=8, label=model_name)
    plt.xlabel('Forecast Horizon (steps ahead)')
    plt.ylabel('MSE')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    if save_name:
        save_figure(fig, save_name)

    return fig

def plot_rnn_results(hidden_sizes, mse_list, param_counts, time_list, 
                     save_name_prefix="rnn_experiment"):
    """
    Plot RNN experiment results: MSE vs hidden size and Parameters vs hidden size.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE vs Hidden Size
    axes[0].plot(hidden_sizes, mse_list, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Hidden Size')
    axes[0].set_ylabel('MSE (Test)')
    axes[0].set_title('Effect of Hidden Size on MSE')
    axes[0].grid(True, alpha=0.3)
    for i, (h, m) in enumerate(zip(hidden_sizes, mse_list)):
        axes[0].annotate(f'{m:.6f}', (h, m), textcoords="offset points", 
                         xytext=(5, 5), ha='center', fontsize=8)
    
    # Parameters vs Hidden Size
    axes[1].plot(hidden_sizes, param_counts, 'rs-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Hidden Size')
    axes[1].set_ylabel('Number of Parameters')
    axes[1].set_title('Model Complexity vs Hidden Size')
    axes[1].grid(True, alpha=0.3)
    for i, (h, p) in enumerate(zip(hidden_sizes, param_counts)):
        axes[1].annotate(f'{p:,}', (h, p), textcoords="offset points", 
                         xytext=(5, 5), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    save_figure(fig, f"{save_name_prefix}_mse_params")
    
    # Also plot training time separately
    fig2 = plt.figure(figsize=(8, 5))
    plt.plot(hidden_sizes, time_list, 'g^--', linewidth=2, markersize=8)
    plt.xlabel('Hidden Size')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs Hidden Size')
    plt.grid(True, alpha=0.3)
    for i, (h, t) in enumerate(zip(hidden_sizes, time_list)):
        plt.annotate(f'{t:.2f}s', (h, t), textcoords="offset points", 
                     xytext=(5, 5), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    save_figure(fig2, f"{save_name_prefix}_time")


def plot_seq_length_results(seq_lengths, mse_list, save_name_prefix="rnn_seq_length"):
    """Plot MSE vs sequence length."""
    fig = plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, mse_list, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Sequence Length (N)')
    plt.ylabel('MSE (Test)')
    plt.title('Effect of Sequence Length on Prediction Error')
    plt.grid(True, alpha=0.3)
    for i, (n, m) in enumerate(zip(seq_lengths, mse_list)):
        plt.annotate(f'{m:.6f}', (n, m), textcoords="offset points", 
                     xytext=(5, 5), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    save_figure(fig, f"{save_name_prefix}_mse")


def plot_num_layers_results(num_layers_list, mse_list, time_list, 
                            save_name_prefix="rnn_num_layers"):
    """Plot MSE and training time vs number of layers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE vs Number of Layers
    axes[0].plot(num_layers_list, mse_list, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of RNN Layers')
    axes[0].set_ylabel('MSE (Test)')
    axes[0].set_title('Effect of Depth on Prediction Error')
    axes[0].grid(True, alpha=0.3)
    for i, (n, m) in enumerate(zip(num_layers_list, mse_list)):
        axes[0].annotate(f'{m:.6f}', (n, m), textcoords="offset points", 
                         xytext=(5, 5), ha='center', fontsize=8)
    
    # Training Time vs Number of Layers
    axes[1].plot(num_layers_list, time_list, 'rs-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of RNN Layers')
    axes[1].set_ylabel('Training Time (seconds)')
    axes[1].set_title('Training Time vs Model Depth')
    axes[1].grid(True, alpha=0.3)
    for i, (n, t) in enumerate(zip(num_layers_list, time_list)):
        axes[1].annotate(f'{t:.2f}s', (n, t), textcoords="offset points", 
                         xytext=(5, 5), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    save_figure(fig, f"{save_name_prefix}_mse_time")


def create_results_table(results_list, columns, filename):
    """Create and save a results table."""
    df = pd.DataFrame(results_list, columns=columns)
    print("\n" + "="*80)
    print(df.to_string(index=False))
    print("="*80)
    save_table(df, filename)
    return df

def plot_training_stability(all_train_losses, all_test_losses, config_labels, 
                            title_prefix="RNN", save_name=None):
    """
    Plot training stability metrics for multiple model configurations.
    
    Args:
        all_train_losses: list of train loss histories
        all_test_losses: list of test loss histories  
        config_labels: list of labels for each configuration (e.g., ['1 layer', '2 layers'])
        title_prefix: prefix for plot titles
        save_name: filename to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Test Loss over epochs (log scale)
    ax = axes[0, 0]
    for i, (test_losses, label) in enumerate(zip(all_test_losses, config_labels)):
        ax.plot(range(1, len(test_losses)+1), test_losses, linewidth=1.5, label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss (MSE)')
    ax.set_title(f'{title_prefix} - Test Loss Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Train Loss over epochs (log scale)
    ax = axes[0, 1]
    for i, (train_losses, label) in enumerate(zip(all_train_losses, config_labels)):
        ax.plot(range(1, len(train_losses)+1), train_losses, linewidth=1.5, label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss (MSE)')
    ax.set_title(f'{title_prefix} - Train Loss Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Loss volatility (standard deviation of last 20 epochs)
    ax = axes[0, 2]
    volatilities = [np.std(losses[-20:]) if len(losses) >= 20 else np.std(losses) 
                    for losses in all_test_losses]
    bars = ax.bar(range(len(config_labels)), volatilities, color=['green', 'yellow', 'orange', 'red'][:len(config_labels)])
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Loss Std Dev (last 20 epochs)')
    ax.set_title(f'{title_prefix} - Training Stability (lower is better)')
    ax.set_xticks(range(len(config_labels)))
    ax.set_xticklabels(config_labels)
    for bar, val in zip(bars, volatilities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Train-Test Gap (overfitting indicator)
    ax = axes[1, 0]
    gaps = [train[-1] - test[-1] for train, test in zip(all_train_losses, all_test_losses)]
    bars = ax.bar(range(len(config_labels)), gaps, color=['blue', 'cyan', 'lightblue', 'gray'][:len(config_labels)])
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Train - Test Loss Gap')
    ax.set_title(f'{title_prefix} - Overfitting Indicator (smaller is better)')
    ax.set_xticks(range(len(config_labels)))
    ax.set_xticklabels(config_labels)
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{gap:.4f}', ha='center', va='bottom', fontsize=9)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 5: Final vs Best MSE
    ax = axes[1, 1]
    best_mses = [min(losses) for losses in all_test_losses]
    final_mses = [losses[-1] for losses in all_test_losses]
    x = np.arange(len(config_labels))
    width = 0.35
    ax.bar(x - width/2, best_mses, width, label='Best MSE', color='green', alpha=0.7)
    ax.bar(x + width/2, final_mses, width, label='Final MSE', color='orange', alpha=0.7)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('MSE')
    ax.set_title(f'{title_prefix} - Best vs Final MSE')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.legend()
    
    # Plot 6: Loss Change between epochs (stability over time)
    ax = axes[1, 2]
    for i, (test_losses, label) in enumerate(zip(all_test_losses, config_labels)):
        changes = [abs(test_losses[j] - test_losses[j-1]) for j in range(1, len(test_losses))]
        ax.plot(range(2, len(test_losses)+1), changes, linewidth=1, label=label, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Epoch-to-Epoch Loss Change')
    ax.set_title(f'{title_prefix} - Loss Change Over Time (stability)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_name:
        save_figure(fig, save_name)
    else:
        plt.show()
    
    return fig


def get_cross_correlation(data, feature='x3', target='y', max_lag=50):
    """
    Compute cross-correlation between a feature and target for different lags.
    Useful for verifying the 10 and 20 step dependencies mentioned in the problem.
    """
    y = data[target]
    x = data[feature]
    
    correlations = []
    for lag in range(max_lag + 1):
        if lag < len(x):
            x_lagged = x[lag:]
            y_aligned = y[:len(x_lagged)]
            if len(x_lagged) > 0:
                corr = np.corrcoef(x_lagged, y_aligned)[0, 1]
                correlations.append(corr)
            else:
                correlations.append(np.nan)
        else:
            correlations.append(np.nan)
    
    return correlations

def plot_gru_gates(update_gates, reset_gates, save_name=None, figsize=(12, 5)):
    """Plot GRU update and reset gates over time."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].plot(update_gates, linewidth=1.5, color='green')
    axes[0].set_title('Update Gates (z_t) over Time')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Gate Value')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(reset_gates, linewidth=1.5, color='orange')
    axes[1].set_title('Reset Gates (r_t) over Time')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Gate Value')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    if save_name:
        save_figure(fig, save_name)
    return fig

def plot_models_comparison_bar(models, mse_list, mae_list, time_list, params_list, save_name=None):
    """Bar chart comparison of three models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MSE
    axes[0].bar(models, mse_list, color=['red', 'blue', 'green'])
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Mean Squared Error')
    for i, v in enumerate(mse_list):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    # MAE
    axes[1].bar(models, mae_list, color=['red', 'blue', 'green'])
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    for i, v in enumerate(mae_list):
        axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    # Training Time
    axes[2].bar(models, time_list, color=['red', 'blue', 'green'])
    axes[2].set_ylabel('Time (seconds)')
    axes[2].set_title('Training Time (50 epochs)')
    for i, v in enumerate(time_list):
        axes[2].text(i, v + 1, f'{v:.1f}s', ha='center')
    
    plt.tight_layout()
    if save_name:
        save_figure(fig, save_name)
    else:
        plt.show()
    return fig