import pandas as pd
import matplotlib.pyplot as plt
import math
import os

def plot(input_path, output_path, filename='metrics.jpg'):
    df = pd.read_csv(input_path)
    metrics = [col for col in df.columns if col != 'model']
    models = df['model'].tolist()
    num_metrics = len(metrics)

    cols = 3
    rows = math.ceil(num_metrics / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()  # Make 1D for easy indexing

    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = df[metric].tolist()
        ax.bar(models, values)
        ax.set_title(metric)
        ax.set_ylabel('Value')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
    plt.close()

