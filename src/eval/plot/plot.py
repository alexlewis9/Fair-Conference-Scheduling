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
        arrow = '↑' if 'silhouette' in metric.lower() else '↓'
        ax.set_title(f"{metric} {arrow}")
        ax.set_ylabel('Value')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')

        # If the metric starts with "fjr" or "core", start y-axis at 1
        if metric.startswith("fjr") or metric.startswith("core"):
            min_val = min(values)
            if min_val >= 1:
                ax.set_ylim(bottom=0.95)  # ensures bars with height 1 are visible
            else:
                ax.set_ylim(bottom=min_val * 0.95 if min_val > 0 else 0)
            ax.axhline(1, color='orange', linestyle='dotted', linewidth=1)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
    plt.close()

