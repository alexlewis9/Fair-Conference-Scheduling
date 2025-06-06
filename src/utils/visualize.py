import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import plotly.express as px
import os

def get_tsne(emb, n_components=2, perplexity=30, random_state=42):
    return TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state).fit_transform(emb)

def get_pca(emb, n_components=2):
    return PCA(n_components=n_components).fit_transform(emb)

def get_umap(emb, n_components=2):
    return umap.UMAP(n_components=n_components).fit_transform(emb)

def rgb_to_rgba(rgb_string, alpha=0.2):
    return rgb_string.replace('rgb(', 'rgba(').replace(')', f',{alpha})')

def visualize_clustering(df, method_names, papers_html = '', output_path='', viz_method='tsne', n_components=2, perplexity=30, random_state=42):
    X = np.vstack(df['emb'].values)
    if viz_method == 'tsne':
        components = get_tsne(X, n_components=n_components, perplexity=perplexity, random_state=random_state)
    elif viz_method == 'pca':
        components = get_pca(X, n_components=n_components)
    elif viz_method == 'umap':
        components = get_umap(X, n_components=n_components)
    else:
        raise ValueError(f"Invalid method: {viz_method}")

    for i in range(n_components):
        df[f'comp_{i}'] = components[:, i]

    color_map = px.colors.qualitative.Set3
    for method in method_names:
        fig = go.Figure()
        unique_clusters = df[method].dropna().unique()

        for i, cluster in enumerate(unique_clusters):
            cluster_df = df[df[method] == cluster]
            cluster_points = cluster_df[[f'comp_{i}' for i in range(n_components)]].values
            color = color_map[i % len(color_map)]
            rgba_fill = rgb_to_rgba(color, 0.2)

            customdata = cluster_df['id'].apply(lambda id_:  f"../../papers/{id_}.html").values

            if n_components == 2:
                fig.add_trace(go.Scatter(
                    x=cluster_df['comp_0'],
                    y=cluster_df['comp_1'],
                    mode='markers',
                    name=cluster,
                    legendgroup=cluster,
                    marker=dict(color=color, size=6),
                    customdata=customdata,
                    hovertext=[
                        f"<b>{title}</b><br>{authors}"
                        for title, authors in zip(cluster_df['title'], cluster_df['authors'])
                    ],
                    hoverinfo='text',
                    showlegend=True
                ))
                fig.update_layout(
                    title=f"{viz_method} Clustering with Convex Hulls — {method}",
                    xaxis_title="comp 1",
                    yaxis_title="comp 2",
                    legend_title="Cluster",
                    template="plotly_white"
                )
            elif n_components == 3:
                fig.add_trace(go.Scatter(
                    x=cluster_df['comp_0'],
                    y=cluster_df['comp_1'],
                    z=cluster_df['comp_2'],
                    mode='markers',
                    name=cluster,
                    legendgroup=cluster,
                    marker=dict(color=color, size=6),
                    customdata=customdata,
                    hovertext=[
                        f"<b>{title}</b><br>{authors}"
                        for title, authors in zip(cluster_df['title'], cluster_df['authors'])
                    ],
                    hoverinfo='text',
                    showlegend=True
                ))

                fig.update_layout(
                    title=f"{viz_method} Clustering with Convex Hulls — {method}",
                    xaxis_title="comp 1",
                    yaxis_title="comp 2",
                    zaxis_title="comp 3",
                    legend_title="Cluster",
                    template="plotly_white"
                )
            elif n_components >= 4:
                raise ValueError(f"Unsupported due to creator's limitation to view in {n_components}D")
            else:
                raise ValueError(f"Invalid n_components: {n_components}")

            # Convex hull
            if len(cluster_points) >= 3:
                hull = ConvexHull(cluster_points)
                vertices = cluster_points[hull.vertices]
                vertices = np.vstack([vertices, vertices[0]])

                fig.add_trace(go.Scatter(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    mode='lines',
                    name=cluster,
                    legendgroup=cluster,
                    line=dict(color=color, width=1.5, dash='dot'),
                    fill='toself',
                    fillcolor=rgba_fill,
                    opacity=0.6,
                    showlegend=False,
                    hoverinfo='skip'
                ))

        # Save to HTML
        os.makedirs(os.path.join(output_path, f'{viz_method}'), exist_ok=True)
        # After fig.write_html(...)
        html_path = os.path.join(output_path, f'{viz_method}', f"{viz_method}_{method}.html")

        # Save first
        fig.write_html(html_path)

        # Inject JS after saving
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        injection = """
        <script>
          var plot = document.querySelectorAll(".js-plotly-plot")[0];
          plot.on("plotly_click", function(data){
            var point = data.points[0];
            var url = point.customdata;
            window.open(url, "_blank");
          });
        </script>
        </body>
        """

        html_content = html_content.replace("</body>", injection)

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

def write_paper_html(df, paper_dir):
    columns_to_include = [
        col for col in df.columns
        if col not in {"forum_content", "emb"}
    ]

    for _, row in df.iterrows():
        paper_id = row["id"]
        file_path = os.path.join(paper_dir, f"{paper_id}.html")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"<html><head><meta charset='utf-8'><title>{row['title']}</title></head><body>")
            f.write(f"<h1>{row['title']}</h1>")
            f.write(f"<h3>By: {row['authors']}</h3>")
            f.write("<hr>")

            for col in columns_to_include:
                if col in ("title", "authors", "id"):
                    continue
                f.write(f"<h4>{col}</h4>")
                f.write(f"<div style='margin-bottom:1em'>{row[col]}</div>")

            f.write("</body></html>")






