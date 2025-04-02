import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def outline_plot(k_means, embeddings, df, current_id, doc_type, model_name, palette, global_x_min, global_x_max, global_y_min, global_y_max):

    # Convert embeddings to NumPy array
    embeddings = np.array(embeddings)

    # Perform t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=min(30, (len(embeddings) - 1) // 3), random_state=42)
    data_map = tsne.fit_transform(embeddings)

    # Store axis limits globally
    x_min, x_max = data_map[:, 0].min(), data_map[:, 0].max()
    y_min, y_max = data_map[:, 1].min(), data_map[:, 1].max()

    # Update global axis limits if this is the first plot or new extremes are found
    if global_x_min is None or x_min < global_x_min:
        global_x_min = x_min
    if global_x_max is None or x_max > global_x_max:
        global_x_max = x_max
    if global_y_min is None or y_min < global_y_min:
        global_y_min = y_min
    if global_y_max is None or y_max > global_y_max:
        global_y_max = y_max

    # Prepare DataFrame for plotting
    df_plot = pd.DataFrame(data_map, columns=["x", "y"])
    df_plot["title"] = df["Title"].values
    df_plot["cluster_int"] = k_means['topic_int']
    df_plot["cluster_name"] = k_means['main_topic_name']
    df_plot["source"] = df["Source"].values  # Ensure DataFrame contains "Source" column

    # Dynamically define markers for all unique sources
    unique_sources = df["Source"].unique()
    marker_styles = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>']  # Add more if needed
    markers = {source: marker_styles[i % len(marker_styles)] for i, source in enumerate(unique_sources)}

    # Plot points and cluster outlines
    plt.figure(figsize=(10, 8))
    for cluster_id in sorted(df_plot["cluster_int"].unique()):
        cluster_points = df_plot[df_plot["cluster_int"] == cluster_id][["x", "y"]].values
        cluster_sources = df_plot[df_plot["cluster_int"] == cluster_id]["source"]
        
        # Compute and plot convex hull for the cluster
        if len(cluster_points) >= 3:  # ConvexHull requires at least 3 points
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]
            plt.plot(
                np.append(hull_points[:, 0], hull_points[0, 0]),
                np.append(hull_points[:, 1], hull_points[0, 1]),
                linestyle='--', linewidth=1.5
            )

        for source in palette.keys():
            source_points = df_plot[(df_plot["cluster_int"] == cluster_id) & (df_plot["source"] == source)][["x", "y"]].values
            plt.scatter(source_points[:, 0], source_points[:, 1], color=palette[source], s=10, label=source, marker=markers[source])


        # Add cluster name as text at the centroid of the cluster
        centroid_x = cluster_points[:, 0].mean()
        centroid_y = cluster_points[:, 1].mean()
        cluster_name = df_plot[df_plot["cluster_int"] == cluster_id]["cluster_name"].iloc[0]
        plt.text(centroid_x, centroid_y, cluster_name, fontsize=10, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.5))

    # Add legend for the sources
    handles = [plt.Line2D([0], [0], marker=markers[source], color='w', markerfacecolor=color, markersize=10, label=source)
               for source, color in palette.items()]
    
    plt.legend(handles=handles, title="Source")

    plt.title(f"Cluster Source Visualization for {model_name}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.xlim(global_x_min, global_x_max)
    plt.ylim(global_y_min, global_y_max)
    plt.grid(True)

    # Save Visualization to CSV
    output_dir = 'outputs/modelling/'

    plt.savefig(f"{output_dir}{current_id}_{doc_type}_{model_name}_outlineplot.png")
    print(f"{model_name} with {doc_type}: Outline Source plot saved successfully for {current_id}")

    return global_x_min, global_x_max, global_y_min, global_y_max



