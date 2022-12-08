import numpy as np
import umap
import matplotlib.pyplot as plt
import mpld3

def normalize_layout(layout, min_percentile=1, max_percentile=99, relative_margin=0.1):
    """Removes outliers and scales layout to between [0,1]."""

    # compute percentiles
    mins = np.percentile(layout, min_percentile, axis=(0))
    maxs = np.percentile(layout, max_percentile, axis=(0))

    # add margins
    mins -= relative_margin * (maxs - mins)
    maxs += relative_margin * (maxs - mins)

    # `clip` broadcasts, `[None]`s added only for readability
    clipped = np.clip(layout, mins, maxs)

    # embed within [0,1] along both axes
    clipped -= clipped.min(axis=0)
    clipped /= clipped.max(axis=0)

    return clipped

def reduce_dim_umap(activations):
    layout = umap.UMAP(n_components=2, verbose=True, n_neighbors=2, min_dist=0.01, metric="cosine").fit_transform(activations)
    return layout

def plot_umap(xs, ys, data_sources, organs, filenames, layer_to_use):
    fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'), figsize=(16, 8))
    colors = {"lung":"r", "kidney":"b", "largeintestine":"y", "prostate":"c", "spleen":"g"}
    markers = {"HPA":"o", "HUBMAP":"s"}
    colors_list = np.array([colors[org] for org in organs])
    hpa_idx = []
    hubmap_idx = []
    for i, src in enumerate(data_sources):
        if src == "HPA":
            hpa_idx.append(i)
        else:
            hubmap_idx.append(i)

    scatter = ax.scatter(xs[hpa_idx], ys[hpa_idx], color=colors_list[hpa_idx], marker=markers["HPA"])
    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=list(np.array(filenames)[hpa_idx]))
    mpld3.plugins.connect(fig, tooltip)   

    scatter = ax.scatter(xs[hubmap_idx], ys[hubmap_idx], color=colors_list[hubmap_idx], marker=markers["HUBMAP"])
    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=list(np.array(filenames)[hubmap_idx]))
    mpld3.plugins.connect(fig, tooltip)    

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    m = [plt.Line2D([0,0],[0,0], color="black", marker=marker, linestyle='') for marker in markers.values()]
    c = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colors.values()]
    legend1 = ax.legend(c+m, list(colors.keys())+list(markers.keys()), loc="center left", bbox_to_anchor=(1, 0.5))
    ax.add_artist(legend1)

    ax.set_title(f'UMAP projection of the Hubmap + HPA dataset - Layer {layer_to_use}')
    mpld3.save_html(fig,f"Umapplot_{layer_to_use[0]}.html")
