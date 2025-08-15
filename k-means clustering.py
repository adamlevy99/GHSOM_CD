import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import random
from scipy.stats import chisquare




# Fix global random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Function to load CSV or Excel file via a file dialog
def load_csv():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Excel/CSV File",
        filetypes=[("CSV files","*.csv"), ("Excel files","*.xlsx;*.xls")]
    )
    if not file_path:
        raise ValueError("No file selected!")
    if file_path.lower().endswith(('.xlsx','.xls')):
        return pd.read_excel(file_path), file_path
    else:
        return pd.read_csv(file_path), file_path

# Silhouette score sweep with permutation p-values
def find_optimal_k(data, k_min=2, k_max=20, n_perms=200, random_state=42):
    """
    For each k in [k_min, k_max], compute:
      - obs_score: the usual silhouette score
      - p_val: fraction of permuted-label scores >= obs_score
    Returns a list of (k, obs_score, p_val).
    """
    rng = np.random.RandomState(random_state)
    results = []

    for k in range(k_min, k_max+1):
        # fit KMeans & get observed silhouette
        km = KMeans(n_clusters=k, random_state=random_state, n_init=5, max_iter=100)
        labels = km.fit_predict(data)
        obs_score = silhouette_score(
            data, labels,
            sample_size=min(len(data), 5000),
            random_state=random_state
        )

        # build null distribution by permuting the labels
        perm_scores = []
        for _ in range(n_perms):
            perm_labels = rng.permutation(labels)
            perm_scores.append(
                silhouette_score(
                    data, perm_labels,
                    sample_size=min(len(data), 5000),
                    random_state=random_state
                )
            )

        p_val = np.mean([s >= obs_score for s in perm_scores])
        results.append((k, obs_score, p_val))
        print(f"k={k}, silhouette={obs_score:.4f}, p-value={p_val:.3f}")

    return results


# Plot silhouette results (with optional p-value annotation)
def plot_silhouette(sil_avgs, out_folder, plot_pvals=False):
    """
    sil_avgs: list of (k, score, p_val)
    if plot_pvals=True, annotate each point with its p-value.
    """
    ks, scores, pvals = zip(*sil_avgs)
    fig, ax = plt.subplots()
    ax.plot(ks, scores, marker='o')
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Sweep")
    ax.grid(True)

    #if plot_pvals:
       # for x, y, p in zip(ks, scores, pvals):
           # ax.text(x, y + 0.01, f"p={p:.2f}", ha='center', va='bottom', fontsize=8)

    fp = os.path.join(out_folder, 'silhouette_scores.png')
    fig.savefig(fp)
    plt.close(fig)
    print(f"Silhouette plot saved: {fp}")

# Plot individual cluster profiles as spider (radar) plots
def plot_profiles(centroids, out_folder):
    labels = centroids.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # fixed radial ticks from -2 to +3
    rticks = [-2, 0, 2]

    for i, (cluster_id, profile) in enumerate(centroids.iterrows()):
        values = profile.tolist()
        values += values[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster_id}')
        ax.fill(angles, values, alpha=0.25)

        # fix radial axis limits and ticks
        ax.set_ylim(-2, 3)
        ax.set_yticks(rticks)
        ax.set_yticklabels([f"{t:.2f}" for t in rticks])
        ax.set_rlabel_position(angles[0] * 180/np.pi)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"Profile {i + 1}", y=1.08)

        plt.tight_layout()
        outfile = os.path.join(out_folder, f"profile_{i + 1}.png")
        fig.savefig(outfile, bbox_inches='tight', dpi=150)
        plt.close()

# Plot overall group profiles (Control vs CD) as spider plot
def plot_group_spider(group_centroids, out_folder):
    labels = group_centroids.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # fixed radial ticks from -2 to +2
    rticks = np.linspace(-2, 2, 5)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for group in group_centroids.index:
        values = group_centroids.loc[group].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=group)
        ax.fill(angles, values, alpha=0.15)

    # fix radial axis limits and ticks
    ax.set_ylim(-2, 2)
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{t:.2f}" for t in rticks])
    ax.set_rlabel_position(angles[0] * 180/np.pi)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Control vs CD Profile (z-scores)", y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    fp = os.path.join(out_folder, 'profile_control_vs_CD.png')
    plt.savefig(fp)
    plt.close()
    print(f"Control vs CD spider plot saved: {fp}")

# KMeans clustering
def perform_kmeans(data, k):
    km = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=42)
    return km.fit_predict(data)

# Compute cluster centroids (cognitive profiles)
def compute_profiles(data, labels):
    dfc = data.copy()
    dfc['Cluster'] = labels
    return dfc.groupby('Cluster').mean()

# Pairwise permutation tests on cluster profiles (Euclidean distance)
def permutation_test_between_profiles(data, labels, n_perms=200, random_state=42):
    rng = np.random.RandomState(random_state)
    clusters = np.unique(labels)
    centroids = compute_profiles(data, labels)
    rows = []
    for i, ci in enumerate(clusters):
        for cj in clusters[i+1:]:
            obs_dist = np.linalg.norm(centroids.loc[ci] - centroids.loc[cj])
            perm_dists = []
            for _ in range(n_perms):
                perm_lbls = rng.permutation(labels)
                perm_cent = compute_profiles(data, perm_lbls)
                dist = np.linalg.norm(perm_cent.loc[ci] - perm_cent.loc[cj])
                perm_dists.append(dist)
            p_val = np.mean([d >= obs_dist for d in perm_dists])
            rows.append({
                'Cluster_i': ci,
                'Cluster_j': cj,
                'ObsDist':   obs_dist,
                'p_value':   p_val
            })
    return pd.DataFrame(rows)


# Plot PCA cluster visualization
def plot_clusters_pca(data, labels, out_folder, k):
    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(data)
    plt.figure()
    for cl in np.unique(labels):
        mask = labels == cl
        plt.scatter(
            comps[mask, 0],
            comps[mask, 1],
            label=f"Cluster {cl}",
            alpha=0.7
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA of Clusters (k={k})")
    plt.legend()
    fp = os.path.join(out_folder, f"clusters_k{k}.png")
    plt.savefig(fp)
    plt.close()
    print(f"Cluster PCA plot saved: {fp}")

# Main execution
def main():
    # Load data
    df, path = load_csv()
    print(f"Loaded: {path}")
    out_folder = os.path.dirname(path)

    # --- Separate out metadata ---      <-- include CD_OnsetType here
    df['ID'] = df['ID'].astype(str)
    df['Sex'] = df['Sex'].astype(int)
    df['Group'] = df.get(
        'Group',
        np.where(df['ID'].astype(int) <= 542, 'CD', 'Control')
    )
    # keep CD_OnsetType in meta for mapping
    meta = df[['ID', 'Sex', 'Group', 'Age_raw', 'CD_OnsetType']].copy()
    meta['Sex'] = meta['Sex'].map({1: 'Female', 2: 'Male'})

    # include CD_OnsetType in metadata
    meta = df[['ID', 'Sex', 'Group', 'Age_raw', 'CD_OnsetType']].copy()
    meta['Sex'] = meta['Sex'].map({1: 'Female', 2: 'Male'})
    # --- Build feature matrix without ID/Sex/Group/Age_raw/CD_OnsetType ---
    features_df = df.drop(columns=['ID', 'Sex', 'Group', 'Age_raw', 'CD_OnsetType'])
    num_data = features_df.select_dtypes(include=[np.number])
    data = num_data.dropna()

    # Plot overall Control vs CD spider plot
    group_centroids = num_data.loc[data.index].copy()
    group_centroids['Group'] = meta.loc[data.index, 'Group']
    group_centroids = group_centroids.groupby('Group').mean()
    plot_group_spider(group_centroids, out_folder)


    # silhouette + permutation p-values
    sil_avgs = find_optimal_k(data, k_min=2, k_max=20, n_perms=200, random_state=42)
    plot_silhouette(sil_avgs, out_folder, plot_pvals=True)
    # pick k by max silhouette
    optk = max(sil_avgs, key=lambda x: x[1])[0]
    ui = input(f"Enter k or press Enter for optimal ({optk}): ")
    try:
        k = int(ui) if ui.strip() else optk
    except:
        k = optk
    print(f"Using k={k}")

    # Clustering
    labels = perform_kmeans(data, k)

    # Compute cluster centroids and plot profiles
    centroids = compute_profiles(data, labels)
    plot_profiles(centroids, out_folder)

    # Pairwise permutation tests
    print(f"Running permutation tests on {k} cluster profiles... (200 perms)")
    perm_df = permutation_test_between_profiles(data, labels, n_perms=200, random_state=42)

    # Re-attach metadata and save results
    map_df = meta.loc[data.index].copy()
    map_df['Cluster'] = labels

    # Calculate overall gender proportions
    overall_gender_counts = map_df['Sex'].value_counts(normalize=True)
    expected_props = {
        'Male': overall_gender_counts.get('Male', 0),
        'Female': overall_gender_counts.get('Female', 0)
    }

    print("\n=== Chi-Square Goodness-of-Fit Test: Gender Distribution Within Each Cluster ===")

    results = []

    # Loop through each cluster
    for cluster in sorted(map_df['Cluster'].unique()):
        cluster_data = map_df[map_df['Cluster'] == cluster]
        observed_counts = cluster_data['Sex'].value_counts()

        # Ensure both Male and Female are in observed
        observed = [observed_counts.get('Male', 0), observed_counts.get('Female', 0)]
        total = sum(observed)
        expected = [expected_props['Male'] * total, expected_props['Female'] * total]

        chi2, p = chisquare(f_obs=observed, f_exp=expected)

        results.append({
            'Cluster': cluster,
            'Male': observed[0],
            'Female': observed[1],
            'Chi2': round(chi2, 3),
            'p_value': round(p, 4)
        })
    gof_df = pd.DataFrame(results)


    out_excel = os.path.splitext(path)[0] + f"_results_k{k}.xlsx"
    with pd.ExcelWriter(out_excel, engine='openpyxl') as writer:
        # --- write the silhouette + p-values sheet correctly ---
        sil_df = pd.DataFrame(sil_avgs, columns=['k', 'Silhouette', 'p_value'])
        sil_df.set_index('k', inplace=True)
        sil_df.to_excel(writer, sheet_name='Silhouette')
        centroids.to_excel(writer, sheet_name='Profiles')
        perm_df.to_excel(writer, sheet_name='Permutation_Test', index=False)
        map_df.to_excel(writer, sheet_name='Mapping', index=False)
        summary_rows = []
        for cl in sorted(map_df['Cluster'].unique()):
            sub = map_df[map_df['Cluster'] == cl]
            summary_rows.append({
                'Cluster': cl,
                'Total': len(sub),
                'Male_CD': ((sub.Sex == 'Male') & (sub.Group == 'CD')).sum(),
                'Female_CD': ((sub.Sex == 'Female') & (sub.Group == 'CD')).sum(),
                'Male_Control': ((sub.Sex == 'Male') & (sub.Group == 'Control')).sum(),
                'Female_Control': ((sub.Sex == 'Female') & (sub.Group == 'Control')).sum(),
                # counts of CD_OnsetType
                'OnsetType_4': (sub['CD_OnsetType'] == 4).sum(),
                'OnsetType_5': (sub['CD_OnsetType'] == 5).sum(),
                'OnsetType_6': (sub['CD_OnsetType'] == 6).sum(),
                'Mean_Age': sub['Age_raw'].mean()
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name='Cluster_Summary', index=False)

        # Save Chi-square goodness-of-fit results per cluster
        gof_df.to_excel(writer, sheet_name='Gender_GOF_ChiSquare', index=False)

        print(f"Results saved to Excel: {out_excel}")

    # Cluster visualization
    plot_clusters_pca(data, labels, out_folder, k)

if __name__ == '__main__':
    main()


