import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from langfuse import Langfuse
from loader import load_metadata
from openai import AzureOpenAI
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_embedding(openai_client: AzureOpenAI, embedding_model: str, text: str):
    response = openai_client.embeddings.create(input=[text], model=embedding_model)
    return response.data[0].embedding


def generate_cluster_keywords(
    openai_client: AzureOpenAI,
    model: str,
    clusters,
    documents,
):
    """Generate keywords for each cluster using OpenAI GPT."""
    cluster_keywords = {}
    for label, keys in clusters.items():
        # Concatenate summaries for the cluster
        summaries = [
            documents[k]["metadata"].get("summary", "") for k in keys[:10]
        ]  # limit to 10 for prompt size

        prompt = (
            "Given the following document summaries, generate 1 to 3 short keywords that best describe the common topic or theme. "
            "Return only the keywords, comma-separated.\n\n"
            + "\n".join(f"- {s}" for s in summaries if s)
        )

        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0,
            )
            keywords = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating keywords for cluster {label}: {e}")
            keywords = ""
        cluster_keywords[label] = keywords
    return cluster_keywords


def visualize_clusters(
    embeddings,
    labels,
    ordered_keys,
    clusters,
    data_folder: Path,
    cluster_keywords=None,
):
    plt.style.use("seaborn-v0_8-whitegrid")  # Use a modern style
    # PCA Visualization
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap="tab20",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.3,
    )
    # Annotate a few points per cluster
    for _, keys in clusters.items():
        for key in keys[:2]:  # Annotate up to 2 per cluster
            idx = ordered_keys.index(key)
            plt.annotate(key, (reduced[idx, 0], reduced[idx, 1]), fontsize=7, alpha=0.6)
    plt.title("Document Clusters (PCA 2D)", fontsize=14, fontweight="bold")
    plt.xlabel("PCA 1", fontsize=12)
    plt.ylabel("PCA 2", fontsize=12)
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plot_path = data_folder / "generated_data/plots/clusters_plot.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info(f"Cluster plot saved to {plot_path}")

    # t-SNE Visualization
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    tsne_reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    tsne_scatter = plt.scatter(
        tsne_reduced[:, 0],
        tsne_reduced[:, 1],
        c=labels,
        cmap="tab20",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.3,
    )
    for label, keys in clusters.items():
        for key in keys[:2]:
            idx = ordered_keys.index(key)
            plt.annotate(
                key,
                (tsne_reduced[idx, 0], tsne_reduced[idx, 1]),
                fontsize=7,
                alpha=0.6,
            )
        # Annotate cluster centroid with keywords if provided
        if cluster_keywords and label in cluster_keywords:
            idxs = [ordered_keys.index(k) for k in keys]
            centroid = tsne_reduced[idxs].mean(axis=0)
            plt.annotate(
                cluster_keywords[label],
                (centroid[0], centroid[1]),
                fontsize=10,
                fontweight="bold",
                color="darkred",
                alpha=0.85,
            )
    plt.title("Document Clusters (t-SNE 2D)", fontsize=14, fontweight="bold")
    plt.xlabel("t-SNE 1", fontsize=12)
    plt.ylabel("t-SNE 2", fontsize=12)
    plt.colorbar(tsne_scatter, label="Cluster")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    tsne_plot_path = data_folder / "generated_data/plots/clusters_tsne_plot.png"
    plt.savefig(tsne_plot_path, dpi=200)
    plt.close()
    logger.info(f"t-SNE cluster plot saved to {tsne_plot_path}")


def analyze_embedding_distribution(embeddings, data_folder: Path):
    """
    Save histograms of the distribution of embedding values for each dimension.
    Also save a summary plot of mean and std per dimension.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    embeddings = np.array(embeddings)
    n_dims = embeddings.shape[1]
    # Plot mean and std per dimension
    means = embeddings.mean(axis=0)
    stds = embeddings.std(axis=0)
    plt.figure(figsize=(12, 6))
    plt.plot(means, label="Mean", color="blue")
    plt.plot(stds, label="Std", color="orange")
    plt.title("Embedding Dimension Mean and Std")
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    summary_path = data_folder / "generated_data/plots/embedding_dim_mean_std.png"
    plt.savefig(summary_path, dpi=200)
    plt.close()
    logger.info(f"Saved embedding mean/std summary to {summary_path}")
    # Plot histograms for a sample of dimensions (first 10, or all if <10)
    n_plot = min(10, n_dims)
    plt.figure(figsize=(15, 8))
    for i in range(n_plot):
        plt.subplot(2, 5, i + 1)
        plt.hist(
            embeddings[:, i],
            bins=30,
            color="skyblue",
            edgecolor="k",
            alpha=0.7,
        )
        plt.title(f"Dim {i}")
        plt.xlabel("Value")
        plt.ylabel("Count")
    plt.tight_layout()
    hist_path = data_folder / "generated_data/plots/embedding_dim_histograms.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()
    logger.info(f"Saved embedding histograms to {hist_path}")


def find_optimal_k(embeddings, data_folder: Path, k_range=range(1, 3, 1)):
    inertias = []
    silhouettes = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(embeddings)
        inertias.append(kmeans.inertia_)
        if 2 <= k < len(embeddings):
            score = silhouette_score(embeddings, labels)
        else:
            score = np.nan
        silhouettes.append(score)
    # Plot inertia (elbow)
    plt.figure(figsize=(10, 5))
    plt.plot(list(k_range), inertias, marker="o", label="Inertia")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    elbow_path = data_folder / "generated_data/plots/elbow_plot.png"
    plt.savefig(elbow_path, dpi=400)
    plt.close()
    logger.info(f"Elbow plot saved to {elbow_path}")
    # Plot silhouette score
    plt.figure(figsize=(10, 5))
    plt.plot(
        list(k_range),
        silhouettes,
        marker="o",
        label="Silhouette Score",
        color="orange",
    )
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Different k")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    silhouette_path = data_folder / "generated_data/plots/silhouette_plot.png"
    plt.savefig(silhouette_path, dpi=400)
    plt.close()
    logger.info(f"Silhouette plot saved to {silhouette_path}")
    # Return best k by silhouette score
    best_k = k_range[np.nanargmax(silhouettes)]
    logger.info(f"Best k by silhouette score: {best_k}")
    return best_k


def cluster(
    openai_client: AzureOpenAI,
    model: str,
    embedding_model: str,
    data_folder: Path,
):
    metadata = load_metadata(data_folder)
    ordered_keys = sorted(metadata.keys(), key=lambda x: int(x))
    texts = [
        f"topic: {metadata[key]['metadata'].get('topic', '')}\nsummary: {metadata[key]['metadata'].get('summary', '')}\ntags: {' '.join(metadata[key]['metadata'].get('tags', []))}"
        for key in ordered_keys
    ]

    # Embedding cache
    embedding_cache_path = data_folder / "generated_data/corpuses_embeddings.json"
    if embedding_cache_path.exists():
        with open(embedding_cache_path) as f:
            embedding_cache = json.load(f)
    else:
        embedding_cache = {}

    embeddings = []
    updated = False
    for key, text in tqdm(zip(ordered_keys, texts, strict=False)):
        if key in embedding_cache:
            emb = embedding_cache[key]
        else:
            emb = get_embedding(openai_client, embedding_model, text)
            embedding_cache[key] = emb
            updated = True
        embeddings.append(emb)

    # Save updated cache if new embeddings were added
    if updated:
        with open(embedding_cache_path, "w") as f:
            json.dump(embedding_cache, f)
        logger.info(f"Embeddings cache updated at {embedding_cache_path}")
    else:
        logger.info(f"Loaded embeddings from cache at {embedding_cache_path}")

    embeddings = np.array(embeddings)
    embeddings = normalize(embeddings, norm="l2")
    # Find optimal k using elbow and silhouette
    best_k = find_optimal_k(embeddings, data_folder)
    logger.info(f"Using best_k={best_k} for clustering.")
    n_clusters = best_k
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    clusters: dict = {}
    for idx, label in enumerate(labels):
        key = ordered_keys[idx]
        clusters.setdefault(int(label), []).append(key)

    # Generate cluster keywords
    cluster_keywords = generate_cluster_keywords(
        openai_client,
        model,
        clusters,
        metadata,
    )
    # Visualize clusters with keywords
    visualize_clusters(
        embeddings,
        labels,
        ordered_keys,
        clusters,
        data_folder,
        cluster_keywords=cluster_keywords,
    )
    # Save cluster keys and descriptions to JSON
    cluster_keys = {}
    for label, keys in clusters.items():
        desc = cluster_keywords.get(str(label)) or cluster_keywords.get(label) or ""
        cluster_keys[str(label)] = {"corpus_ids": keys, "description": desc}

    # Cluster file
    cluster_file = data_folder / "generated_data/corpuses_clusters.json"
    with open(cluster_file, "w") as f:
        json.dump(cluster_keys, f, indent=2)
    logger.info(f"Cluster keys and descriptions saved to {cluster_file}")
    # Analyze embedding distribution
    analyze_embedding_distribution(embeddings, data_folder)


def cluster_documents(
    data_folder: Path,
    langfuse_client: Langfuse,
    openai_client: AzureOpenAI,
    model_name: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
):
    """
    Assign documents to questions based on their embeddings.
    This function is a placeholder for the actual implementation.
    """
    logger.info("Clustering documents...")
    cluster(
        openai_client,
        model_name,
        embedding_model,
        data_folder,
    )
    logger.info("Document clustering completed.")
