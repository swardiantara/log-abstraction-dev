import os
import json
import datetime
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score
from sklearn.cluster import AgglomerativeClustering


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.01 or x > 0.2:
        raise argparse.ArgumentTypeError("%r not in range [0.05, 0.2]"%(x,))
    return x


parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default='agglomerative',
                    help="Folder to store the experimental results. Default: agglomerative")
parser.add_argument('--embedding', choices=['sbert', 'instructor-base', 'instructor-large', 'instructor-xl', 'drone-sbert'], default='sbert',
                    help="Embedding model to extract the log's feature. Default: sbert")
parser.add_argument('--linkage', choices=['ward', 'complete', 'average', 'single'], default='average',
                    help="Linkage method to be used. Default: average")
parser.add_argument('--threshold', type=restricted_float,
                    help="Distance threshold for same cluster criteria [0.2,0.05]. Default: 0.07")


def load_dataset(path):
    return pd.read_csv(path)


def get_pred_df(clustering, dataset):
    corpus = dataset['message'].to_list()
    pseudo_label = []
    log_message = []
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(clustering):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(corpus[sentence_id])
    
    for i, cluster in clustered_sentences.items():
        for element in cluster:
            pseudo_label.append(i)
            log_message.append(element)

    cluster_label = pd.DataFrame({
        'message': log_message,
        'cluster_id': pseudo_label
    })

    return cluster_label


def evaluate(input_features, labels_pred, labels_true):
    silhouette_avg = silhouette_score(input_features, labels_pred)
    calinski_harabasz_avg = calinski_harabasz_score(input_features, labels_pred)
    ami_score = adjusted_mutual_info_score(labels_true, labels_pred)

    return ami_score, silhouette_avg, calinski_harabasz_avg


def get_features(dataset, embedding):
    corpus = dataset['message'].to_list()
    if embedding == 'sbert':
        embedding_model = SentenceTransformer('all-mpnet-base-v2')
        corpus_embeddings = embedding_model.encode(corpus)
    elif embedding == 'drone-sbert':
        model_path = os.path.join('experiments', 'embeddings')
        embedding_model = SentenceTransformer(model_path)
        corpus_embeddings = embedding_model.encode(corpus)
    else:
        embedding_model = INSTRUCTOR(f'hkunlp/{embedding}')
        log_dict = []
        for ind in dataset.index:
            log_dict.append(['Represent the Drone Log message for clustering: ', dataset['message'][ind]])
        corpus_embeddings = embedding_model.encode(log_dict)
    
    return corpus_embeddings


def compute_distance_matrix(corpus_embeddings, is_norm=True):
    if is_norm:
        corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    distance_matrix = pairwise_distances(corpus_embeddings, corpus_embeddings, metric='cosine')
    return distance_matrix


def save_results(arguments_dict, cluster_label_df, workdir):
    file_path = os.path.join(workdir, 'prediction.xlsx')
    cluster_label_df.to_excel(file_path, index=False)
    with open(os.path.join(workdir, 'scenario_arguments.json'), 'w') as json_file:
        json.dump(arguments_dict, json_file, indent=4)


def main():
    args = parser.parse_args()
    
    dataset = pd.read_excel(os.path.join('dataset', 'cluster_label.xlsx'))
    labels_true = dataset['cluster_id'].to_list()
    workdir = os.path.join('experiments', args.output_dir, args.embedding, args.linkage, str(args.threshold))
    print(f"[agglomerative] - Current scenario: {workdir}")
    if os.path.exists(workdir):
        if os.path.exists(os.path.join(workdir, 'scenario_arguments.json')):
            print("[agglomerative] - Scenario has been executed.")
            return 0
    else:
        os.makedirs(workdir)

    corpus_embeddings = get_features(dataset, args.embedding)
    distance_matrix = compute_distance_matrix(corpus_embeddings)
    
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                    metric='precomputed',
                                    linkage=args.linkage,
                                    distance_threshold=args.threshold)
    started_at = datetime.datetime.now()
    clustering_model.fit(distance_matrix)
    ended_at = datetime.datetime.now()
    
    ami_score, silhouette_avg, calinski_harabasz_avg = evaluate(corpus_embeddings, clustering_model.labels_, labels_true)
    cluster_label_df = get_pred_df(clustering_model.labels_, dataset)
    
    duration = ended_at - started_at
    arguments_dict = vars(args)
    arguments_dict['started_at'] = str(started_at)
    arguments_dict['ended_at'] = str(ended_at)
    arguments_dict['duration'] = str(duration.total_seconds()) + ' seconds'
    arguments_dict['ami_score'] = str(ami_score)
    arguments_dict['silhouette_avg'] = str(silhouette_avg)
    arguments_dict['calinski_harabasz_avg'] = str(calinski_harabasz_avg)

    save_results(arguments_dict, cluster_label_df, workdir)

    return 0


if __name__ == "__main__":
    main()
