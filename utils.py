import os
import json
import torch
import numpy as np
import pandas as pd
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score


def get_features(dataset, embedding):
    corpus = dataset['message'].to_list()
    if embedding == 'sbert':
        embedding_model = SentenceTransformer('all-mpnet-base-v2')
        corpus_embeddings = embedding_model.encode(corpus)
    elif embedding == 'drone-sbert':
        model_path = os.path.join('experiments', 'embeddings')
        embedding_model = SentenceTransformer(model_path)
        corpus_embeddings = embedding_model.encode(corpus)
    elif embedding == 'simcse':
        model_path = 'princeton-nlp/sup-simcse-roberta-large'
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        inputs = tokenizer(corpus, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            corpus_embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    else:
        embedding_model = INSTRUCTOR(f'hkunlp/{embedding}')
        log_dict = []
        for ind in dataset.index:
            log_dict.append(['Represent the Drone Log message for clustering: ', dataset['message'][ind]])
        corpus_embeddings = embedding_model.encode(log_dict)
    
    return corpus_embeddings


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
    }).sort_values(by='message')

    return cluster_label


def evaluate(input_features, labels_pred, labels_true):
    try:
        silhouette_avg = silhouette_score(input_features, labels_pred)
    except:
        silhouette_avg = -1
    
    try:
        calinski_harabasz_avg = calinski_harabasz_score(input_features, labels_pred)
    except:
        calinski_harabasz_avg = -1
    ami_score = adjusted_mutual_info_score(labels_true, labels_pred)

    return ami_score, silhouette_avg, calinski_harabasz_avg


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
