import os
import json
import torch
import numpy as np
import pandas as pd
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score, adjusted_mutual_info_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score
from evaluation.group_accuracy import evaluate, singleton_accuracy, precision_recall_f1

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


def round_score(score_dict: dict, decimal=5):
    for key, value in score_dict.items():
        score_dict[key] = str(round(value, decimal))

    return score_dict

def evaluation_score(input_features, df_truth, df_pred):
    labels_pred = df_pred['cluster_id']
    labels_true = df_truth['cluster_id']
    try:
        silhouette_avg = silhouette_score(input_features, labels_pred)
    except:
        silhouette_avg = -1
    
    try:
        calinski_harabasz_avg = calinski_harabasz_score(input_features, labels_pred)
    except:
        calinski_harabasz_avg = -1
    ami_score = adjusted_mutual_info_score(labels_true, labels_pred)
    _, group_accuracy = evaluate(df_truth, df_pred)
    singleton_acc, true_singleton_indices, pred_singleton_indices = singleton_accuracy(df_truth, df_pred)
    _, _, singleton_f1 = precision_recall_f1(true_singleton_indices, pred_singleton_indices)
    nmi_score = normalized_mutual_info_score(labels_true, labels_pred)
    ari_score = adjusted_rand_score(labels_true, labels_pred)
    hgi_score = homogeneity_score(labels_true, labels_pred)
    cpi_score = completeness_score(labels_true, labels_pred)
    vmi_score = v_measure_score(labels_true, labels_pred)
    fmi_score = fowlkes_mallows_score(labels_true, labels_pred)

    score_dict =  {
        'group_accuracy': group_accuracy,
        'singleton_f1': singleton_f1,
        'ami_score': ami_score,
        'nmi_score': nmi_score,
        'ari_score': ari_score,
        'hgi_score': hgi_score,
        'cpi_score': cpi_score,
        'vmi_score': vmi_score,
        'fmi_score': fmi_score,
        'silhouette_avg': silhouette_avg,
        'calinski_harabasz_avg': calinski_harabasz_avg,
    }

    return round_score(score_dict)


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
