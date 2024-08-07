import os
import datetime
import argparse
import pandas as pd
from utils import get_features, get_pred_df, evaluate, compute_distance_matrix, save_results
from sklearn.cluster import DBSCAN
from utils import evaluate


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    # if x < 0.1 or x > 1:
    #     raise argparse.ArgumentTypeError("%r not in range [0.05, 0.2]"%(x,))
    return x


parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default='dbscan',
                    help="Folder to store the experimental results. Default: dbscan")
parser.add_argument('--embedding', choices=['sbert', 'instructor-base', 'instructor-large', 'instructor-xl', 'drone-sbert', 'simcse'], default='sbert',
                    help="Embedding model to extract the log's feature. Default: sbert")
parser.add_argument('--threshold', type=restricted_float,
                    help="Distance threshold for same cluster criteria [0.1,1]. Default: 0.1")


def main():
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    args = parser.parse_args()
    
    dataset = pd.read_excel(os.path.join('dataset', 'cluster_label.xlsx'))
    labels_true = dataset['cluster_id'].to_list()
    workdir = os.path.join('experiments', args.output_dir, args.embedding, str(args.threshold))
    print(f"[dbscan] - Current scenario: {workdir}")
    if os.path.exists(workdir):
        if os.path.exists(os.path.join(workdir, 'scenario_arguments.json')):
            print("[dbscan] - Scenario has been executed.")
            return 0
    else:
        os.makedirs(workdir)

    corpus_embeddings = get_features(dataset, args.embedding)
    distance_matrix = compute_distance_matrix(corpus_embeddings)
    
    clustering_model = DBSCAN(eps=args.threshold, min_samples=1, metric='cosine')
    
    started_at = datetime.datetime.now()
    clustering_model.fit(corpus_embeddings)
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
