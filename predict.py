import os
import datetime
import argparse
import pandas as pd
from joblib import load

from utils import get_features, get_pred_df, evaluation_score, compute_distance_matrix, save_results

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    # if x < 0.01 or x > 0.2:
    #     raise argparse.ArgumentTypeError("%r not in range [0.01, 0.2]"%(x,))
    return x


parser = argparse.ArgumentParser()

parser.add_argument('--source_dir', type=str, default='birch',
                    help="The location of the model to be used for prediction.")
parser.add_argument('--dataset', choices=['apache', 'drone'], default='drone',
                    help="Dataset to test. Default: drone")
parser.add_argument('--embedding', choices=['sbert', 'instructor-base', 'instructor-large', 'instructor-xl', 'drone-sbert', 'simcse'], default='sbert',
                    help="Embedding model to extract the log's feature. Default: sbert")
parser.add_argument('--threshold', type=restricted_float,
                    help="Distance threshold for same cluster criteria [0.01,0.2]. Default: 0.07")


def main():
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    args = parser.parse_args()

    if args.dataset == 'drone':
        dataset = pd.read_excel(os.path.join('dataset', 'cluster_label.xlsx')).sort_values(by='message').reset_index(drop=True)
        labels_true = dataset['cluster_id'].to_list()
    elif args.dataset == 'apache':
        dataset = pd.read_csv(os.path.join('dataset', 'Apache_2k.log_structured.csv')).sort_values(by='Content').reset_index(drop=True)
        dataset.rename(columns = {'Content': 'message', 'EventId': 'cluster_id'}, inplace = True)
        labels_true = dataset['cluster_id'].to_list()
    print(dataset.head(5))
    source_workdir = os.path.join('experiments', args.source_dir, args.dataset, args.embedding, str(args.threshold))
    output_dir = os.path.join('predictions', args.source_dir, args.dataset, args.embedding, str(args.threshold))
    print(f"[predict.py] - Current scenario: {source_workdir}")
    if os.path.exists(output_dir):
        if os.path.exists(os.path.join(output_dir, 'scenario_arguments.json')):
            print("[predict.py] - Scenario has been executed.")
            return 0
    else:
        os.makedirs(output_dir)

    corpus_embeddings = get_features(dataset, args.embedding)

    # Load the model from the file
    joblib_model = load(os.path.join(source_workdir, 'birch_model.joblib'))

    started_at = datetime.datetime.now()
    predicted_joblib = joblib_model.predict(corpus_embeddings)
    ended_at = datetime.datetime.now()

    cluster_label_df = get_pred_df(predicted_joblib, dataset)
    eval_score = evaluation_score(corpus_embeddings, dataset, cluster_label_df.sort_values(by='message').reset_index(drop=True))
    print(cluster_label_df.head(5))
    
    duration = ended_at - started_at
    arguments_dict = vars(args)
    arguments_dict['started_at'] = str(started_at)
    arguments_dict['ended_at'] = str(ended_at)
    arguments_dict['duration'] = str(duration.total_seconds()) + ' seconds'
    arguments_dict['eval_score'] = eval_score

    save_results(arguments_dict, cluster_label_df, output_dir)

    return 0


if __name__ == "__main__":
    main()