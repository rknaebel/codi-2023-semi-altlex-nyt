import glob

import pandas as pd

column_names = ('corpus,type,doc_id,paragraph,sentence,signal_prob,indices,signal,'
                'sense1,sense1_prob,sense2,sense2_prob').split(',')


def load_dataframes(corpus):
    paths = glob.glob(f"results/m*/{corpus}.csv")
    dfs = []
    for path in paths:
        try:
            dfs.append(pd.read_csv(path, names=column_names))
        except Exception as e:
            print(e)
            continue
    return dfs


def print_metrics_results(results):
    for key, vals in results.items():
        if key == 'accuracy':
            print(f"{key:32}  {vals * 100:02.2f}")
        else:
            print(
                f"{key:32}  "
                f"{vals['precision'] * 100:05.2f}  "
                f"{vals['recall'] * 100:05.2f}  "
                f"{vals['f1-score'] * 100:05.2f}  "
                f"{vals['support']}")
    print('## ' + '= ' * 50)


def print_final_results(loss, results):
    print("\n===")
    print(f'=== Final Validation Score: {loss}')
    print(f'=== Final Validation Macro AVG: {results.get("macro avg")}')
    print(f'=== Final Validation Weighted AVG: {results.get("weighted avg")}')
    print("===")
