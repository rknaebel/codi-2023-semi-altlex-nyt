import glob

import pandas as pd

column_names = ('corpus,type,doc_id,paragraph,sentence,signal_prob,indices,signal,'
                'sense1,sense1_prob,sense2,sense2_prob').split(',')


# corpora = ['pdtb3', 'essay', 'ted', 'unsc', 'bbc', 'anthology']


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

# def iter_dataframes():
#     for c in corpora:
#         df = load_dataframe(c)
#         if df is not None:
#             yield df
