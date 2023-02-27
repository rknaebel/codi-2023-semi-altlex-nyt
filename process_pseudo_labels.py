import csv
import itertools
from pathlib import Path

import click
import pandas as pd

from helpers.stats import column_names


def get_sample_dict(df, threshold=0.5):
    d = {}
    for (corpus, rtype, doc_id, par_i), group_df in df.groupby(['corpus', 'type', 'doc_id', 'paragraph']):
        min_group_prob = ((group_df.signal_prob + group_df.sense2_prob) / 2).min()
        if min_group_prob > threshold:
            d[doc_id, par_i] = group_df.to_dict(orient='records')
    return d


def open_output_file(output_path, replace=True):
    output_path = Path(output_path)
    if output_path.is_file() and output_path.stat().st_size > 100 and not replace:
        raise FileExistsError('File already exists: Exit without writing.')
    else:
        output_path.parent.mkdir(exist_ok=True)
        output = output_path.open('w')
    return output


@click.command()
@click.argument('predictions-1')
@click.argument('predictions-2')
@click.option('--output-1', default='-')
@click.option('--output-2', default='-')
@click.option('-t', '--threshold', default=0.8, type=float)
@click.option('-b', '--use-both', is_flag=True)
def main(predictions_1, predictions_2, output_1, output_2, threshold, use_both):
    output_1 = open_output_file(output_1)
    output_2 = open_output_file(output_2)

    df1 = pd.read_csv(predictions_1, names=column_names)
    df2 = pd.read_csv(predictions_2, names=column_names)

    samples1 = get_sample_dict(df1, threshold=threshold)
    samples2 = get_sample_dict(df2, threshold=threshold)

    doc_ids = set(samples1.keys()) | set(samples2.keys())

    docs_both = []
    docs_m1 = []
    docs_m2 = []
    for doc_id in doc_ids:
        if doc_id in samples1 and doc_id in samples2:
            signals1 = {s['indices'] for s in samples1[doc_id]}
            signals2 = {s['indices'] for s in samples2[doc_id]}
            if signals1 == signals2:
                docs_both.append(doc_id)
            else:
                if len(signals1 - signals2):
                    docs_m2.append(doc_id)
                if len(signals2 - signals1):
                    docs_m1.append(doc_id)
        else:
            if doc_id in samples1:
                docs_m2.append(doc_id)
            if doc_id in samples2:
                docs_m1.append(doc_id)

    csv_columns = ['corpus', 'type', 'doc_id', 'paragraph', 'sentence',
                   'signal_prob', 'indices', 'signal', 'sense1', 'sense1_prob', 'sense2', 'sense2_prob']
    csv_out = csv.DictWriter(output_1, fieldnames=csv_columns)
    for sample_id in itertools.chain(docs_both, docs_m2):
        print(samples1[sample_id])
        csv_out.writerows(samples1[sample_id])

    csv_out = csv.DictWriter(output_2, fieldnames=csv_columns)
    for sample_id in itertools.chain(docs_both, docs_m1):
        csv_out.writerows(samples2[sample_id])


if __name__ == '__main__':
    main()
