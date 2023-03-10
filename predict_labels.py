import json
import sys
from pathlib import Path

import click
import torch

from helpers.data import get_corpus_path, load_docs
from helpers.labeling import DiscourseSignalExtractor


@click.command()
@click.argument('corpus')
@click.argument('save-path')
@click.option('-r', '--replace', is_flag=True)
@click.option('-o', '--output-path', default='-')
@click.option('-l', '--limit', default=0, type=int)
def main(corpus, save_path, replace, output_path, limit):
    if output_path == '-':
        output = sys.stdout
    else:
        output_path = Path(output_path)
        if output_path.is_file() and output_path.stat().st_size > 100 and not replace:
            sys.stderr.write('File already exists: Exit without writing.')
            return
        else:
            output_path.parent.mkdir(exist_ok=True, parents=True)
            output = output_path.open('w')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    signal_model = DiscourseSignalExtractor.load_model(save_path, device=device)

    corpus_path = get_corpus_path(corpus)
    for doc_i, doc in enumerate(load_docs(corpus_path)):
        if limit and doc_i >= limit:
            break
        signals = signal_model.predict(doc)
        for s in signals:
            json.dump(s, output)
            output.write('\n')
        output.flush()


if __name__ == '__main__':
    main()
