import itertools
import json
import sys
from pathlib import Path

import click
import torch

from helpers.data import get_corpus_path, load_docs
from helpers.labeling import DiscourseSignalExtractor


def iter_documents_paragraphs(docs):
    def fmt(par, par_i):
        return {
            'doc_id': doc.doc_id,
            'paragraph_idx': par_i,
            'sentences': par,
        }

    for doc in docs:
        par = []
        par_i = 0
        for s in doc.sentences:
            if len(par) == 0 or (par[-1].tokens[-1].offset_end + 1 == s.tokens[0].offset_begin):
                par.append(s)
            else:
                yield fmt(par, par_i)
                par_i += 1
                par = [s]
        yield fmt(par, par_i)


@click.command()
@click.argument('corpus')
@click.argument('save-path')
@click.option('-r', '--replace', is_flag=True)
@click.option('-o', '--output-path', default='-')
@click.option('-l', '--limit', default=0, type=int)
@click.option('-b', '--batch-size', default=32, type=int)
def main(corpus, save_path, replace, output_path, limit, batch_size):
    if output_path == '-':
        output = sys.stdout
    else:
        output_path = Path(output_path)
        if output_path.is_file() and output_path.stat().st_size > 100 and not replace:
            print('File already exists: Exit without writing.', file=sys.stderr)
            return
        else:
            output_path.parent.mkdir(exist_ok=True, parents=True)
            output = output_path.open('w')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    signal_model = DiscourseSignalExtractor.load_model(save_path, device=device)

    corpus_path = get_corpus_path(corpus)
    paragraphs = filter(lambda p: sum(len(s.tokens) for s in p['sentences']) > 7,
                        iter_documents_paragraphs(load_docs(corpus_path)))
    while True:
        batch = list(itertools.islice(paragraphs, batch_size))
        if len(batch) == 0:
            break
        signals = signal_model.predict_paragraphs(batch)
        for s in filter(lambda s: len(s['relations']) > 0, signals):
            json.dump(s, output)
            output.write('\n')
        output.flush()


if __name__ == '__main__':
    main()
