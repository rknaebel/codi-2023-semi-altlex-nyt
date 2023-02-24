# a discourse signal is defined by its type, the corresponding token indices, and its attached sense.
# signal = (type: ['altlex', 'explicit'], indices: List[int], sense: List[str])
import numpy as np

from helpers.data import get_sense


def evaluate_connectives(gold_list, predicted_list, threshold=0.9, signal_types=('explicit',)):
    explicit_gold_list = [indices for (rtype, indices, senses) in gold_list if rtype in signal_types]
    explicit_predicted_list = [indices for (rtype, indices, senses) in predicted_list if rtype in signal_types]
    connective_cm, unmatched = compute_confusion_counts(
        explicit_gold_list, explicit_predicted_list, compute_span_f1, threshold)
    return connective_cm, unmatched


def compute_span_f1(g_index_set, p_index_set):
    g_index_set = set(g_index_set)
    p_index_set = set(p_index_set)
    correct = len(g_index_set.intersection(p_index_set))
    if correct == 0:
        return 0.0
    precision = float(correct) / len(p_index_set)
    recall = float(correct) / len(g_index_set)
    return 2 * (precision * recall) / (precision + recall)


def evaluate_sense(gold_list, predicted_list, threshold=0.9):
    tp = fp = fn = 0
    gold_to_predicted_map = _link_gold_predicted(gold_list, predicted_list, threshold)
    for gi, (gtype, gidxs, gs) in enumerate(gold_list):
        if gi in gold_to_predicted_map:
            # TODO check change
            # if any(g.startswith(p) for g in gr.senses for p in predicted_list[gold_to_predicted_map[gi]].senses):
            if any(g in predicted_list[gold_to_predicted_map[gi]][2] for g in gs):
                tp += 1
            else:
                fp += 1
        else:
            fn += 1

    return np.array([tp, fp, fn]), gold_to_predicted_map


def compute_confusion_counts(gold_list, predicted_list, matching_fn, threshold=0.9):
    """
    Args:
        gold_list:
        predicted_list:
        matching_fn:
        threshold:
    """
    tp = fp = 0
    unmatched = np.ones(len(predicted_list), dtype=bool)
    for gold_span in gold_list:
        for i, predicted_span in enumerate(predicted_list):
            if unmatched[i] and matching_fn(gold_span, predicted_span) >= threshold:
                tp += 1
                unmatched[i] = 0
                break
        else:
            fp += 1
    # Predicted span that does not match with any
    fn = unmatched.sum()

    return np.array([tp, fp, fn]), np.nonzero(unmatched)[0]


def compute_prf(tp, fp, fn):
    """
    Args:
        tp:
        fp:
        fn:
    """
    if tp + fp == 0:
        precision = 1.0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 1.0
    else:
        recall = tp / (tp + fn)

    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def _link_gold_predicted(gold_list, predicted_list, threshold=0.9):
    """Link gold relations to the predicted relations that fits best based on
    the almost exact matching criterion

    Args:
        gold_list:
        predicted_list:
        threshold:
    """
    gold_to_predicted_map = {}

    for gi, (gtype, gidxs, gs) in enumerate(gold_list):
        for pi, (ptype, pidxs, ps) in enumerate(predicted_list):
            if compute_span_f1(gidxs, pidxs) >= threshold:
                gold_to_predicted_map[gi] = pi
    return gold_to_predicted_map


def get_surface_tokens_context(doc, idxs, ctx_size=5):
    tokens = [t.surface for s in doc.sentences for t in s.tokens]
    ctx_min = max(0, min(idxs) - ctx_size)
    ctx_max = max(idxs) + ctx_size
    return ' '.join((f"_{t}_" if t_i in idxs else t) for t_i, t in enumerate(tokens) if ctx_min <= t_i <= ctx_max)


def score_df(df):
    counts_explicits = []
    counts_altlex = []
    for group_index, group in df.groupby(['doc_id']):
        doc = docs_by_id[group_index]
        gold_relations = [
            (rel.type.lower(), [t.idx for t in rel.conn.tokens], [get_sense(s, 2) for s in rel.senses])
            for rel in doc.relations if len(rel.conn.tokens)
        ]
        #     print(group_index)
        #     print(gold_relations)
        predicted_relations = [
            (row.type.lower(), [int(i) for i in row.indices.split('-')], [row.sense2])
            for k, row in group.iterrows()
        ]
        #     print(predicted_relations)
        explicit_counts, explicit_unmatched = evaluate_connectives(gold_relations, predicted_relations, threshold=0.90)
        altlex_counts, altlex_unmatched = evaluate_connectives(gold_relations, predicted_relations, threshold=0.90,
                                                               signal_types=('altlex',))

        for unmatched_idx in explicit_unmatched:
            print(get_surface_tokens_context(doc, predicted_relations[unmatched_idx][1], ctx_size=5))
        for unmatched_idx in altlex_unmatched:
            print(get_surface_tokens_context(doc, predicted_relations[unmatched_idx][1], ctx_size=5))

        counts_explicits.append(explicit_counts)
        counts_altlex.append(altlex_counts)

    print('explicits:', compute_prf(*np.sum(np.stack(counts_explicits), axis=0)))
    print('altlex:', compute_prf(*np.sum(np.stack(counts_altlex), axis=0)))
