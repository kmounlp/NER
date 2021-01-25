from copy import deepcopy
from collections import namedtuple

def collect_named_entities(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.
    :param tokens: a list of labels
    :return: a list of Entity named-tuples
    """
    Entity = namedtuple("Entity", "e_type start_offset end_offset")

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, tag in enumerate(tokens):

        token_tag = tag

        if token_tag == 'O':
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:]:
            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type and start_offset and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, offset))

    return named_entities


def compute_metrics(true_named_entities, pred_named_entities):

    eval_metrics = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurius': 0}
    target_tags_no_schema = ['PNT', 'DUR', 'MNY', 'POH', 'ORG', 'NOH', 'DAT', 'TIM', 'LOC', 'PER']

    # overall results
    evaluation = {'strict': deepcopy(eval_metrics), 'ent_type': deepcopy(eval_metrics)}

    # results by entity type
    evaluation_agg_entities_type = {e: deepcopy(evaluation) for e in target_tags_no_schema}

    true_which_overlapped_with_pred = []  # keep track of entities that overlapped

    # go through each predicted named-entity
    for pred in pred_named_entities:
        found_overlap = False

        # check if there's an exact match, i.e.: boundary and entity type match
        if pred in true_named_entities:
            true_which_overlapped_with_pred.append(pred)
            evaluation['strict']['correct'] += 1
            evaluation['ent_type']['correct'] += 1

            # for the agg. by e_type results
            evaluation_agg_entities_type[pred.e_type]['strict']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['ent_type']['correct'] += 1

        else:

            # check for overlaps with any of the true entities
            for true in true_named_entities:

                # check for an exact boundary match but with a different e_type
                if true.start_offset <= pred.end_offset and pred.start_offset <= true.end_offset and true.e_type != pred.e_type:

                    # overall results
                    evaluation['strict']['incorrect'] += 1
                    evaluation['ent_type']['incorrect'] += 1

                    # aggregated by entity type results
                    evaluation_agg_entities_type[pred.e_type]['strict']['incorrect'] += 1
                    evaluation_agg_entities_type[pred.e_type]['ent_type']['incorrect'] += 1

                    true_which_overlapped_with_pred.append(true)
                    found_overlap = True
                    break

                # check for an overlap (not exact boundary match) with true entities
                elif pred.start_offset <= true.end_offset and true.start_offset <= pred.end_offset:
                    true_which_overlapped_with_pred.append(true)
                    if pred.e_type == true.e_type:  # overlaps with the same entity type
                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['correct'] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[pred.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[pred.e_type]['ent_type']['correct'] += 1

                        found_overlap = True
                        break

                    else:  # overlaps with a different entity type
                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['incorrect'] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[pred.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[pred.e_type]['ent_type']['incorrect'] += 1

                        found_overlap = True
                        break

            # count spurius (i.e., over-generated) entities
            if not found_overlap:
                # overall results
                evaluation['strict']['spurius'] += 1
                evaluation['ent_type']['spurius'] += 1

                # aggregated by entity type results
                evaluation_agg_entities_type[pred.e_type]['strict']['spurius'] += 1
                evaluation_agg_entities_type[pred.e_type]['ent_type']['spurius'] += 1

    # count missed entities
    for true in true_named_entities:
        if true in true_which_overlapped_with_pred:
            continue
        else:
            # overall results
            evaluation['strict']['missed'] += 1
            evaluation['ent_type']['missed'] += 1

            # for the agg. by e_type
            evaluation_agg_entities_type[true.e_type]['strict']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['ent_type']['missed'] += 1

    # Compute 'possible', 'actual', according to SemEval-2013 Task 9.1
    for eval_type in ['strict', 'ent_type']:
        correct = evaluation[eval_type]['correct']
        incorrect = evaluation[eval_type]['incorrect']
        partial = evaluation[eval_type]['partial']
        missed = evaluation[eval_type]['missed']
        spurius = evaluation[eval_type]['spurius']

        # possible: nr. annotations in the gold-standard which contribute to the final score
        evaluation[eval_type]['possible'] = correct + incorrect + partial + missed

        # actual: number of annotations produced by the NER system
        evaluation[eval_type]['actual'] = correct + incorrect + partial + spurius

        actual = evaluation[eval_type]['actual']
        possible = evaluation[eval_type]['possible']

        if eval_type == 'partial_matching':
            precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
            recall = (correct + 0.5 * partial) / possible if possible > 0 else 0
        else:
            precision = correct / actual if actual > 0 else 0
            recall = correct / possible if possible > 0 else 0

        evaluation[eval_type]['precision'] = precision
        evaluation[eval_type]['recall'] = recall

    return evaluation, evaluation_agg_entities_type


def compute_f1(test_sents_labels, y_pred):
    metrics_results = {'correct': 0, 'incorrect': 0, 'partial': 0,
                       'missed': 0, 'spurius': 0, 'possible': 0, 'actual': 0}

    # overall results
    results = {'strict': deepcopy(metrics_results),
               'ent_type': deepcopy(metrics_results)
               }

    # results aggregated by entity type
    evaluation_agg_entities_type = {e: deepcopy(results) for e in ['PNT', 'DUR', 'MNY', 'POH', 'ORG', 'NOH', 'DAT', 'TIM', 'LOC', 'PER']}

    for true_ents, pred_ents in zip(test_sents_labels, y_pred):
        # compute results for one message
        tmp_results, tmp_agg_results = compute_metrics(collect_named_entities(true_ents),
                                                       collect_named_entities(pred_ents))

        # aggregate overall results
        for eval_schema in results.keys():
            for metric in metrics_results.keys():
                results[eval_schema][metric] += tmp_results[eval_schema][metric]

        # aggregate results by entity type
        for e_type in ['PNT', 'DUR', 'MNY', 'POH', 'ORG', 'NOH', 'DAT', 'TIM', 'LOC', 'PER']:
            for eval_schema in tmp_agg_results[e_type]:
                for metric in tmp_agg_results[e_type][eval_schema]:
                    evaluation_agg_entities_type[e_type][eval_schema][metric] += tmp_agg_results[e_type][eval_schema][
                        metric]
    return results

if __name__=="__main__":

    with open('./result_kmou/score/testb.preds.txt', 'r', encoding="utf8") as fb:
        print("computing_F1")
        pred = []
        answer = []
        for line in fb.readlines():
            if len(line.strip()) == 0:
                continue
            _, tag, pred_tag = line.strip().split()
            pred.append(pred_tag)
            answer.append(tag)
        eval, _ = compute_metrics(collect_named_entities(answer), collect_named_entities(pred))
        print("eval_metrics: ", eval['strict'])
        print("F1-score: ", 2 * eval['strict']['precision'] * eval['strict']['recall'] / (
                    eval['strict']['precision'] + eval['strict']['recall']))


