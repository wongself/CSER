import jinja2
import json
from sklearn.metrics import precision_recall_fscore_support as prfs
import torch
from transformers import BertTokenizer
from typing import List, Tuple, Dict

from cser import util
from cser.dataset import Document, Dataset, EntityType
from cser.reader import JsonReader


class Evaluator:
    def __init__(
        self, dataset: Dataset, reader: JsonReader,
        text_encoder: BertTokenizer, overlapping: bool,
        predictions_path: str, examples_path: str,
        epoch: int, dataset_label: str): # noqa

        self._text_encoder = text_encoder
        self._reader = reader
        self._dataset = dataset
        self._overlapping = overlapping

        self._epoch = epoch
        self._dataset_label = dataset_label

        self._predictions_path = predictions_path

        self._examples_path = examples_path

        # entities
        self._gt_entities = []  # ground truth
        self._pred_entities = []  # prediction

        self._pseudo_entity_type = EntityType('Entity', 1, 'Entity', 'Entity')  # for span only evaluation

        self._convert_gt(self._dataset.documents)

    def eval_batch(self, batch_entity_clf: torch.tensor, batch: dict):
        batch_size = batch_entity_clf.shape[0]

        # get maximum activation (index of predicted entity type)
        batch_entity_types = batch_entity_clf.argmax(dim=-1)
        # apply entity sample mask
        batch_entity_types *= batch['entity_sample_masks'].long()

        for i in range(batch_size):
            # get model predictions for sample
            entity_types = batch_entity_types[i]

            # get entities that are not classified as 'None'
            valid_entity_indices = entity_types.nonzero().view(-1)
            valid_entity_types = entity_types[valid_entity_indices]
            valid_entity_spans = batch['entity_spans'][i][valid_entity_indices]
            valid_entity_scores = torch.gather(
                batch_entity_clf[i][valid_entity_indices], 1,
                valid_entity_types.unsqueeze(1)).view(-1)

            sample_pred_entities = self._convert_pred_entities(
                valid_entity_types,
                valid_entity_spans,
                valid_entity_scores)

            if not self._overlapping:
                sample_pred_entities = self._remove_overlapping(sample_pred_entities)

            self._pred_entities.append(sample_pred_entities)

    def store_predictions(self):
        predictions = []

        for i, doc in enumerate(self._dataset.documents):
            tokens = doc.tokens
            pred_entities = self._pred_entities[i]

            # convert entities
            converted_entities = []
            for entity in pred_entities:
                entity_span = entity[:2]
                span_tokens = util.get_span_tokens(tokens, entity_span)
                entity_type = entity[2].identifier
                converted_entity = dict(type=entity_type, start=span_tokens[0].index, end=span_tokens[-1].index + 1)
                converted_entities.append(converted_entity)
            converted_entities = sorted(converted_entities, key=lambda e: e['start'])

            doc_predictions = dict(
                tokens=[t.phrase for t in tokens],
                entities=converted_entities)
            predictions.append(doc_predictions)

        # store as json
        label, epoch = self._dataset_label, self._epoch
        with open(self._predictions_path % (label, epoch), 'w') as predictions_file:
            json.dump(predictions, predictions_file)

    def store_examples(self, template_path: str):
        entity_examples = []

        for i, doc in enumerate(self._dataset.documents):
            # entities
            entity_example = self._convert_example(
                doc, self._gt_entities[i], self._pred_entities[i],
                include_entity_types=True, to_html=self._entity_to_html)
            entity_examples.append(entity_example)

        label, epoch = self._dataset_label, self._epoch

        # entities
        self._store_examples(
            entity_examples,
            file_path=self._examples_path % ('entities', label, epoch),
            template_path=template_path)

        self._store_examples(
            sorted(entity_examples, key=lambda k: k['length']),
            file_path=self._examples_path % ('entities_sorted', label, epoch),
            template_path=template_path)

    def compute_scores(self):
        print("Evaluation")

        print("")
        print("--- Entities (Named Entity Recognition ---")
        print("An entity is considered correct if the entity span and type is predicted correctly")
        print("")
        gt, pred = self._convert_by_setting(self._gt_entities, self._pred_entities, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=True)

        return ner_eval

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            gt_entities = doc.entities

            # convert ground truth entities for precision/recall/f1 evaluation
            sample_gt_entities = [entity.as_tuple() for entity in gt_entities]

            if not self._overlapping:
                sample_gt_entities = self._remove_overlapping(sample_gt_entities)

            self._gt_entities.append(sample_gt_entities)

    def _convert_pred_entities(
        self, pred_types: torch.tensor,
        pred_spans: torch.tensor, pred_scores: torch.tensor): # noqa

        converted_preds = []

        for i in range(pred_types.shape[0]):
            label_idx = pred_types[i].item()
            entity_type = self._reader.get_entity_type(label_idx)

            start, end = pred_spans[i].tolist()
            score = pred_scores[i].item()

            converted_pred = (start, end, entity_type, score)
            converted_preds.append(converted_pred)

        return converted_preds

    def _remove_overlapping(self, entities):
        non_overlapping_entities = []

        for entity in entities:
            if not self._is_overlapping(entity, entities):
                non_overlapping_entities.append(entity)

        return non_overlapping_entities

    def _is_overlapping(self, e1, entities):
        for e2 in entities:
            if self._check_overlap(e1, e2):
                return True

        return False

    def _check_overlap(self, e1, e2):
        if e1 == e2 or e1[1] <= e2[0] or e2[1] <= e1[0]:
            return False
        else:
            return True

    def _convert_by_setting(
        self, gt: List[List[Tuple]], pred: List[List[Tuple]],
        include_entity_types: bool = True,
        include_score: bool = False): # noqa

        assert len(gt) == len(pred)

        # either include or remove entity types based on setting
        def convert(t):
            if not include_entity_types:
                # remove entity type and score for evaluation
                if type(t[0]) == int:  # entity
                    c = [t[0], t[1], self._pseudo_entity_type]
                else:
                    c = [
                        (t[0][0], t[0][1], self._pseudo_entity_type),
                        (t[1][0], t[1][1], self._pseudo_entity_type),
                        t[2]]
            else:
                c = list(t[:3])

            if include_score and len(t) > 3:
                # include prediction scores
                c.append(t[3])

            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])

        return converted_gt, converted_pred

    def _score(
        self, gt: List[List[Tuple]], pred: List[List[Tuple]],
        print_results: bool = False): # noqa
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)

            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(0)

        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t.index for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('TYPE', 'PRECISION', 'RECALL', 'F1', 'AMOUNT')

        row_fmt = "%12s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t.short_name))
            results.append('\n')

        results.append('\n')

        # micro
        results.append(row_fmt % self._get_row(micro, 'micro'))
        results.append('\n')

        # macro
        results.append(row_fmt % self._get_row(macro, 'macro'))

        results_str = ''.join(results)
        print(results_str)

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    def _convert_example(
        self, doc: Document, gt: List[Tuple], pred: List[Tuple],
        include_entity_types: bool, to_html): # noqa
        encoding = doc.encoding

        gt, pred = self._convert_by_setting([gt], [pred], include_entity_types=include_entity_types, include_score=True)
        gt, pred = gt[0], pred[0]

        # get micro precision/recall/f1 scores
        if gt or pred:
            pred_s = [p[:3] for p in pred]  # remove score
            precision, recall, f1 = self._score([gt], [pred_s])[:3]
        else:
            # corner case: no ground truth and no predictions
            precision, recall, f1 = [100] * 3

        scores = [p[-1] for p in pred]
        pred = [p[:-1] for p in pred]
        union = set(gt + pred)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for s in union:
            type_verbose = s[2].verbose_name

            if s in gt:
                if s in pred:
                    score = scores[pred.index(s)]
                    tp.append((to_html(s, encoding), type_verbose, score))
                else:
                    fn.append((to_html(s, encoding), type_verbose, -1))
            else:
                score = scores[pred.index(s)]
                fp.append((to_html(s, encoding), type_verbose, score))

        tp = sorted(tp, key=lambda p: p[-1], reverse=True)
        fp = sorted(fp, key=lambda p: p[-1], reverse=True)

        text = self._prettify(self._text_encoder.decode(encoding))
        return dict(text=text, tp=tp, fn=fn, fp=fp, precision=precision, recall=recall, f1=f1, length=len(doc.tokens))

    def _entity_to_html(self, entity: Tuple, encoding: List[int]):
        start, end = entity[:2]
        entity_type = entity[2].verbose_name

        tag_start = ' <span class="entity">'
        tag_start += '<span class="type">%s</span>' % entity_type

        ctx_before = self._text_encoder.decode(encoding[:start])
        e1 = self._text_encoder.decode(encoding[start:end])
        ctx_after = self._text_encoder.decode(encoding[end:])

        html = ctx_before + tag_start + e1 + '</span> ' + ctx_after
        html = self._prettify(html)

        return html

    def _prettify(self, text: str):
        text = text.replace('_start_', '').replace('_classify_', '').replace('<unk>', '').replace('⁇', '')
        text = text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        return text

    def _store_examples(self, examples: List[Dict], file_path: str, template_path: str):
        # read template
        with open(str(template_path)) as f:
            template = jinja2.Template(f.read())

        # write to disc
        template.stream(examples=examples).dump(file_path)
