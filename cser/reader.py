from abc import abstractmethod, ABC
from collections import OrderedDict
import json
from tqdm import tqdm
from transformers import BertTokenizer
from typing import Iterable, List

from cser.dataset import Dataset, EntityType, Entity, Document


class BaseReader(ABC):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None, max_span_size: int = None):
        types = json.load(open(types_path), object_pairs_hook=OrderedDict)

        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()

        # Add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # Specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type

        self._neg_entity_count = neg_entity_count
        self._max_span_size = max_span_size

        self._datasets = dict()

        self._tokenizer = tokenizer

        self._vocabulary_size = tokenizer.vocab_size
        self._context_size = -1

    @abstractmethod
    def read(self, datasets):
        pass

    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        return self._idx2entity_type[idx]

    def _get_context_size(self, datasets: Iterable[Dataset]):
        sizes = []

        for dataset in datasets:
            for doc in dataset.documents:
                sizes.append(len(doc.encoding))

        context_size = max(sizes)
        return context_size

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def context_size(self):
        return self._context_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonReader(BaseReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None, max_span_size: int = None):
        super().__init__(types_path, tokenizer, neg_entity_count, max_span_size)

    def read(self, dataset_label, dataset_path, dataset_mode):
        dataset = Dataset(
            dataset_label, self._entity_types,
            self._neg_entity_count, self._max_span_size, dataset_mode)
        self._parse_dataset(dataset_path, dataset)
        self._datasets[dataset_label] = dataset

        self._context_size = self._get_context_size(self._datasets.values())

    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, doc, dataset) -> Document:
        jtokens = doc['tokens']
        jentities = doc['entities']

        # parse tokens
        tokens, encodings = self._parse_tokens(jtokens, dataset)
        # parse entity mentions
        entities = self._parse_entities(jentities, tokens, dataset)
        # create document
        dataset.create_document(tokens, entities, encodings)

    def _parse_tokens(self, jtokens, dataset):
        tokens = []
        encodings = [self._tokenizer.convert_tokens_to_ids('[CLS]')]

        # Parse tokens
        for i, token_phrase in enumerate(jtokens):
            token_encoding = self._tokenizer.encode(
                token_phrase, add_special_tokens=False)
            span_start = len(encodings)
            span_end = len(encodings) + len(token_encoding)

            t = dataset.create_token(i, span_start, span_end, token_phrase)
            tokens.append(t)
            encodings += token_encoding

        encodings += [self._tokenizer.convert_tokens_to_ids('[SEP]')]

        return tokens, encodings

    def _parse_entities(self, jentities, doc_tokens, dataset) -> List[Entity]:
        entities = []

        for entity_idx, jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']

            # Create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            e = dataset.create_entity(entity_type, tokens, phrase)
            entities.append(e)

        return entities
