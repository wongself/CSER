from collections import OrderedDict
from torch.utils.data import Dataset as TorchDataset
from typing import List

from cser import sampling


class EntityType:
    def __init__(self, identifier, index, short_name, verbose_name):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, EntityType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)


class Token:
    def __init__(self, tid: int, index: int, span_start: int, span_end: int, phrase: str):
        self._tid = tid
        self._index = index
        self._span_start = span_start
        self._span_end = span_end
        self._phrase = phrase

    @property
    def tid(self):
        return self._tid

    @property
    def index(self):
        return self._index

    @property
    def span_start(self):
        return self._span_start

    @property
    def span_end(self):
        return self._span_end

    @property
    def span(self):
        return self._span_start, self._span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Token):
            return self._tid == other._tid
        return False

    def __hash__(self):
        return hash(self._tid)

    def __str__(self):
        return self._phrase

    def __repr__(self):
        return self._phrase


class Span:
    def __init__(self, tokens):
        self._tokens = tokens

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    def __getitem__(self, s):
        if isinstance(s, slice):
            return Span(self._tokens[s.start:s.stop:s.step])
        else:
            return self._tokens[s]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)


class Entity:
    def __init__(self, eid: int, entity_type: EntityType, tokens: List[Token], phrase: str):
        self._eid = eid
        self._entity_type = entity_type
        self._tokens = tokens
        self._phrase = phrase

    def as_tuple(self):
        return self.span_start, self.span_end, self._entity_type

    @property
    def eid(self):
        return self._eid

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def tokens(self):
        return Span(self._tokens)

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self._eid == other._eid
        return False

    def __hash__(self):
        return hash(self._eid)

    def __str__(self):
        return self._phrase


class Document:
    def __init__(self, did: int, tokens: List[Token], entities: List[Entity], encoding: List[int]):
        self._did = did
        self._tokens = tokens
        self._entities = entities
        self._encoding = encoding

    @property
    def did(self):
        return self._did

    @property
    def entities(self):
        return self._entities

    @property
    def tokens(self):
        return Span(self._tokens)

    @property
    def encoding(self):
        return self._encoding

    @encoding.setter
    def encoding(self, value):
        self._encoding = value

    def __eq__(self, other):
        if isinstance(other, Document):
            return self._did == other._did
        return False

    def __hash__(self):
        return hash(self._did)


class Dataset(TorchDataset):
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'

    def __init__(self, label, entity_types, neg_entity_count, max_span_size, dataset_mode):
        self._label = label
        self._entity_types = entity_types
        self._neg_entity_count = neg_entity_count
        self._max_span_size = max_span_size
        self._mode = dataset_mode

        # self._entities = OrderedDict()
        self._documents = OrderedDict()

        # Current ids
        self._tid = 0
        self._eid = 0
        self._rid = 0
        self._did = 0

    def create_token(self, idx, span_start, span_end, phrase) -> Token:
        t = Token(self._tid, idx, span_start, span_end, phrase)
        self._tid += 1
        return t

    def create_entity(self, entity_type, tokens, phrase) -> Entity:
        e = Entity(self._eid, entity_type, tokens, phrase)
        # self._entities[self._eid] = e
        self._eid += 1
        return e

    def create_document(self, tokens, entities, encoding) -> Document:
        d = Document(self._did, tokens, entities, encoding)
        self._documents[self._did] = d
        self._did += 1
        # return d

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index: int):
        doc = self._documents[index]

        if self._mode == Dataset.TRAIN_MODE:
            return sampling.create_train_sample(
                doc, self._neg_entity_count, self._max_span_size)
        elif self._mode == Dataset.EVAL_MODE:
            return sampling.create_eval_sample(
                doc, self._max_span_size)
        else:
            raise Exception(
                "Dataset mode not in ['train', 'eval'].")

    @property
    def label(self):
        return self._label

    @property
    def tid(self):
        return self._tid

    @property
    def eid(self):
        return self._eid

    @property
    def rid(self):
        return self._rid

    @property
    def did(self):
        return self._did

    @property
    def documents(self):
        return list(self._documents.values())
