import numpy as np
from itertools import chain
from scipy.sparse import csr_matrix

from typing import List, Iterable, Set
import random
from dssm.config import *

NO_OF_INDICES = NO_OF_TRIGRAMS

DEFAULT_IRRELEVANT_SAMPLES = 4
DEFAULT_BATCH_SIZE = 5

CSV_SEPARATOR = ";"


def to_dense(indices: List[str]) -> List[int]:
    array = np.zeros((1, NO_OF_INDICES))
    for freq_index in indices:
        freq, index = freq_index.split(" ")
        array[0][int(index)] = int(freq)
    return array


def readCsvLines(file) -> List[List[str]]:
    return list(map(lambda line: line.split(CSV_SEPARATOR),
        file.readlines()))


def sample_numbers(end: int, size: int, exclude: int) -> List[int]:
    """
    Samples uniformly from 1 to end (inclusive) and excludes exclude.
    :param end:
    :param size:
    :param exclude:
    :return:
    """
    numbers: List[int] = np.random.choice(end, replace=False, size=size + 1) + 1
    numbers: List[int] = filter(lambda number: number != exclude, numbers)
    numbers: List[int] = list(numbers)[:size]
    return numbers


#def filterWords(wordIndices: Iterable) -> np.ndarray:
#    return np.array(list(filter(lambda wordIndex: wordIndex in FILTERED_WORD_INDICES, wordIndices)))


class DataPoint():

    def __init__(self, _id: str, qId: str, docId: str, query_indices: str, relevant_indices: str, irrelevant_indices: np.ndarray):
        """
        :param query_indices: vector of integers
        :param relevant_indices: as above
        :param irrelevant_indices: matrix of integers, shape is [no_of_irrelevants, None]
        """
        self._id = _id
        self._qId = qId
        self._docId = docId

        if query_indices:
            self._query_indices: np.ndarray = np.array(to_dense(query_indices.split(",")))
        else:
            self._query_indices = None

        self._relevant_ngrams: np.ndarray = np.array(to_dense(relevant_indices.split(",")))
        self._irrelevant_ngrams: np.ndarray = irrelevant_indices


    def get_id(self):
        return self._id


    def get_qId(self):
        return self._qId


    def get_docId(self):
        return self._docId


    def get_query_ngrams(self) -> np.ndarray:
        return self._query_indices


    def get_relevant_ngrams(self) -> np.ndarray:
        return self._relevant_ngrams


    def get_irrelevant_ngrams(self) -> np.ndarray:
        return self._irrelevant_ngrams


class DataPointFactory():

    @staticmethod
    def fromNGramsData(_id: str, qId: str, docId: str, query_ngrams: str, relevant_ngrams: str, irrelevant_ngrams: List[str]) -> DataPoint:
        return DataPoint(
            _id,
            qId,
            docId,
            query_ngrams,
            relevant_ngrams,
            np.array([to_dense(ngrams.split(",")) for ngrams in irrelevant_ngrams])
        )


    @staticmethod
    def fromWordIndicesData(_id: str, qId: str, docId: str, queryWordIndices: str, relevantWordIndices: str, irrelevantWordIndices: List[str]) -> DataPoint:
        return DataPoint(
            _id,
            qId,
            docId,
            queryWordIndices,
            relevantWordIndices,
            np.array([to_dense(ngrams.split(",")) for ngrams in irrelevantWordIndices])
        )


class DataPointBatch():

    def __init__(self, data_points: List[DataPoint], no_of_irrelevant_samples = 4):
        self.data_points = data_points
        self._no_of_irrelevant_samples = no_of_irrelevant_samples


    def get_q_indices(self) -> np.ndarray:
        return self.create_batch(list(map(lambda data_point: data_point.get_query_ngrams(), self.data_points)))


    def get_q_dense(self) -> np.ndarray:
        #return self.create_batch_dense(list(map(lambda data_point: data_point.get_query_ngrams(), self.data_points)))
        return np.vstack([q for q in map(lambda data_point: data_point.get_query_ngrams(), self.data_points)])


    def get_relevant_indices(self) -> np.ndarray:
        return self.create_batch(list(map(lambda data_point: data_point.get_relevant_ngrams(), self.data_points)))


    def get_relevant_dense(self) -> np.ndarray:
        #return self.create_batch_dense(list(map(lambda data_point: data_point.get_relevant_ngrams(), self.data_points)))
        return np.vstack([rel for rel in map(lambda data_point: data_point.get_relevant_ngrams(), self.data_points)])


    def get_irrelevant_indices(self) -> List[np.ndarray]:
        irrelevants_batches = [] # [irr1_batch, irr2_batch, irr3_batch]
        for i in range(self._no_of_irrelevant_samples):
            irrelevants_batches.append(self.create_batch(list(map(lambda data_point: data_point.get_irrelevant_ngrams()[i], self.data_points))))

        return irrelevants_batches


    def get_irrelevant_dense(self) -> List[np.ndarray]:
        irrelevants_batches = []
        for i in range(self._no_of_irrelevant_samples):
            #batch = self.create_batch_dense(
            #    list(map(lambda data_point: data_point.get_irrelevant_ngrams()[i], self.data_points)))
            batch = np.vstack([irr for irr in map(lambda data_point: data_point.get_irrelevant_ngrams()[i], self.data_points)])
            irrelevants_batches.append(batch)

        return irrelevants_batches


    def create_batch(self, dataPointIndices: List[np.ndarray]) -> np.ndarray:
        indices = np.empty((0, 2), np.int64)
        for i, indices in enumerate(dataPointIndices):
            new_indices = np.array([[i, index] for index in indices])
            indices = np.concatenate((indices, new_indices))
        return indices


    def create_batch_dense(self, batchIndices: List[np.ndarray]) -> np.ndarray:
        data = []
        row_ind = []
        col_ind = []
        for row, indices in enumerate(batchIndices):
            for index in indices:
                row_ind.append(row)
                col_ind.append(index)
                data.append(1)
        return csr_matrix(
            (data,
            (row_ind, col_ind)), shape=(len(batchIndices), NO_OF_INDICES)).toarray()


    def get_ids(self) -> List[str]:
        return list(map(lambda data_point: data_point.get_id(), self.data_points))


    def get_qIds(self) -> List[str]:
        return list(map(lambda data_point: data_point.get_qId(), self.data_points))


    def get_docIds(self) -> List[str]:
        return list(map(lambda data_point: data_point.get_docId(), self.data_points))


class RandomBatchIterator():
    """
    Randomly calls the __next__() methods of the given iterators.
    """
    def __init__(self, *args):
        """

        :param args: iterators to uniformly sample from.
        """
        self.iterators = list(args)
        self._turns = []
        for iterator in self.iterators:
            self._turns += list((iterator for _ in range(len(iterator))))

        random.shuffle(self._turns)


    def __iter__(self):
        return self


    def __next__(self):
        if self.iterators:
            try:
                iterator = self._turns.pop()
                return iterator.__next__()
            except IndexError:
                raise StopIteration
        else:
            raise StopIteration


    def restart(self):
        for iterator in self.iterators:
            iterator.restart()

        self._turns = []
        for iterator in self.iterators:
            self._turns += list((iterator for _ in range(len(iterator))))

        random.shuffle(self._turns)


    def getNoOfDataPoints(self):
        return sum(map(lambda it: it.getNoOfDataPoints(), self.iterators))


    def __len__(self):
        return sum(map(lambda iterator: len(iterator), self.iterators))
