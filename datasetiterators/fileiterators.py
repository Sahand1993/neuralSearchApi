from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Set
import random
import json
import numpy as np

from datasetiterators.batchiterators import DataPoint, DataPointFactory, readCsvLines, sample_numbers, CSV_SEPARATOR, DataPointBatch


class FileIterator(ABC):

    def __init__(self, batch_size = 5, no_of_irrelevant_samples = 4, encodingType ="NGRAM", csvPath=None, dense=True):
        if csvPath:
            self._file = open(csvPath)
            self._file.readline()
        self._batch_size = batch_size
        self._no_of_irrelevant_samples = no_of_irrelevant_samples
        self._traversal_order: List[int] = None
        self._current_idx = 0
        self._encodingType = encodingType
        self._dense = dense


    def __next__(self) -> DataPointBatch:
        indices: List[int] = self._traversal_order[self._current_idx: self._current_idx + self._batch_size]
        self._current_idx += self._batch_size
        if self._dense:
            if len(indices) == 0:
                raise StopIteration
        else:
            if len(indices) < self._batch_size:
                raise StopIteration

        return DataPointBatch(self.get_samples(indices), self._no_of_irrelevant_samples)


    @abstractmethod
    def get_samples(self, ids: List[int]) -> List[DataPoint]:
        pass


    @abstractmethod
    def get_irrelevants(self, q_id: int) -> List:
        pass


    @abstractmethod
    def restart(self):
        pass


    def __len__(self):
        if self._dense:
            rest = len(self._traversal_order) % self._batch_size
            return len(self._traversal_order) // self._batch_size + (1 if rest else 0)
        else:
            return len(self._traversal_order) // self._batch_size


    def __iter__(self):
        return self


    def getNoOfDataPoints(self):
        if self._dense:
            return len(self._traversal_order)
        else:
            return (len(self._traversal_order) // self._batch_size) * self._batch_size


class QuoraFileIterator(FileIterator):

    def __init__(self, totalPath, csvPath:str, batch_size = 5, no_of_irrelevant_samples = 4, encodingType="NGRAM", dense=True, shuffle=True):
        super().__init__(batch_size,
                         no_of_irrelevant_samples,
                         encodingType,
                         csvPath,
                         dense)
        self._totalPath = totalPath
        self._questionIdToDuplicates: Dict[int, List[int]] = dict()
        self._pairIdToQuestionPair: Dict[int, Tuple[str, str]] = dict()
        self._questionIdToIndices: Dict[int, str] = dict()
        self._duplicate_pair_ids: List[int] = []
        self.index_file()
        self._questionIds = list(self._questionIdToDuplicates.keys())
        self._total_samples = len(self._duplicate_pair_ids)
        self._traversal_order = self._get_traversal_order(csvPath) #[_id for _id in self._duplicate_pair_ids] # TODO Implement with ability to provide training or val set but still index all, like in reuters
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self._traversal_order)


    def restart(self):
        self._current_idx = 0
        if self.shuffle:
            random.shuffle(self._traversal_order)


    def get_samples(self, pairIds: List[int]) -> List[DataPoint]:
        samples: List[DataPoint] = []
        for pairId in pairIds:
            q1_id: int = self._pairIdToQuestionPair[pairId][0]
            q2_id: int = self._pairIdToQuestionPair[pairId][1]
            question1_indices: str = self._questionIdToIndices[q1_id]
            question2_indices: str = self._questionIdToIndices[q2_id]
            dataPoint: DataPoint = DataPointFactory.fromNGramsData(pairId, str(q1_id), str(q2_id), question1_indices, question2_indices, self.get_irrelevants(pairId))
            samples.append(dataPoint)

        return samples


    def get_irrelevants(self, q_id: int) -> List[str]:
        irrelevant_ngrams: List[str] = []
        while len(irrelevant_ngrams) < self._no_of_irrelevant_samples:
            new_q_id: int = random.choice(self._questionIds)
            new_q_duplicates: List[str] = self._questionIdToDuplicates[new_q_id]
            if q_id not in new_q_duplicates:
                irrelevant_ngrams.append(self._questionIdToIndices[new_q_id])

        return irrelevant_ngrams


    def index_file(self):
        file = open(self._totalPath)
        file.readline()
        for line in file.readlines():
            csvValues = line.split(CSV_SEPARATOR)
            pair_id: int = int(csvValues[0])
            q1_id: int = int(csvValues[1])
            q2_id: int = int(csvValues[2])
            if self._encodingType == "NGRAM":
                q1_tokens: List[str] = csvValues[3]
                q2_tokens: List[str] = csvValues[4]
            elif self._encodingType == "WORD":
                q1_tokens: List[str] = csvValues[5]
                q2_tokens: List[str] = csvValues[6]
            else:
                raise ValueError("Wrong value of self._encoding_type")

            is_duplicate = csvValues[7]
            if is_duplicate:
                self._duplicate_pair_ids.append(pair_id)
                try:
                    q1_duplicates: List[int] = self._questionIdToDuplicates[q1_id]
                    if q2_id not in q1_duplicates:
                        q1_duplicates.append(q2_id)
                except KeyError:
                    self._questionIdToDuplicates[q1_id] = [q2_id]

                try:
                    q2_duplicates = self._questionIdToDuplicates[q2_id]
                    if q1_id not in q2_duplicates:
                        q2_duplicates.append(q1_id)
                except KeyError:
                    self._questionIdToDuplicates[q2_id] = [q1_id]

            self._pairIdToQuestionPair[pair_id] = (q1_id, q2_id)

            self._questionIdToIndices[q1_id] = q1_tokens
            self._questionIdToIndices[q2_id] = q2_tokens

    def _get_traversal_order(self, csvPath):
        file = open(csvPath)
        return self._getPairIdsFrom(file)

    def _getPairIdsFrom(self, file) -> List[int]:
        file.readline()
        return list(map(lambda csvLine: self._getPairIdFrom(csvLine), file))

    def _getPairIdFrom(self, csvLine: str) -> int:
        return int(csvLine.split(";")[0])


class WikiQABM25Iterator():
    def __init__(self, totalPath:str, partPath:str, no_of_irrelevant_samples = 4):
        self._no_of_irrelevant_samples = no_of_irrelevant_samples
        self._all_examples: Dict[str, Dict] = self.getExamples(totalPath)
        self._allQIds: List[str] = list(self._all_examples.keys())
        self._partIds: List[str] = self.getIds(partPath)
        self._traversal_order = list(range(len(self._partIds)))
        self._current_idx = 0

        # TODO: Make sure incorrect samples are sampled from entire file
        # TODO: Make sure correct samples are only sampled from relevant dataset part

    def getExamples(self, path: str) -> Dict[str, Dict]:
        examples: Dict[str, Dict] = {}
        file = open(path)
        for line in file:
            jsonObj = json.loads(line)
            _id = jsonObj["questionId"] # TODO
            if _id in examples:
                raise ValueError("There are duplicate questions in WikiQA")
            else:
                examples[_id] = jsonObj

        return examples


    def getIds(self, path:str):
        _ids = []
        file = open(path)
        for line in file:
            _id = json.loads(line)["questionId"]
            _ids.append(_id)

        return _ids


    def __next__(self) -> Dict:
        try:
            example = self._all_examples[self._partIds[self._current_idx]]
        except IndexError:
            raise StopIteration

        irrelevants = []
        seenIndices = set()
        while len(irrelevants) < self._no_of_irrelevant_samples:
            index = np.random.randint(0, len(self._all_examples))
            if index != self._current_idx and index not in seenIndices:
                irrelevants.append(self._all_examples[self._allQIds[index]]["titleTokens"])
            seenIndices.add(index)

        self._current_idx += 1

        return {"id": example["questionId"],
               "question_tokens": example["questionTokens"],
                "relevant_tokens": example["titleTokens"],
                "irrelevant_tokens": irrelevants}


    def __iter__(self):
        return self


    def __len__(self):
        return len(self._partIds)


    def restart(self):
        self._current_idx = 0


class NaturalQuestionsBM25Iterator():
    def __init__(self, totalPath: str, partPath:str, no_of_irrelevant_samples = 4, title=True):
        self._no_of_irrelevant_samples = no_of_irrelevant_samples
        self._title = title
        self._examples: Dict[int, Dict] = dict(map(self.get_important_fields, open(totalPath)))
        self._partIds: List[int] = self.getIds(partPath)
        self._current_idx = 0


    def get_important_fields(self, jsonl) -> Tuple[int, Dict]:
        jsonobj = json.loads(jsonl)
        if self._title:
            document_tokens: List[int] = jsonobj["titleTokens"]
        else:
            document_tokens: List[int] = jsonobj["documentTokens"]
        question_tokens: List[str] = jsonobj["questionTokens"]
        example_id = jsonobj["exampleId"]
        return example_id, {"document_tokens": document_tokens, "question_tokens": question_tokens, "id": example_id}


    def getIds(self, path:str):
        ids: List[int] = []
        file = open(path)
        for line in file:
            jsonObj = json.loads(line)
            ids.append(jsonObj["exampleId"])

        return ids

    def __next__(self):
        try:
            example = self._examples[self._partIds[self._current_idx]]
        except IndexError:
            raise StopIteration

        irrelevants = []
        seenIndices = set()
        while len(irrelevants) < self._no_of_irrelevant_samples:
            index = np.random.randint(0, len(self._partIds))
            if index != self._current_idx and index not in seenIndices:
                irrelevants.append(self._examples[self._partIds[index]]["document_tokens"])
            seenIndices.add(index)

        self._current_idx += 1

        return {"id": example["id"],
               "question_tokens": example["question_tokens"],
                "relevant_tokens": example["document_tokens"],
                "irrelevant_tokens": irrelevants}


    def __iter__(self):
        return self


    def __len__(self):
        return len(self._partIds)

    def restart(self):
        self._current_idx = 0


class NaturalQuestionsFileIterator(FileIterator):
    def __init__(self, filePath: str, batch_size = 5, no_of_irrelevant_samples = 4, encodingType="NGRAM", dense=True, shuffle=True, title=False):
        """

        :param filePath:
        :param batch_size:
        :param no_of_irrelevant_samples:
        :param encodingType: Can be NGRAM or WORD. Determines which document representation will be used.
        """
        super().__init__(batch_size, no_of_irrelevant_samples, encodingType, filePath, dense)
        self._title = title
        self._csvValues: List[List[str]] = readCsvLines(self._file)
        self._traversal_order = list(range(len(self._csvValues)))
        example_ids = [line[1] for line in self._csvValues]
        self._traversal_order_to_example_id: Dict[int, str] = { order : example_id for order, example_id in zip(self._traversal_order, example_ids)}
        self._shuffle = shuffle
        if self._shuffle:
            random.shuffle(self._traversal_order)


    def get_example_id(self, traversal_order: int) -> str:
        return self._traversal_order_to_example_id[traversal_order]

    def get_samples(self, indices: List[int]) -> List[DataPoint]:
        samples: List[DataPoint] = []
        for idx in indices:
            dataPoint: DataPoint = self.getDataPoint(idx)
            samples.append(dataPoint)

        return samples


    def getDataPoint(self, idx: int) -> DataPoint:
        csvRow = self._csvValues[idx]
        _id = csvRow[0]
        irrelevantDocuments: List[str] = self.get_irrelevants(idx)
        question, document = self.get_query_doc_pair(csvRow)
        if self._encodingType == "NGRAM":
            return DataPointFactory.fromNGramsData(_id, None, None, question, document, irrelevantDocuments)
        elif self._encodingType == "WORD":
            return DataPointFactory.fromWordIndicesData(_id, None, None, question, document, irrelevantDocuments)
        else:
            raise ValueError("Incorrect value of self._encodingType")


    def get_sample(self, idx: int) -> Tuple[str, str]:
        csvRow = self._csvValues[idx]
        return self.get_query_doc_pair(csvRow)


    def get_query_doc_pair(self, csvRow):

        if self._encodingType == "WORD":
            questionWordIndices = csvRow[6]
            if self._title:
                return questionWordIndices, csvRow[3]
            else:
                return questionWordIndices, csvRow[7]
        elif self._encodingType == "NGRAM":
            questionNGramIndices = csvRow[4]
            if self._title:
                return questionNGramIndices, csvRow[2]
            else:
                return questionNGramIndices, csvRow[5]
        else:
            raise ValueError("Incorrect value of self._encodingType")

    def get_irrelevants(self, idx: int) -> List[str]:
        irrelevantIndices: List[int] \
            = list(map(lambda x: x - 1, sample_numbers(len(self._csvValues), self._no_of_irrelevant_samples, idx)))
        irrelevantDocuments: List[str] = []
        for idx in irrelevantIndices:
            query, document = self.get_sample(idx)
            if (document == ''):
                print("{} has empty document".format(query))
            irrelevantDocuments.append(document)
        return irrelevantDocuments


    def restart(self):
        self._current_idx = 0
        if self._shuffle:
            random.shuffle(self._traversal_order)


class ReutersFileIterator(FileIterator):
    def __init__(self, totalPath: str, dataSetPartPathJson: str, batch_size = 5, no_of_irrelevant_samples = 4, encodingType="NGRAM", dense=True):
        """

        :param set: either "train" or "val"
        :param noOfSamples:
        :param batch_size:
        :param no_of_irrelevant_samples:
        :param encodingType:
        """
        super().__init__(batch_size, no_of_irrelevant_samples, encodingType, dense=dense)
        self._totalPath = totalPath
        self._idToArticle: Dict[int, dict] = dict()
        self._tagToId: Dict[str, Set] = dict()
        self.NON_TAG_KEYS = ["queryArticleNGramIndices", "queryArticleWordIndices", "relevantId", "articleId", "id"]
        self._index()
        self._traversal_order: List[int] = self._getArticleIdsFromFile(dataSetPartPathJson)
        random.shuffle(self._traversal_order)
        self.TAG_KEYS = [
                "c11",
                "c12",
                "c13",
                "c14",
                "c15",
                "c16",
                "c17",
                "c18",
                "c21",
                "c22",
                "c23",
                "c24",
                "c31",
                "c32",
                "c33",
                "c34",
                "c41",
                "c42",
                "e11",
                "e12",
                "e13",
                "e14",
                "e21",
                "e31",
                "e41",
                "e51",
                "e61",
                "e71",
                "g11",
                "g12",
                "g13",
                "g14",
                "g15",
                "gcrim",
                "gdef",
                "gdip",
                "gdis",
                "gedu",
                "gent",
                "genv",
                "gfas",
                "ghea",
                "gjob",
                "gmil",
                "gobit",
                "godd",
                "gpol",
                "gpro",
                "grel",
                "gsci",
                "gspo",
                "gtour",
                "gvio",
                "gvote",
                "gwea",
                "gwelf",
                "m11",
                "m12",
                "m13",
                "m14",
                "meur"
            ]

    def get_samples(self, ids: List[int]) -> List[DataPoint]:
        samples: List[DataPoint] = []
        for _id in ids:
            queryArticle = self._idToArticle[_id]
            relevantId = queryArticle["relevantId"]
            relevantArticle = self._idToArticle[relevantId]
            irrelevants: List[str] = self.get_irrelevants(_id)
            if self._encodingType == "NGRAM":
                samples.append(DataPointFactory.fromNGramsData(_id,
                                                               str(_id),
                                                               relevantId,
                                                               queryArticle["queryArticleNGramIndices"],
                                                               relevantArticle["queryArticleNGramIndices"],
                                                               irrelevants))
            elif self._encodingType == "WORD":
                samples.append(DataPointFactory.fromWordIndicesData(_id,
                                                                    str(_id),
                                                                    relevantId,
                                                                    queryArticle["queryArticleWordIndices"],
                                                                    relevantArticle["queryArticleWordIndices"],
                                                                    irrelevants))
            else:
                raise ValueError("Incorrect value of self._encodingType")

        return samples


    def restart(self):
        self._current_idx = 0
        random.shuffle(self._traversal_order)


    def get_irrelevants(self, _id):
        irrelevants: List[str] = []
        while len(irrelevants) < self._no_of_irrelevant_samples:
            irrelevantId: int = random.choice(self._traversal_order)
            if irrelevantId == _id:
                continue
            if not self.isRelevant(irrelevantId, _id):
                if self._encodingType == "NGRAM":
                    irrelevants.append(self._idToArticle[irrelevantId]["queryArticleNGramIndices"])
                else:
                    irrelevants.append(self._idToArticle[irrelevantId]["queryArticleWordIndices"])

        return irrelevants


    def isRelevant(self, id1: int, id2: int) -> bool:
        article1: Dict = self._idToArticle[id1]
        article2: Dict = self._idToArticle[id2]
        article1BoolTagVector: List[bool] = self.getBooleanTagVector(article1)
        article2BoolTagVector: List[bool] = self.getBooleanTagVector(article2)
        andVector: List[bool] = list(map(lambda tags: tags[0] and tags[1], zip(article1BoolTagVector, article2BoolTagVector)))
        for andResult in andVector:
            if andResult:
                return True
        return False


    def getBooleanTagVector(self, article: Dict) -> List[bool]:
        boolVec: List[bool] = []
        for tagKey in self.TAG_KEYS:
            try:
                boolVec.append(bool(article[tagKey]))
            except KeyError:
                boolVec.append(False)
        return boolVec


    def _index(self):
        file = open(self._totalPath)
        for i, line in enumerate(file.readlines()): # TODO remove enumerate later
            article: Dict = json.loads(line)
            _id = article["id"]
            article.pop("id")
            self._idToArticle[_id] = article

            for tag in self._getTags(article):
                try:
                    self._tagToId[tag].add(_id)
                except KeyError:
                    self._tagToId[tag] = {_id}


    def _getTags(self, article: Dict) -> List[str]:
        tagKeys = filter(lambda key: key not in self.NON_TAG_KEYS, article)
        tags = list(filter(lambda key: article[key] is not False, tagKeys))

        return tags

    def _getArticleIdsFromFile(self, dataSetPathJson) -> List[int]:
        f = open(dataSetPathJson)
        ids = list()
        for line in f:
            jsonObj = json.loads(line)
            ids.append(jsonObj["id"])

        return ids


class SquadFileIterator(FileIterator):
    def __init__(self, totalPath, datasetPartPath, batch_size = 5, no_of_irrelevant_samples = 4, encodingType="NGRAM", dense=True, shuffle=True):
        super().__init__(batch_size,
                         no_of_irrelevant_samples,
                         encodingType,
                         datasetPartPath,
                         dense)
        self._shuffle = shuffle
        self._totalPath = totalPath
        self._articleIdToQuestions: Dict[str, List[Dict]] = {}
        self._ids: List[str] = []
        self._questions: Dict[str, Dict[str, str]] = {}
        self.index_file()
        self._traversal_order: List[int] = list(range(len(self._ids)))
        if shuffle:
            random.shuffle(self._traversal_order)


    def restart(self):
        self._current_idx = 0
        if self._shuffle:
            random.shuffle(self._traversal_order)


    def get_samples(self, indices: List[int]) -> List[DataPoint]:
        samples: List[DataPoint] = []
        for idx in indices:
            dataPoint = self.get_data_point(idx)
            samples.append(dataPoint)

        return samples


    def get_data_point(self, idx: int) -> DataPoint:
        _id = self._ids[idx]
        question = self._questions[_id]
        questionId = question["question_id"]
        docId = question["title_id"]
        questionIndices = question["question_indices"]
        titleIndices = question["title_indices"]
        irrelevants = self.get_irrelevants(_id)
        if self._encodingType == "NGRAM":
            return DataPointFactory.fromNGramsData(questionId, questionId, docId, questionIndices, titleIndices, irrelevants)
        elif self._encodingType == "WORD":
            return DataPointFactory.fromWordIndicesData(questionId, questionId, docId, questionIndices, titleIndices, irrelevants)

    def get_irrelevants(self, rowNumber: str) -> List[str]:
        titleId = self._questions[rowNumber]["title_id"]
        irrelevants = []
        while len(irrelevants) < 4:
            randomId = random.choice(self._ids)
            randomQuestion = self._questions[randomId]
            randomTitleId = randomQuestion["title_id"]
            if randomTitleId != titleId:
                irrelevants.append(randomQuestion["title_indices"])

        return irrelevants

    def index_file(self):
        total = open(self._totalPath)
        total.readline()
        for line in total:
            csvValues = line.split(";")
            _id = csvValues[0]
            questionId = csvValues[1]
            titleId = csvValues[2]
            if self._encodingType == "NGRAM":
                question_indices = csvValues[3]
                title_indices = csvValues[5]
            elif self._encodingType == "WORD":
                question_indices = csvValues[4]
                title_indices = csvValues[6]
            else:
                raise RuntimeError("Wrong value of self._encodingType.")


            self._questions[_id] = {"question_indices": question_indices, "title_indices": title_indices, "question_id": questionId, "title_id": titleId}
            if (questionId in self._articleIdToQuestions):
                self._articleIdToQuestions[questionId].append({"question_id": questionId,
                                                              "question_indices": question_indices,
                                                              "title_indices": title_indices})

        for line in self._file:
            csvValues = line.split(";")
            _id = csvValues[0]
            self._ids.append(_id)


class WikiQAFileIterator(FileIterator):
    def __init__(self, totalPath, datasetPartPath, batch_size = 5, no_of_irrelevant_samples = 4, encodingType="NGRAM", dense=True, shuffle=True):
        super().__init__(batch_size,
                         no_of_irrelevant_samples,
                         encodingType,
                         datasetPartPath,
                         dense)
        self._shuffle = shuffle
        self._totalPath = totalPath
        self._docIdToQuestions: Dict[str, List[Dict]] = {}
        self._all_ids: List[str] = []
        self._datasetPartIds: List[str] = []
        self._idToQuestion: Dict[str, Dict[str, str]] = {}
        self.index_file()
        self._traversal_order: List[str] = self._datasetPartIds
        if shuffle:
            random.shuffle(self._traversal_order)


    def restart(self):
        self._current_idx = 0
        if self._shuffle:
            random.shuffle(self._traversal_order)


    def get_samples(self, indices: List[int]) -> List[DataPoint]:
        samples: List[DataPoint] = []
        for idx in indices:
            dataPoint = self.get_data_point(idx)
            samples.append(dataPoint)

        return samples

    def get_data_point(self, id) -> DataPoint:
        question = self._idToQuestion[id]
        _id: str = question["id"]
        questionId = question["question_id"]
        questionIndices = question["question_indices"]
        titleIndices = question["title_indices"]
        documentId = question["document_id"]
        irrelevants = self.get_irrelevants(id)
        if self._encodingType == "NGRAM":
            return DataPointFactory.fromNGramsData(_id, questionId, documentId, questionIndices, titleIndices, irrelevants)
        elif self._encodingType == "WORD":
            return DataPointFactory.fromWordIndicesData(_id, questionId, documentId, questionIndices, titleIndices, irrelevants)


    def get_irrelevants(self, q_id: int) -> List[str]:
        question = self._idToQuestion[q_id]
        docId = question["document_id"]
        irrelevants: List[str] = []
        while len(irrelevants) < self._no_of_irrelevant_samples:
            random_id = random.choice(self._all_ids)
            random_question = self._idToQuestion[random_id]
            if random_question["document_id"] != docId:
                irrelevants.append(random_question["title_indices"])

        return irrelevants


    def index_file(self):
        total = open(self._totalPath)
        total.readline()
        for line in total:
            csvValues = line.split(";")
            _id, questionId, docId, questionNGramIndices, questionWordIndices, titleNGramIndices, titleWordIndices = csvValues
            self._all_ids.append(_id)
            if self._encodingType == "NGRAM":
                self._idToQuestion[_id] = {"id": _id, "question_id": questionId, "document_id": docId, "question_indices": questionNGramIndices, "title_indices": titleNGramIndices}
            elif self._encodingType == "WORD":
                self._idToQuestion[_id] = {"id": _id, "question_id": questionId, "document_id": docId,
                                           "question_indices": questionWordIndices, "title_indices": titleWordIndices}

        for line in self._file:
            csvValues = line.split(";")
            _id, questionId, docId, questionNGramIndices, questionWordIndices, titleNGramIndices, titleWordIndices = csvValues
            self._datasetPartIds.append(_id)


class ConfluenceFileIterator(FileIterator):
    def __init__(self, filePath, batch_size = 1, dense=True):
        super().__init__(batch_size,
                         0,
                         "NGRAM",
                         filePath,
                         dense)
        self._filePath = filePath
        self._idToExample: Dict[str, Dict[str, str]] = {}
        self._traversal_order: List[str] = []
        self._index_file()


    def _index_file(self) -> None:
        for line in self._file:
            _id, ngrams = line.split(";")
            self._idToExample[_id] = {"document_indices": ngrams}
            self._traversal_order.append(_id)


    def get_samples(self, ids: List[int]) -> List[DataPoint]:
        samples: List[DataPoint] = []
        for _id in ids:
            data_point = self._get_datapoint(_id)
            samples.append(data_point)
        return samples


    def _get_datapoint(self, _id) -> DataPoint:
        return DataPointFactory.fromNGramsData(_id, None, _id, None, self._idToExample[_id]["document_indices"], [])


    def get_irrelevants(self, q_id: int):
        pass


    def restart(self):
        self._current_idx = 0