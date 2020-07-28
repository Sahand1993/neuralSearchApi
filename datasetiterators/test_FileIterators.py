import json
from typing import Dict, List
from unittest import TestCase

import numpy as np

from datasetiterators.fileiterators import SquadFileIterator, WikiQAFileIterator, NaturalQuestionsFileIterator, \
    ReutersFileIterator, QuoraFileIterator, ConfluenceFileIterator


def relevant(article1, article2):
    sharedKeys = set(article1.keys()).intersection(set(article2.keys()))
    sharedTagKeys = list(filter(lambda key: key not in ["articleId", "id", "queryArticleNGramIndices", "queryArticleWordIndices", "relevantId", ""], sharedKeys))
    for tagKey in sharedTagKeys:
        if tagKey in ["gcrim", "gdef", "gdip", "gdis", "gedu", "gent", "genv", "gfas", "ghea", "gjob", "gmil", "gobit", "godd", "gpol", "gpro", "grel", "gsci", "gspo", "gtour", "gvio", "gvote", "gwea" ,"gwelf", "meur"]:
            if (article1[tagKey] and article2[tagKey]):
                return True
        else:
            return True


class TestFileIterators(TestCase):

    idTotrigram: Dict[int, str] = {}
    trigramToId: Dict[str, int] = {}

    squadQuestions: Dict[str, Dict] = {}

    wikiqaQuestions: Dict[str, Dict] = {}

    nqQuestions: Dict[str, Dict] = {}

    rcv1ArticleIdToArticle: Dict[int, Dict] = {}
    rcv1IdToArticleId: Dict[int, int] = {}
    rcv1IdToRelevantId: Dict[int, int] = {}

    quoraIdToPair: Dict[int, Dict] = {}
    quoraQuestionIdToDuplicates: Dict[int, int] = {}

    confluenceIdToDoc: Dict[str, List[str]] = {}


    def containsAllNGrams(self, tokens: List[str], nGramIdToFreq: Dict[int, int]) -> bool:
        tokens = list(map(lambda token: "^" + token + "$", tokens))

        tokensNGramIdToFreq: Dict[int, int] = {}
        for token in tokens:
            for i in range(len(token) - 2):
                trigram = token[i:i + 3]
                trigramId = self.trigramToId[trigram]
                trigramId = trigramId
                try:
                    tokensNGramIdToFreq[trigramId] += 1
                except KeyError:
                    tokensNGramIdToFreq[trigramId] = np.float64(1)

        if len(tokensNGramIdToFreq) != len(nGramIdToFreq):
            return False

        for key in tokensNGramIdToFreq:
            if tokensNGramIdToFreq[key] != nGramIdToFreq[key]:
                return False

        return True


    def loadTrigrams(self):
        trigramFile = open("/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/trigrams.txt")
        trigramFile.readline()
        for line in trigramFile:
            trigram, _id = line.split()
            self.idTotrigram[int(_id)] = trigram
            self.trigramToId[trigram] = int(_id)


    def loadSquadQuestions(self):
        squadFile = open("/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/squad/mid.json")
        for line in squadFile:
            obj = json.loads(line)
            self.squadQuestions[obj["questionId"]] = obj


    def loadWikiQAQuestions(self):
        wikiqaFile = open("/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/wikiqa/mid.json")
        for i, line in enumerate(wikiqaFile):
            obj = json.loads(line)
            self.wikiqaQuestions[i] = obj


    def loadNqQuestions(self):
        nqFile = open("/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/nq/mid.json")
        for line in nqFile:
            obj = json.loads(line)
            self.nqQuestions[obj["id"]] = obj


    def loadRcv1Questions(self):
        rcv1File = open("/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/rcv1/mid.json")
        for line in rcv1File:
            obj = json.loads(line)
            self.rcv1ArticleIdToArticle[int(obj["article_id"])] = obj

        rcv1PreprocessedFile = open("/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/rcv1/total.json")
        for line in rcv1PreprocessedFile:
            obj = json.loads(line)
            self.rcv1IdToArticleId[obj["id"]] = obj["articleId"]
            self.rcv1IdToRelevantId[obj["id"]] = obj["relevantId"]


    def loadQuoraQuestions(self):
        quoraFile = open("/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/quora/mid.json")
        for line in quoraFile:
            obj = json.loads(line)
            if obj["isDuplicate"]:
                self.quoraIdToPair[obj["id"]] = obj
                try:
                    self.quoraQuestionIdToDuplicates[obj["question1Id"]].append(obj["question2Id"])
                except KeyError:
                    self.quoraQuestionIdToDuplicates[obj["question1Id"]] = [obj["question2Id"]]
                try:
                    self.quoraQuestionIdToDuplicates[obj["question2Id"]].append(obj["question1Id"])
                except KeyError:
                    self.quoraQuestionIdToDuplicates[obj["question2Id"]] = [obj["question1Id"]]


    def loadConfluenceDocs(self):
        confluenceFile = open("/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/confluence/mid.json")
        for line in confluenceFile:
            obj = json.loads(line)
            self.confluenceIdToDoc[obj["id"]] = obj["titleTokens"]


    def get_nonzeroindex_to_value(self, dense_vector) -> Dict[int, int]:
        nGramIndices, = np.nonzero(dense_vector)
        return {nGramIndex: dense_vector[nGramIndex] for nGramIndex in nGramIndices}


    def setUp(self):
        self.loadTrigrams()


    def test_squad_get_samples(self):
        self.loadSquadQuestions()
        iterator = SquadFileIterator("/Users/sahandzarrinkoub/School/year5/thesis/clustertraining/cluster/clustertraining/datasets/squad/data.csv",
                                     "/Users/sahandzarrinkoub/School/year5/thesis/clustertraining/cluster/clustertraining/datasets/squad/train.csv",
                                     batch_size = 1,
                                     no_of_irrelevant_samples=2,
                                     encodingType="NGRAM",
                                     dense=True,
                                     shuffle=True)

        for i in range(1000):
            batch = iterator.__next__()
            _id = batch.get_ids()[0]

            q = batch.get_q_dense()[0]
            qNGrams: Dict[int, int] = self.get_nonzeroindex_to_value(q)
            questionTokens: List[str] = self.squadQuestions[_id]["questionTokens"]
            self.assertTrue(self.containsAllNGrams(questionTokens, qNGrams))

            title = batch.get_relevant_dense()[0]
            titleNGrams: Dict[int, int] = self.get_nonzeroindex_to_value(title)
            titleTokens: List[str] = self.squadQuestions[_id]["titleTokens"]
            if not self.containsAllNGrams(titleTokens, titleNGrams):
                print(_id, "failed")
            self.assertTrue(self.containsAllNGrams(titleTokens, titleNGrams))


    def test_wikiqa_get_samples(self):
        self.loadWikiQAQuestions()
        iterator = WikiQAFileIterator("/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/wikiqa/data.csv",
                                      "/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/wikiqa/train.csv",
                                      batch_size=1,
                                      no_of_irrelevant_samples=2,
                                      encodingType="NGRAM",
                                      dense=True,
                                      shuffle=True)


        for i in range(2000):
            batch = iterator.__next__()
            _id = batch.get_ids()[0]

            q = batch.get_q_dense()[0]
            qNGrams: Dict[int, int] = self.get_nonzeroindex_to_value(q)
            questionTokens: List[str] = self.wikiqaQuestions[int(_id)]["questionTokens"]
            self.assertTrue(self.containsAllNGrams(questionTokens, qNGrams))

            title = batch.get_relevant_dense()[0]
            titleNGrams: Dict[int, int] = self.get_nonzeroindex_to_value(title)
            titleTokens: List[str] = self.wikiqaQuestions[int(_id)]["titleTokens"]
            self.assertTrue(self.containsAllNGrams(titleTokens, titleNGrams))


    def test_nq_get_samples_title(self):
        self.loadNqQuestions()
        iterator = NaturalQuestionsFileIterator("/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/nq/data.csv",
                                                batch_size=1,
                                                no_of_irrelevant_samples=2,
                                                encodingType="NGRAM",
                                                dense=True,
                                                shuffle=True,
                                                title=True)

        for i in range(1000):
            try:
                batch = iterator.__next__()
            except StopIteration:
                break
            _id = batch.get_ids()[0]

            q = batch.get_q_dense()[0]
            qNGrams: Dict[int, int] = self.get_nonzeroindex_to_value(q)
            questionTokens: List[str] = self.nqQuestions[int(_id)]["questionTokens"]
            self.assertTrue(self.containsAllNGrams(questionTokens, qNGrams))

            title = batch.get_relevant_dense()[0]
            titleNGrams: Dict[int, int] = self.get_nonzeroindex_to_value(title)
            titleTokens: List[str] = self.nqQuestions[int(_id)]["titleTokens"]
            self.assertTrue(self.containsAllNGrams(titleTokens, titleNGrams))


    def test_nq_get_samples_document(self):
        self.loadNqQuestions()
        iterator = NaturalQuestionsFileIterator(
            "/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/nq/data.csv",
            batch_size=1,
            no_of_irrelevant_samples=2,
            encodingType="NGRAM",
            dense=True,
            shuffle=True,
            title=False)

        for i in range(1000):
            try:
                batch = iterator.__next__()
            except StopIteration:
                break
            _id = batch.get_ids()[0]

            question = batch.get_q_dense()[0]
            questionNGrams: Dict[int, int] = self.get_nonzeroindex_to_value(question)
            questionTokens: List[str] = self.nqQuestions[int(_id)]["questionTokens"]
            self.assertTrue(self.containsAllNGrams(questionTokens, questionNGrams))

            document = batch.get_relevant_dense()[0]
            documentNGrams: Dict[int, int] = self.get_nonzeroindex_to_value(document)
            documentTokens: List[str] = self.nqQuestions[int(_id)]["documentTokens"]
            self.assertTrue(self.containsAllNGrams(documentTokens, documentNGrams))


    def test_rcv1_get_samples(self):
        self.loadRcv1Questions()
        iterator = ReutersFileIterator("/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/rcv1/total.json",
                                       "/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/rcv1/total.json",
                                       batch_size=1,
                                       no_of_irrelevant_samples=2,
                                       encodingType="NGRAM",
                                       dense=True)


        for i in range(1000):
            batch = iterator.__next__()
            _id = batch.get_ids()[0]

            question = batch.get_q_dense()[0]
            questionNGrams: Dict[int, int] = self.get_nonzeroindex_to_value(question)
            questionArticleId = self.rcv1IdToArticleId[_id]
            questionArticle = self.rcv1ArticleIdToArticle[questionArticleId]
            questionTokens: List[str] = questionArticle["tokens"]
            self.assertTrue(self.containsAllNGrams(questionTokens, questionNGrams))

            relevantArticle = batch.get_relevant_dense()[0]
            relevantArticleNGrams: Dict[int, int] = self.get_nonzeroindex_to_value(relevantArticle)
            relevantId = self.rcv1IdToRelevantId[_id]
            relevantArticleId = self.rcv1IdToArticleId[relevantId]
            relevantArticleTokens: List[str] = self.rcv1ArticleIdToArticle[relevantArticleId]["tokens"]
            self.assertTrue(self.containsAllNGrams(relevantArticleTokens, relevantArticleNGrams))

            questionArticle = self.rcv1ArticleIdToArticle[questionArticleId]
            relevantArticle = self.rcv1ArticleIdToArticle[relevantArticleId]
            self.assertTrue(relevant(questionArticle, relevantArticle))


    def test_quora_get_samples(self):
        self.loadQuoraQuestions()
        iterator = QuoraFileIterator(
            "/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/quora/data.csv",
            "/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/quora/data.csv",
            batch_size=1,
            no_of_irrelevant_samples=2,
            encodingType="NGRAM",
            dense=True,
            shuffle=True)

        for i in range(1000):
            try:
                batch = iterator.__next__()
            except StopIteration:
                break
            _id = batch.get_ids()[0]

            question1 = batch.get_q_dense()[0]
            question1NGrams: Dict[int, int] = self.get_nonzeroindex_to_value(question1)
            question1Tokens: List[str] = self.quoraIdToPair[_id]["question1_tokens"]
            self.assertTrue(self.containsAllNGrams(question1Tokens, question1NGrams))

            question2 = batch.get_relevant_dense()[0]
            question2NGrams: Dict[int, int] = self.get_nonzeroindex_to_value(question2)
            question2Tokens: List[str] = self.quoraIdToPair[_id]["question2_tokens"]
            self.assertTrue(self.containsAllNGrams(question2Tokens, question2NGrams))

            question1Id = self.quoraIdToPair[_id]["question1Id"]
            question2Id = self.quoraIdToPair[_id]["question2Id"]
            self.assertTrue(question1Id in self.quoraQuestionIdToDuplicates[question2Id])


    def test_confluence_get_samples(self):
        self.loadConfluenceDocs()
        iterator = ConfluenceFileIterator(
            "/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/confluence/data.csv",
            dense=True
        )

        for i in range(5000):
            try:
                batch = iterator.__next__()
            except StopIteration:
                break
            _id = batch.get_ids()[0]

            docNGrams = batch.get_relevant_dense()[0]
            docNGrams = self.get_nonzeroindex_to_value(docNGrams)
            docTokens = self.confluenceIdToDoc[_id]
            self.assertTrue(self.containsAllNGrams(docTokens, docNGrams))