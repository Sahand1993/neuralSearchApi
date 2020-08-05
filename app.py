import json
import random
from pymongo import MongoClient
from flask import Flask, request
import requests
from flask_cors import CORS
from typing import List, Dict, Tuple
from numpy import ndarray
import numpy as np
import os

from requests.auth import HTTPBasicAuth

from datasetiterators.fileiterators import ConfluenceFileIterator
from dssm.model_dense_ngram import *
from helpers.helpers import cosine_similarity

app = Flask(__name__)
CORS(app)

mongoclient = MongoClient()
mongodb = mongoclient.neuralsearchdataset # neuralsearchdataset is the name of the mongodb database
mongodataset = mongodb.queries # queries is the name of the collection in mongodb, where queries and documents are stored

NEURAL_SEARCH_CUTOFF = 20
CONFLUENCE_INDICES_PATH = os.environ["CONFLUENCE_INDICES_FILE"]
CONFLUENCE_TEXT_PATH = os.environ["CONFLUENCE_TEXT_FILE"]
TRIGRAMS_PATH = os.environ["NEURALSEARCH_TRIGRAMS_PATH"]


CONFLUENCE_BASE_URL = "https://confluence.braincourt.net/rest/api/search"
CONFLUENCE_USER = os.environ["confluence_username"]
CONFLUENCE_PASS = os.environ["confluence_password"]

trigramN = 3
docs: Dict[str, Dict] = {}
trigramIndices: Dict[str, int] = {}

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

def index_dataset():
    """
    Indexes the confluence data set.
    docs = {id: {}, id: {}}
    :return:
    """
    confluence_indices = ConfluenceFileIterator(CONFLUENCE_INDICES_PATH, batch_size=1, dense=True)
    confluence_text_file = open(CONFLUENCE_TEXT_PATH)
    for line in confluence_text_file:
        example = json.loads(line)
        if example["id"] != "-":
            docs[example["id"]] = example


    for batch in confluence_indices:
        doc_vec, = sess.run([y], feed_dict={x: batch.get_relevant_dense()})
        docs[batch.get_ids()[0]]["representation"] = doc_vec


def load_trigram_mappings():
    trigram_file = open(TRIGRAMS_PATH)
    trigram_file.readline()

    for line in trigram_file:
        try:
            trigram, _id = line.split()
            trigramIndices[trigram] = int(_id)
        except:
            pass



def random_list_of_mock_results() -> List[Dict]:
    results = []
    max_no_of_results = random.choice(range(20))

    for i in range(1, max_no_of_results + 1):
        results.append(
            {"id": "doc" + str(i),
             "title": "Document " + str(i),
             "contentExcerpt": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."})

    return results


def to_trigram_count_vector(s: str) -> ndarray:
    trigram_count_vector = np.zeros(NO_OF_TRIGRAMS)
    s = "^" + s + "$"
    for i in range(len(s) - 2):
        trigram = s[i:i + 3]
        trigram_idx = trigramIndices.get(trigram)
        if trigram_idx != None:
            trigram_count_vector[trigram_idx] += 1

    return np.reshape(trigram_count_vector, (1, len(trigram_count_vector)))


def get_representation(trigram_count_vector: ndarray):
    representation, = sess.run([y], feed_dict={x: trigram_count_vector})
    return representation


def find_closest_docs(representation: ndarray, cutoff = 10) -> List[Tuple[int, float]]:
    scored_docs: List[Tuple[int, float]] = []

    for _id, doc in docs.items():
        scored_docs.append((_id, cosine_similarity(doc["representation"], representation)))

    return sorted(scored_docs, key=lambda item: item[1], reverse=True)[0:cutoff]


def ranked_docs_to_search_results(scored_docs: List[Tuple[int, float]]) -> List[Dict]:
    results = []
    for _id, score in scored_docs:
        doc_dict = {"score": score}
        for key, value in docs[_id].items():
            if key != "representation":
                doc_dict[key] = value
        results.append(doc_dict)
    return results


def neural_search(query: str) -> List[Dict]:
    query_trigram_count_vector: ndarray = to_trigram_count_vector(query)
    query_representation: ndarray = get_representation(query_trigram_count_vector)
    ranked_docs: List[Tuple[int, float]] = find_closest_docs(query_representation, NEURAL_SEARCH_CUTOFF)

    results: List[Dict] = ranked_docs_to_search_results(ranked_docs)
    return results


def confluence_results_to_search_results(results: List[Dict]):
    return list(map(
        lambda result:
        {
            "id": result["content"]["id"],
            "title": result["content"]["title"],
            "webUi": result["content"]["_links"]["webui"]
        },
        results))


def default_confluence_search(query: str) -> List[Dict]:
    response = requests.get(
        CONFLUENCE_BASE_URL + '?cql=extranet.privacy.granted=true AND siteSearch ~ "' + query + '" AND type in ("space"%2C"user"%2C"page"%2C"blogpost"%2C"attachment"%2C"net.seibertmedia.plugin.confluence.microblog%3AmicropostContent"%2C"com.atlassian.confluence.plugins.confluence-questions%3Aquestion"%2C"com.atlassian.confluence.plugins.confluence-questions%3Aanswer")&start=0&limit=20&excerpt=highlight&expand=space.icon&includeArchivedSpaces=false&src=next.ui.search',
        auth=HTTPBasicAuth(CONFLUENCE_USER, CONFLUENCE_PASS))

    results = json.loads(response.text)["results"]
    return confluence_results_to_search_results(results)


index_dataset()
print("indexed dataset")
load_trigram_mappings()
print("loaded trigram mappings")


# We call the API like: localhost:5000/neuralSearch/
@app.route("/neuralSearch")
def get_neural_search():
    query = request.args.get("query")
    if query == None:
        return ""
    else:
        return json.dumps({
            "results": neural_search(query),
            "query": query,
        }, indent=4)


@app.route("/defaultConfluence")
def dummy_results():
    query = request.args.get("query")
    if query == None:
        return ""
    return json.dumps({
        "results": default_confluence_search(query),
        "query": query,
    }, indent=4)


@app.route("/")
def test():
    return "Hello World"


@app.route("/savedata")
def save_datapoint():
    """Saves a (query, document) pair for future training of the search model"""
    query = request.args.get("query")
    document_id= request.args.get("documentId")
    document_title = request.args.get("documentTitle")
    example = {"query": query, "document_id": document_id, "document_title": document_title}
    mongodataset.insert_one(example)
    print("inserted {} into database".format(example))
    return "Success"
