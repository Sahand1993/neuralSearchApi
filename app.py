import json
from flask import Flask, request
from flask_cors import CORS
import random
from typing import List, Dict

app = Flask(__name__)
CORS(app)


def random_list_of_results() -> List[Dict]:
    results = []
    # max = random.choice(range(20))
    #
    # for i in range(1, max + 1):
    #     results.append(
    #         {"id": "doc" + str(i),
    #          "name": "Document " + str(i),
    #          "contentExcerpt": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."})

    return results

# We call the API like: localhost:5000/neuralSearch/
@app.route("/neuralSearch")
def dummy_results():
    query = request.args.get("query")

    return json.dumps({
        "results": random_list_of_results(),
        "query": query,
    })