import os

trigrams = open(os.environ["NEURALSEARCH_TRIGRAMS_PATH"])
trigrams.readline()

NO_OF_TRIGRAMS = 0

for line in trigrams:
    NO_OF_TRIGRAMS += 1