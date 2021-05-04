import os
from elasticsearch import Elasticsearch
import json
import re

es = Elasticsearch()
index = "ipa_dict"

def get_phonemes_for_word(query_string):
    doc = {
        "query": {
            "match": {
                "word.keyword": query_string.lower()
            }
        }
    }
    resp = es.search(index=index, body=doc)
    if resp["hits"]["hits"]:
        return resp["hits"]["hits"][0]["_source"]["phones"]
    else:
        return " "

def get_phonemes_for_sentence(sentence):
    words = sentence.split(' ')
    phones = []
    for word in words:
        phones.append(get_phonemes_for_word(word))
    return phones

# print( get_phonemes_for_sentence('WATCH THE SAVAGES OUTSIDE SAID ROBERT'))

def MatchPhonesToText(query_string):
    # query_string = "ɐbˈɒlɪʃɪ"
    doc = {
        "_source": ["_source", "word", "phones", "locales"],
        "query": {
            "dis_max": {
                "tie_breaker": 0.7,
                "boost": 1.2,
                "queries": [
                    {
                        "match": {
                            "phones": query_string
                        }
                    },
                    {
                        "fuzzy": {
                            "phones": {
                                "value": query_string
                            }
                        }
                    }
                ]
            }
        }
    }

    resp = es.search(index=index, body=doc)
    return resp["hits"]["hits"]
