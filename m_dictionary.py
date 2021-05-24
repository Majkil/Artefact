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
        phone_array = resp["hits"]["hits"][0]["_source"]["phones"][0].split(',')[0]
        phone_array = phone_array.replace(
            '\u200d', '').replace("/", '').replace("ˈ", '').replace("ˌ",'')
        return phone_array
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
    doc = {
        "query": {
            "fuzzy": {
                "phones": {
                    "value": query_string,
                    "fuzziness": "AUTO",
                    "max_expansions": 50,
                    "prefix_length": 0,
                    "transpositions": True,
                    "rewrite": "constant_score"
                }
            }
        }
    }

    resp = es.search(index=index, body=doc)
    return resp["hits"]["hits"]
