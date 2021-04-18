import os
from  elasticsearch import Elasticsearch
import json
import re
es = Elasticsearch()
index = "ipa_dict"

def MatchPhonesToText(query_string):
    #query_string = "ɐbˈɒlɪʃɪ"
    doc={
        "_source": ["_source","word","phones","locales"], 
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

    resp = es.search(index= index, body=doc)
    return resp["hits"]["hits"]