import os
from  elasticsearch import Elasticsearch
import json
import re

es = Elasticsearch()

def get_phonemes_for_word(query_string):
    index = "ipa_dict"
    doc = {
      "query": {
        "match": {
          "word.keyword": query_string.lower()
            }
          }
        }
    resp = es.search(index= index, body=doc)
    if resp["hits"]["hits"]:
      return resp["hits"]["hits"][0]["_source"]["phones"]
    else:
      return  " "

def get_phonemes_for_sentence(sentence):
  words = sentence.split(' ')
  phones =[]
  for word in words:
    phones.append(get_phonemes_for_word(word))
  return phones


#print( get_phonemes_for_sentence('WATCH THE SAVAGES OUTSIDE SAID ROBERT'))