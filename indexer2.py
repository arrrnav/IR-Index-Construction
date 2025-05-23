import os, json, bs4
from collections import defaultdict
from nltk.stem import PorterStemmer
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup, NavigableString, Tag


EXAMPLE_INDEX ='''
{
    "token": {
        doc_id_1: {
            "c": count,
            "s": score
        },
        doc_id_1: {
            "c": count,
            "s": score
        }
    },
    .
    .
    .
    "example": {
        0: {
            "c": 104,
            "s": 10
        },
        1: {
            "c": 83,
            "s": 8
        }
    }
}
'''

URLS_PATH = './analyst/ANALYST'

class Indexer:
    def __init__(self):
        # example index key, value:
        # 'gilbert': {'doc1': [1, 2], 'doc2': [3, 4]}
        # - gilbert appears in doc1 at positions 1 and 2, and in doc2 at positions 3 and 4
        self.inverted_index = defaultdict(lambda: defaultdict(list))
        self.url_to_id = defaultdict(int)
        self.id_to_url = defaultdict(str)
        self.next_available_id = 0
        self.important_tags = {
            "title": 15,
            "h1": 10,
            "h2": 6,
            "h3": 4,
            "strong": 2.5,
            "b": 2.5
            # default: 1
        }
        self.stemmer = PorterStemmer()
        self.position = 0

    def defrag_url(self, url):
        # Parse the URL and remove the fragment
        parsed_url = urlparse(url)
        url_without_fragment = parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
        return url_without_fragment
    
    def index(self, url, content):
        # Check if the URL is already indexed (for use in multiple URLS with same path but different fragments)
        url = self.defrag_url(url)
        if url in self.url_to_id:
            pass

        # Assign a new ID to the URL if it doesn't exist 
        doc_id = self.next_available_id
        self.next_available_id += 1

        # Update the URL to ID mapping
        self.url_to_id[url] = doc_id
        self.id_to_url[doc_id] = url

        soup = BeautifulSoup(content, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose() # remove script and style tags
        
        # position = 0 # track position of each token in the document

        def recursive_tokenize(element):
            if isinstance(element, NavigableString):
                # tokenize text
                text = re.sub(r'[^a-zA-Z0-9\s]', '', element)
                text = text.lower()
                unstemmized_tokens = text.split()
                for pre_token in unstemmized_tokens:
                    token = self.stemmer.stem(pre_token)
                    # increment the count and update score if the tag is more important than previously found
                    self.inverted_index[token][self.doc_id]["c"] += 1
                    self.inverted_index[token][self.doc_id]["s"] = max(self.important_tags.get(element.parent.name, 1), self.inverted_index[token][self.doc_id]["s"]) 

            elif isinstance(element, Tag):
                for child in element.children:
                    recursive_tokenize(child)

        # Call the recursive function on the soup object
        recursive_tokenize(soup)



        
        # # Go through the important tags and extract text
        # for tag in self.important_tags:
        #     for element in soup.find_all(tag):
        #         text = element.get_text()
        #         # Tokenize text
        #         text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        #         text = text.lower()
        #         self.words_in_tags[tag] += text + ' '

        # print(f"url: {url}")
        # # print(self.words_in_tags)
        # text = soup.get_text()
        # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # self.words = text.split()
        # print(words)
        # print(f"doc_id: {doc_id}")
        
        # for position, word in enumerate(self.words):
        #     word = word.lower()
        #     stemmed_word = self.stemmer.stem(word)
        #     self.inverted_index[stemmed_word][doc_id].append(position)

class Search:
    def __init__(self, index_path):
        self.inverted_index = index_path

    def get_importance_factor(self, word):
        pass
        # importance based on TF-IDF score and tag importance

        #TF-IDF score = TF(term, document) * IDF(term)
        # TF(term, document) = (number of times term appears in document) / (total number of terms in document)
        # IDF(term) = log(total number of documents / number of documents containing term)

        # importance factor = TF-IDF score * tag importance