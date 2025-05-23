import os, json, bs4
from collections import defaultdict
from nltk.stem import PorterStemmer
import re
from urllib.parse import urlparse


EXAMPLE_INDEX ='''
{
    "token1": {
        "doc1_id": [
            [position, weight],
            [position, weight],
            [position, weight]
        ],
        "doc4_id": [
            [position, weight],
            [position, weight]
        ]
    },
    "example": {
        "0": [
            [24, 3],
            [502, 1]
        ],
        "1": [
            [2, 5]
        ]
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
            "title": 10,
            "h1": 7,
            "h2": 5,
            "h3": 3,
            "strong": 1.5,
            "b": 0.5
        }
        self.stemmer = PorterStemmer()
        # self.words_in_tags = defaultdict(str)
        self.words = []
        self.docs = 0
        '''
        {
            "token1": {
                "doc1_id": [
                    [position, weight],
                    [position, weight],
                    [position, weight]
                ],
                "doc4_id": [
                    [position, weight],
                    [position, weight]
                ]
            },
            "example": {
                "0": [
                    [24, 3],
                    [502, 1]
                ],
                "1": [
                    [2, 5]
                ]
            }
        }
        '''

    def defrag_url(self, url):
        # Parse the URL and remove the fragment
        parsed_url = urlparse(url)
        url_without_fragment = parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
        return url_without_fragment
    
    def index(self, url, content):
        if url in self.url_to_id:
            pass

        # Parse the URL and remove the fragment

        # Assign a new ID to the URL if it doesn't exist 
        doc_id = self.next_available_id
        self.next_available_id += 1

        # Update the URL to ID mapping
        self.url_to_id[url] = doc_id
        self.id_to_url[doc_id] = url

        soup = bs4.BeautifulSoup(content, 'html.parser')
        word
        
        # Go through the important tags and extract text
        for tag in self.important_tags:
            for element in soup.find_all(tag):
                text = element.get_text()
                # Tokenize text
                text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
                text = text.lower()
                self.words_in_tags[tag] += text + ' '

        print(f"url: {url}")
        # print(self.words_in_tags)
        text = soup.get_text()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        self.words = text.split()
        # print(words)
        # print(f"doc_id: {doc_id}")
        
        for position, word in enumerate(self.words):
            word = word.lower()
            stemmed_word = self.stemmer.stem(word)
            self.inverted_index[stemmed_word][doc_id].append(position)