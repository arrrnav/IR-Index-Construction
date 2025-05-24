import os, json, bs4
from collections import defaultdict
from nltk.stem import PorterStemmer
import re
from urllib.parse import urlparse, urlunparse
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
        self.inverted_index = defaultdict(lambda: defaultdict(lambda: {"c": 0, "s": 0}))
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
        # self.position = 0

    def defrag_url(self, url):
        # Parse the URL and remove the fragment
        parsed_url = urlparse(url)
        url_without_fragment = urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            parsed_url.params,
            parsed_url.query,
            ''
        ))
        # url_without_fragment = parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
        return url_without_fragment
    
    def index(self, url, content):
        # Check if the URL is already indexed (for use in multiple URLS with same path but different fragments)
        if url in self.url_to_id:
            return

        # for debugging
        print(f"url: {url}")

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
                    self.inverted_index[token][doc_id]["c"] += 1
                    self.inverted_index[token][doc_id]["s"] = max(self.important_tags.get(element.parent.name, 1), self.inverted_index[token][doc_id]["s"]) 

            elif isinstance(element, Tag):
                for child in element.children:
                    recursive_tokenize(child)

        # Call the recursive function on the soup object
        recursive_tokenize(soup)

    def index_all(self):
        for root, _, files in os.walk(URLS_PATH):
            for filename in files:
                if not filename.endswith('.json'):
                    continue

                filepath = os.path.join(root, filename)

                # try:
                with open(filepath, 'r') as json_file:
                    content = json.load(json_file)
                    url_without_fragment = self.defrag_url(content['url'])
                    self.index(url_without_fragment, content['content'])               
            
                # except Exception as e:
                #     print(f"An error occurred while processing {filepath}: {e}")

    def generate_report(self):
        unique_tokens = len(self.inverted_index)
        unique_urls = len(self.url_to_id)

        most_common_token = None
        common_token_count = 0

        for token in self.inverted_index:
            # Get the most common word in the inverted index
            occurences = sum(self.inverted_index[token][doc_id]["c"] for doc_id in self.inverted_index[token])
            # print(temp_occurences)
            if occurences > common_token_count:
                common_token_count = occurences
                most_common_token = token
        
        with open('stats.txt', 'w') as f:
            print(f"Number of unique tokens: {unique_tokens}", file=f)
            print(f"Number of unique urls: {unique_urls}", file=f)
            print(f"Most common token: '{most_common_token}' with {common_token_count} occurrences in {unique_urls} urls", file=f)

        # Save the inverted index to a file
        with open('inverted_index.json', 'w') as f:
            json.dump(dict(self.inverted_index), f, indent=4, separators=(',', ': '), ensure_ascii=False)
        # Save the URL to ID mapping to a file
        with open('url_to_id.json', 'w') as f:
            json.dump(dict(self.url_to_id), f, indent=4, separators=(',', ': '), ensure_ascii=False)
        # Save the ID to URL mapping to a file
        with open('id_to_url.json', 'w') as f:
            json.dump(dict(self.id_to_url), f, indent=4, separators=(',', ': '), ensure_ascii=False)


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

    def search(self, query):
        # list of the words in the query list tokenized
        query_text = re.sub(r'[^a-zA-Z0-9\s]', '', query)
        query_tokens = [self.stemmer.stem(word.lower()) for word in query_text.split()]

        if not query_tokens:
            return [] # return empty list
        
        # fetch doc ids for each word
        doc_ids = []
        for t in query_tokens:
            if t in self.inverted_index:
                doc_ids.append(set(self.inverted_index[t].keys()))
            else:
                return []
        
        # boolean and intersection of all doc sets
        res_docs = set.intersection(*doc_ids)

        # list of matching urls
        return [self.id_to_url[id] for id in res_docs]

if __name__ == "__main__":
    indexer = Indexer()
    indexer.index_all()
    indexer.generate_report()
    # print(indexer.inverted_index)
    # print(indexer.url_to_id)
    # print(indexer.id_to_url)