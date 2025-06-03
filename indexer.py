import os, json
from collections import defaultdict
from nltk.stem import PorterStemmer
import re
from urllib.parse import urlparse, urlunparse
from bs4 import BeautifulSoup
from math import log

PARTIAL_INDEX_URLS = 5000
PARTIAL_INDEX_ROOT = "./partial_indexes"

# PARTIAL_INDEX_URLS = 750
# PARTIAL_INDEX_ROOT = "./partial_indexes_analyst"

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

URLS_PATH = './developer/DEV'
# URLS_PATH = './analyst/ANALYST'

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
    "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves"
}

TOKEN_FILTERS = ['ensm', 'ensg']

class Indexer:
    def __init__(self):
        self.inverted_index = defaultdict(lambda: defaultdict(lambda: {"c": 0, "s": 0}))
        self.url_to_id = defaultdict(int)
        self.id_to_url = defaultdict(str)
        self.next_available_id = 0
        self.important_tags = {
            "title": 8,
            "h1": 10,
            "h2": 6,
            "h3": 4,
            "strong": 2,
            "b": 2
            # default: 1
        }
        self.stemmer = PorterStemmer()
        self.index_num = 1
        # self.position = 0
    
    def get_importance_factor(self, token, doc_id):
        # importance based on TF-IDF score and tag importance

        #TF-IDF score = TF(token, document) * IDF(token)
        # TF(token, document) = (number of times token appears in document) / (total number of tokens in document)
        # IDF(token) = log(total number of documents / number of documents containing token)

        # importance factor = TF-IDF score * tag importance
        tag_importance = self.inverted_index[token][doc_id]["s"]
        tf_score = self.inverted_index[token][doc_id]["c"] / len(self.inverted_index)
        idf_score = log(len(self.url_to_id) / len(self.inverted_index[token]))

        tf_idf_score = tf_score * idf_score

        return tf_idf_score * tag_importance

    def new_partial_index(self):
        print(f"New partial index created! index_{self.index_num}.json")
        self.inverted_index = dict(sorted(self.inverted_index.items(), key=lambda x: x[0]))
        # Save the current SORTED inverted index to a file (prep for merging later)
        with open(f'{PARTIAL_INDEX_ROOT}/index_{self.index_num}.json', 'w') as f:
            json.dump(dict(self.inverted_index), f, indent=4, separators=(',', ': '), ensure_ascii=False)
        self.index_num += 1
        self.inverted_index = defaultdict(lambda: defaultdict(lambda: {"c": 0, "s": 0}))
        

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
    
    def isFilterable(self, token) -> bool:
        if any(token.startswith(prefix) for prefix in TOKEN_FILTERS):
            return True
        try:
            int(token)
            return (len(token) != 4)
        except ValueError:
            return False
    
    def index(self, url, content):
        # Check if the URL is already indexed (for use in multiple URLS with same path but different fragments)
        if url in self.url_to_id:
            return

        # for debugging
        print(f"id: {self.next_available_id} url: {url}")

        # Assign a new ID to the URL if it doesn't exist 
        doc_id = self.next_available_id
        self.next_available_id += 1

        # Offload when reaching threshold
        if self.next_available_id % PARTIAL_INDEX_URLS == 0:
            self.new_partial_index()

        # Update the URL to ID mapping
        self.url_to_id[url] = doc_id
        self.id_to_url[doc_id] = url

        soup = BeautifulSoup(content, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
            tag.decompose() # remove script and style tags
        

        # First, parse all text within tags of importance. 
        for tag_name, importance_score in self.important_tags.items():
            for element in soup.find_all(tag_name):
                tag_text = element.get_text(separator=' ', strip=True)
                if not tag_text:
                    continue

                tag_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', tag_text)
                tag_text = re.sub(r'\s+', ' ', tag_text).lower().strip()

                unstemmized_tokens = [token for token in tag_text.split() if len(token) > 2 and token not in STOP_WORDS]

                for pre_token in unstemmized_tokens:
                    token = self.stemmer.stem(pre_token)
                    if self.isFilterable(token):
                        continue
                    # increment the count and update score if the tag is more important than previously found
                    self.inverted_index[token][doc_id]["c"] += 1
                    self.inverted_index[token][doc_id]["s"] = max(
                            importance_score, 
                            self.inverted_index[token][doc_id]["s"]
                        )

        # After text in important tags is processed, remove them from beautifulsoup
        tags_list = [tag for tag in self.important_tags.keys()]
        for tag in soup(tags_list):
            tag.decompose()

        # Repeat same tokenizing and parsing process for default text
        default_text = soup.get_text(separator=' ', strip=True)

        if not default_text:
            return

        default_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', default_text)
        default_text = re.sub(r'\s+', ' ', default_text).lower().strip()

        unstemmized_tokens = [token for token in default_text.split() if len(token) > 2 and token not in STOP_WORDS]

        for pre_token in unstemmized_tokens:
            token = self.stemmer.stem(pre_token)
            if self.isFilterable(token):
                continue
            # increment the count and update score if the tag is more important than previously found
            self.inverted_index[token][doc_id]["c"] += 1
            self.inverted_index[token][doc_id]["s"] = max(
                    1, 
                    self.inverted_index[token][doc_id]["s"]
                )

    def index_all(self):
        for root, _, files in os.walk(URLS_PATH):
            for filename in files:
                if not filename.endswith('.json'):
                    continue

                filepath = os.path.join(root, filename)

                try:
                    with open(filepath, 'r') as json_file:
                        content = json.load(json_file)
                        url_without_fragment = self.defrag_url(content['url'])
                        self.index(url_without_fragment, content['content'])               
                
                except Exception as e:
                    print(f"An error occurred while processing {filepath}: {e}")

        self.new_partial_index()

    def generate_logs(self):
        # Save the URL to ID mapping to a file
        with open('./stats/url_to_id.json', 'w') as f:
            json.dump(dict(self.url_to_id), f, indent=4, separators=(',', ': '), ensure_ascii=False)
        # Save the ID to URL mapping to a file
        with open('./stats/id_to_url.json', 'w') as f:
            json.dump(dict(self.id_to_url), f, indent=4, separators=(',', ': '), ensure_ascii=False)



if __name__ == "__main__":
    indexer = Indexer()
    indexer.index_all()
    indexer.generate_logs()
