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

TOKEN_FILTERS = ['ensm', 'ensg']


class Indexer:
    def __init__(self):
        self.inverted_index = defaultdict(lambda: defaultdict(lambda: {"c": 0, "s": 0}))
        self.url_to_id = defaultdict(int)
        self.id_to_url = defaultdict(str)
        self.next_available_id = 0
        self.important_tags = {
            "title": 20,
            "h1": 18,
            "h2": 16,
            "h3": 14,
            "strong": 12,
            "b": 12
            # default: 10
        }
        self.stemmer = PorterStemmer()
        self.index_num = 1
        self.stats_path = "./stats"
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
    
    def is_valid_token(self, token) -> bool:
        # if token in STOP_WORDS:
        #     return False
        if any(token.startswith(prefix) for prefix in TOKEN_FILTERS):
            return False
        try:
            int(token)
            return (len(token) <= 4)
        except ValueError:
            # try to clean any negatives and scientific notation
            cleaned_token = re.sub(r'[.\-eE/]', '', token)
            try:
                int(cleaned_token)
                return False
            except ValueError:
                return True
        
    def is_valid_url(self, url):
        # Decide whether to crawl this url or not. 
        # If you decide to crawl it, return True; otherwise return False.
        # There are already some conditions that return False.
        split_url = url.split('=')
        if split_url[-1] == "txt":
            return False

        try:
            parsed = urlparse(url)
            if parsed.scheme not in set(["http", "https"]):
                return False
            return not re.match(
                r".*.(css|js|bmp|gif|jpe?g|ico|svg"
                + r"|png|tiff?|mid|mp2|mp3|mp4"
                + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
                + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names|xml"
                + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
                + r"|epub|dll|cnf|tgz|sha1|m|ma|nb|pd|sql"
                + r"|thmx|mso|arff|rtf|jar|csv|shtml|htm|txt"
                + r"|rm|smil|wmv|swf|wma|zip|rar|gz|war|img|mpg|apk"
                + r"|c|cc|py|ipynb|h|cp?p|pov|lif|ppsx|pps|patch)$", parsed.path.lower())
        except TypeError:
            print ("TypeError for ", parsed)
            raise
    
    def index(self, url, content):
        # Check if the URL is already indexed (for use in multiple URLS with same path but different fragments)
        if url in self.url_to_id:
            return
        
        if not self.is_valid_url(url):
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

                unstemmized_tokens = [token for token in tag_text.split()]

                for pre_token in unstemmized_tokens:
                    token = self.stemmer.stem(pre_token)
                    if not self.is_valid_token(token):
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

        unstemmized_tokens = [token for token in default_text.split()]

        for pre_token in unstemmized_tokens:
            token = self.stemmer.stem(pre_token)
            if not self.is_valid_token(token):
                continue
            # increment the count and update score if the tag is more important than previously found
            self.inverted_index[token][doc_id]["c"] += 1
            self.inverted_index[token][doc_id]["s"] = max(
                    10, 
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
        with open(f'{self.stats_path}/url_to_id.json', 'w') as f:
            json.dump(dict(self.url_to_id), f, indent=4, separators=(',', ': '), ensure_ascii=False)
        # Save the ID to URL mapping to a file
        with open(f'{self.stats_path}/id_to_url.json', 'w') as f:
            json.dump(dict(self.id_to_url), f, indent=4, separators=(',', ': '), ensure_ascii=False)



if __name__ == "__main__":
    indexer = Indexer()
    indexer.index_all()
    indexer.generate_logs()
