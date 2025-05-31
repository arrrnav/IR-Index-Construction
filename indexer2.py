import os, json, ijson
from collections import defaultdict
from nltk.stem import PorterStemmer
import re
from urllib.parse import urlparse, urlunparse
from bs4 import BeautifulSoup
from math import log

PARTIAL_INDEX_URLS = 5000
PARTIAL_INDEX_ROOT = "./partial_indexes_dev"

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
        self.inverted_index = dict(sorted(self.inverted_index.items(), key=lambda x: x[0]))

        # Save the current SORTED inverted index to a file (prep for merging later)
        with open(f'{PARTIAL_INDEX_ROOT}/index_{self.index_num}.json', 'w') as f:
            json.dump(dict(self.inverted_index), f, indent=4, separators=(',', ': '), ensure_ascii=False)
        self.index_num += 1
        self.inverted_index = defaultdict(lambda: defaultdict(lambda: {"c": 0, "s": 0}))
        print(f"New partial index created! index_{self.index_num}.json")

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

        if self.next_available_id % PARTIAL_INDEX_URLS == 0:
            self.new_partial_index()

        # Update the URL to ID mapping
        self.url_to_id[url] = doc_id
        self.id_to_url[doc_id] = url

        soup = BeautifulSoup(content, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
            tag.decompose() # remove script and style tags
        
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
                    # increment the count and update score if the tag is more important than previously found
                    self.inverted_index[token][doc_id]["c"] += 1
                    self.inverted_index[token][doc_id]["s"] = max(
                            importance_score, 
                            self.inverted_index[token][doc_id]["s"]
                        )

        tags_list = [tag for tag in self.important_tags.keys()]
        for tag in soup(tags_list):
            tag.decompose()

        default_text = soup.get_text(separator=' ', strip=True)

        if not default_text:
            return

        tag_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', tag_text)
        tag_text = re.sub(r'\s+', ' ', tag_text).lower().strip()

        unstemmized_tokens = [token for token in tag_text.split() if len(token) > 2 and token not in STOP_WORDS]

        for pre_token in unstemmized_tokens:
            token = self.stemmer.stem(pre_token)
            # increment the count and update score if the tag is more important than previously found
            self.inverted_index[token][doc_id]["c"] += 1
            self.inverted_index[token][doc_id]["s"] = max(
                    importance_score, 
                    self.inverted_index[token][doc_id]["s"]
                )
        
            

        # def recursive_tokenize(element):
        #     if isinstance(element, NavigableString):
        #         # tokenize text
        #         text = re.sub(r'[^a-zA-Z0-9\s]', '', element)
        #         text = text.lower()
        #         unstemmized_tokens = text.split()
        #         for pre_token in unstemmized_tokens:
        #             token = self.stemmer.stem(pre_token)
        #             # increment the count and update score if the tag is more important than previously found
        #             self.inverted_index[token][doc_id]["c"] += 1
        #             self.inverted_index[token][doc_id]["s"] = max(self.important_tags.get(element.parent.name, 1), self.inverted_index[token][doc_id]["s"]) 

        #     elif isinstance(element, Tag):
        #         for child in element.children:
        #             recursive_tokenize(child)

        # # Call the recursive function on the soup object
        # recursive_tokenize(soup)

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

        # for token in self.inverted_index:
        #     for doc_id in self.inverted_index[token]:
        #         # Calculate the importance factor for each token and document ID
        #         self.inverted_index[token][doc_id]["s"] = self.get_importance_factor(token, doc_id)



    def generate_logs(self):
        # unique_tokens = len(self.inverted_index)
        # unique_urls = len(self.url_to_id)

        # most_common_token = None
        # common_token_count = 0

        # for token in self.inverted_index:
        #     # Get the most common word in the inverted index
        #     occurences = sum(self.inverted_index[token][doc_id]["c"] for doc_id in self.inverted_index[token])
        #     # print(temp_occurences)
        #     if occurences > common_token_count:
        #         common_token_count = occurences
        #         most_common_token = token
        
        # with open('stats_dev.txt', 'w') as f:
        #     print(f"Number of unique tokens: {unique_tokens}", file=f)
        #     print(f"Number of unique urls: {unique_urls}", file=f)
        #     print(f"Most common token: '{most_common_token}' with {common_token_count} occurrences in {unique_urls} urls", file=f)

        # Save the inverted index to a file
        with open('inverted_index_dev.json', 'w') as f:
            sorted_index = dict(sorted(self.inverted_index.items(), key=lambda x: x[0]))
            json.dump(sorted_index, f, indent=4, separators=(',', ': '), ensure_ascii=False)
        # Save the URL to ID mapping to a file
        with open('url_to_id_dev.json', 'w') as f:
            json.dump(dict(self.url_to_id), f, indent=4, separators=(',', ': '), ensure_ascii=False)
        # Save the ID to URL mapping to a file
        with open('id_to_url_dev.json', 'w') as f:
            json.dump(dict(self.id_to_url), f, indent=4, separators=(',', ': '), ensure_ascii=False)


class IndexSearcher:
    

    def __init__(self):
        self.id_to_url = None
        with open('id_to_url.json', 'r') as f:
            self.id_to_url = json.load(f)

        self.stemmer = PorterStemmer()
        

    # def has_boolean_operators(self, query):
    #     """Check if query contains boolean operators"""
    #     boolean_pattern = r'\b(AND|OR|NOT)\b'
    #     return bool(re.search(boolean_pattern, query))

    # def parse_boolean_query(self, query):
    #     """Parse boolean query into tokens and operators"""
    #     # Split on boolean operators while preserving them
    #     tokens = re.split(r'\s+(AND|OR|NOT)\s+', query)
    #     return [token.strip() for token in tokens if token.strip()]

    def search(self, query, index_f):
        index_generator = ijson.kvitems(index_f, '')

        bool_split_query = query.split("AND")
        if len(bool_split_query) > 1:
            intersection_tokens =[]

            split_query = []
            for query in bool_split_query:
                split_query.extend(query.strip().split(" "))

            # print(f"split_query: {split_query}\nbool_split_query: {bool_split_query}")

            for i in range(len(split_query)):
                # split_query[i] = split_query[i].strip()
                token = re.sub(r'[^a-zA-Z0-9\s]', '', split_query[i])
                intersection_tokens.append(self.stemmer.stem(token.lower()))
            # print(f"intersection_tokens: {intersection_tokens}")

            token_sets = defaultdict(set)
            docs_and_scores = defaultdict(int)

            for token in intersection_tokens:
                # doc_ids = None
                while True:
                    try:
                        curr_token, curr_doc_ids = next(index_generator)
                    except StopIteration:
                        break
                    if token == curr_token:
                        doc_ids = curr_doc_ids
                        break

                if doc_ids is None:
                    continue

                # if token in self.inverted_index:
                for doc_id in doc_ids:
                    token_sets[token].add(doc_id)
                    # calculate the importance factor for each token and document ID
                    # stack scores accross all tokens
                    # ids_and_scores[doc_id] += doc_ids[doc_id]["s"]
            # print(f"tokens: {token_sets.keys()}")
            final_set = set.intersection(*token_sets.values())
            return [self.id_to_url[doc_id] for doc_id in final_set][:5] if final_set else []
                
            
            

            
        else:
                

        # list of the tokens in the query list tokenized
            query_text = re.sub(r'[^a-zA-Z0-9\s]', '', query)
            query_tokens = [self.stemmer.stem(token.lower()) for token in query_text.split()]
            query_tokens = sorted(set(query_tokens)) # remove duplicates and sort the tokens
            # print(f"query_tokens: {query_tokens}")
            # print(query_tokens)

            if not query_tokens:
                print("shouldnt happen")
                return [] # return empty list

            ids_and_scores = defaultdict(int)

            for token in query_tokens:
                doc_ids = None
                while True:
                    try:
                        curr_token, curr_doc_ids = next(index_generator)
                    except StopIteration:
                        break
                    if token == curr_token:
                        doc_ids = curr_doc_ids
                        break

                if doc_ids is None:
                    continue

                # if token in self.inverted_index:
                for doc_id in doc_ids:
                    # calculate the importance factor for each token and document ID
                    # stack scores accross all tokens
                    ids_and_scores[doc_id] = (ids_and_scores[doc_id] + doc_ids[doc_id]["s"]) * 15 if doc_id in ids_and_scores else doc_ids[doc_id]["s"]

            # index_f.close()
                

            ids_and_scores = sorted(ids_and_scores.items(), key=lambda x: x[1], reverse=True)
            return [self.id_to_url[doc_id] for doc_id, _ in ids_and_scores[:5]]
        




if __name__ == "__main__":
    indexer = Indexer()
    indexer.index_all()
    indexer.generate_logs()