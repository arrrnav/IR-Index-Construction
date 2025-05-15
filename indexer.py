import os, json, bs4
from collections import defaultdict
from nltk.stem import PorterStemmer

URLS_PATH = './developer/DEV/aiclub_ics_uci_edu'

class Indexer:
    def __init__(self):
        print("GELLO")
        # example index key, value:
        # 'gilbert': {'doc1': [1, 2], 'doc2': [3, 4]}
        # - gilbert appears in doc1 at positions 1 and 2, and in doc2 at positions 3 and 4
        self.inverted_index = defaultdict(lambda: defaultdict(list))
        self.url_to_id = defaultdict(int)
        self.id_to_url = defaultdict(str)
        self.next_available_id = 0
        self.important_tags = ["h1", "h2", "h3", "h4", "h5", "h6", "strong", "b"]
        self.stemmer = PorterStemmer()

    def index(self, url, content):
        doc_id = None
        if url not in self.url_to_id:
            # Assign a new ID to the URL if it doesn't exist
            doc_id = self.next_available_id

            # Maps the URL to a unique ID
            self.url_to_id[url] = doc_id

            # Maps the unique ID to the URL
            self.id_to_url[doc_id] = url
            self.next_available_id += 1
        else:
            doc_id = self.url_to_id[url]

        soup = bs4.BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
        # WORK NEEDS TO BE DONE HERE
        # TOKENIZE AND STEMIZE THE TEXT
        words = text.split()
        # print(words)
        print(f"doc_id: {doc_id}")
        for position, word in enumerate(words):
            word = word.lower()
            stemmed_word = self.stemmer.stem(word)
            self.inverted_index[stemmed_word][doc_id].append(position)
            print(self.inverted_index[stemmed_word][doc_id])

    def index_all(self):
        for root, _, files in os.walk(URLS_PATH):
            for filename in files:
                if not filename.endswith('.json'):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r') as json_file:
                        content = json.load(json_file)
                        self.index(content['url'], content['content'])
                except Exception as e:
                    print(f"An error occurred while processing {filepath}: {e}")


if __name__ == "__main__":
    indexer = Indexer()
    indexer.index_all()
    
    # Save the inverted index to a file
    with open('inverted_index.json', 'w') as f:
        json.dump(dict(indexer.inverted_index), f, indent=4, separators=(',', ': '), ensure_ascii=False)
    # Save the URL to ID mapping to a file
    with open('url_to_id.json', 'w') as f:
        json.dump(dict(indexer.url_to_id), f, indent=4, separators=(',', ': '), ensure_ascii=False)
    # Save the ID to URL mapping to a file
    with open('id_to_url.json', 'w') as f:
        json.dump(dict(indexer.id_to_url), f, indent=4, separators=(',', ': '), ensure_ascii=False)