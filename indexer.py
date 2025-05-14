import os, json, bs4
from collections import defaultdict

URLS_PATH = './developer/DEV'

class Indexer:
    def __init__(self):
        print("GELLO")
        # example index key, value:
        # 'gilbert': {'doc1': [1, 2], 'doc2': [3, 4]}
        # - gilbert appears in doc1 at positions 1 and 2, and in doc2 at positions 3 and 4
        self.inverted_index = defaultdict(defaultdict(list))
        self.url_to_id = defaultdict(int)
        self.id_to_url = defaultdict(str)
        self.next_available_id = 0

    def index(self, url, content):
        if url not in self.url_to_id:
            doc_id = self.next_available_id
            self.url_to_id[url] = doc_id
            self.id_to_url[doc_id] = url
            self.next_available_id += 1

        soup = bs4.BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
        # WORK NEEDS TO BE DONE HERE
        # TOKENIZE AND STEMIZE THE TEXT
        words = text.split()

        for position, word in enumerate(words):
            word = word.lower()
            self.inverted_index[word][url].append(position)

    def index_all(self):
        for root, _, files in os.listdir(URLS_PATH):
            for filename in files:
                if not filename.endswith('.json'):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r') as json_file:
                        content = json.load(json_file)
                        self.index(content['url'], content['content'])
                except Exception as e:
                    print(f"An error occurred while processing {json_file}: {e}")