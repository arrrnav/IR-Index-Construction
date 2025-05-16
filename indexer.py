import os, json, bs4
from collections import defaultdict
from nltk.stem import PorterStemmer
import re
from urllib.parse import urlparse

URLS_PATH = './developer/DEV'


class Indexer:
    def __init__(self):
        # example index key, value:
        # 'gilbert': {'doc1': [1, 2], 'doc2': [3, 4]}
        # - gilbert appears in doc1 at positions 1 and 2, and in doc2 at positions 3 and 4
        self.inverted_index = defaultdict(lambda: defaultdict(list))
        self.url_to_id = defaultdict(int)
        self.id_to_url = defaultdict(str)
        self.next_available_id = 0
        self.important_tags = ["h1", "h2", "h3", "strong", "b", "title"]
        self.stemmer = PorterStemmer()
        self.words_in_tags = defaultdict(str)
        self.words = []

    def index(self, url, content):
        doc_id = None
        parsed_url = urlparse(url)
        url_without_fragment = parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
        if url_without_fragment not in self.url_to_id:
            # Assign a new ID to the URL if it doesn't exist
            doc_id = self.next_available_id

            # Maps the URL to a unique ID
            self.url_to_id[url_without_fragment] = doc_id

            # Maps the unique ID to the URL
            self.id_to_url[doc_id] = url_without_fragment
            self.next_available_id += 1
        else:
            doc_id = self.url_to_id[url_without_fragment]

        soup = bs4.BeautifulSoup(content, 'html.parser')
        
        # Go through the important tags and extract text
        for tag in self.important_tags:
            # Append them in the dictionary so that
            # Each Key corresponds to the tag and the value is the text
            # inside the tag
            for element in soup.find_all(tag):
                text = element.get_text()
                # Apply filters
                text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
                text = text.lower()
                self.words_in_tags[tag] += text + ' '

        print(f"url: {url}")
        # print(self.words_in_tags)
        text = soup.get_text()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # WORK NEEDS TO BE DONE HERE
        # TOKENIZE AND STEMIZE THE TEXT
        self.words = text.split()
        # print(words)
        # print(f"doc_id: {doc_id}")
        
        for position, word in enumerate(self.words):
            word = word.lower()
            stemmed_word = self.stemmer.stem(word)
            self.inverted_index[stemmed_word][doc_id].append(position)
            # score Formula:

    def get_word_importance_factor(self, word):
        for tag in self.important_tags:
            if word in self.words_in_tags[tag]:
                # Calculate the importance factor based on the tag
                if tag == "h1":
                    return 4
                elif tag == "h2":
                    return 3
                elif tag == "h3":
                    return 2
                elif tag == "strong" or tag == "b":
                    return 1
                elif tag == "title":
                    return 5
        return 1

    def term_frequency_score(self, word, doc_id):
        score = len(self.inverted_index[word][doc_id]) / len(self.words)
        
        return score
    
    def inverse_document_frequency_score(self, word):
        score = len(self.url_to_id) / len(self.inverted_index[word])
        return score

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

                        # Calculate the score for each word in the inverted index
                        # and append it to the list of positions
                        for word in self.inverted_index:
                            if (self.url_to_id[content['url']] in self.inverted_index[word]):
                                score = (self.term_frequency_score(word, self.url_to_id[content['url']]) * 
                                        self.inverse_document_frequency_score(word) + self.get_word_importance_factor(word))
                                # Append the score to the list of positions
                                self.inverted_index[word][self.url_to_id[content['url']]].append(score)

                        # Reset dict for every link/doc
                        self.words_in_tags = defaultdict(str)
                
            
                except Exception as e:
                    print(f"An error occurred while processing {filepath}: {e}")
            
                    

    def get_stats(self):
        # Get the number of unique words in the inverted index
        num_unique_words = len(self.inverted_index)
        # Get the number of unique documents in the inverted index
        num_unique_docs = len(self.url_to_id)

        most_common_word = None
        total_occurences = 0
        for word in self.inverted_index:
            # Get the most common word in the inverted index
            temp_occurences = sum((len(positions)-1) for positions in self.inverted_index[word].values())
            # print(temp_occurences)
            if temp_occurences > total_occurences:
                total_occurences = temp_occurences
                most_common_word = word
        
        with open('stats.txt', 'w') as f:
            print(f"Number of unique words: {num_unique_words}", file=f)
            print(f"Number of unique documents: {num_unique_docs}", file=f)
            print(f"Most common word: '{most_common_word}' with {total_occurences} occurrences in {num_unique_docs} documents", file=f)
        return num_unique_words, num_unique_docs, most_common_word, total_occurences

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

    indexer_stats = indexer.get_stats()
    print(f"Number of unique words: {indexer_stats[0]}")
    print(f"Number of unique documents: {indexer_stats[1]}")
    print(f"Most common word: '{indexer_stats[2]}' with {indexer_stats[-1]} occurrences in {indexer_stats[1]} documents")