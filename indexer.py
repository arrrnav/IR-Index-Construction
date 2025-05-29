import os, json, bs4
from collections import defaultdict
from nltk.stem import PorterStemmer
import re
import ijson
from urllib.parse import urlparse

URLS_PATH_A = './ANALYST'
URLS_PATH_D = './ANALYST'  # Using ANALYST for testing
QUERIES = ["cristina lopes" ,"machine learning", "ACM", "master of software engineering"]

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
        self.docs = 0

    def search(self, query):
        # list of the words in the query list tokenized
        query_tokens = [self.stemmer.stem(word.lower()) for word in query.split()]

        if not query_tokens:
            return [] # return empty list
        
        # fetch doc ids for each word
        doc_ids = []
        for t in query_tokens:
            if t in self.inverted_index:
                doc_ids.append(set(self.inverted_index[word].keys()))
            else:
                return []
        
        # boolean and intersection of all doc sets
        res_docs = set.intersection(*doc_ids)

        # list of matching urls
        return [self.id_to_url[id] for id in res_docs]

    def defrag_url(self, url):
        # Parse the URL and remove the fragment
        parsed_url = urlparse(url)
        url_without_fragment = parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
        return url_without_fragment

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
        
        # Go through the important tags and extract text
        for tag in self.important_tags:
            # Append them in the dictionary so that
            # Each Key corresponds to the tag and the value is the text
            # inside the tag
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
    

    def search(self, query):
        # list of the words in the query list tokenized
        query_tokens = [self.stemmer.stem(word.lower()) for word in query.split()]

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
        for root, _, files in os.walk(URLS_PATH_D):
            for filename in files:
                if not filename.endswith('.json'):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r') as json_file:
                        content = json.load(json_file)
                        url_without_fragment = self.defrag_url(content['url'])
                        self.index(url_without_fragment, content['content'])

                        # Calculate the score for each word in the inverted index
                        # and append it to the list of positions
                        for word in self.inverted_index:
                            # if the word appears in the document with the given ID
                            if (self.url_to_id[url_without_fragment] in self.inverted_index[word]):
                                score = (self.term_frequency_score(word, self.url_to_id[url_without_fragment]) * 
                                        self.inverse_document_frequency_score(word) + self.get_word_importance_factor(word))
                                # Append the score to the list of positions
                                self.inverted_index[word][self.url_to_id[url_without_fragment]].append(score)

                        # Reset dict for every link/doc
                        self.words_in_tags = defaultdict(str)
                
            
                except Exception as e:
                    print(f"An error occurred while processing {filepath}: {e}")
                    

    def get_stats(self):
        # gather stats
        num_unique_words = len(self.inverted_index)
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


    def alphaFirst(self,keys):
        # Sort the keys in alphabetical order and returns the first one
        temp = list(keys)
        temp.sort()
        return temp[0]
    

    def writeToFile(self, obj):
        # Opens file and write it so that each json obj is on a single line
        with open("combined_index.jsonl", "a") as f:
            json.dump(obj, f, separators=(',', ':'))
            f.write('\n')

    def merge_files(self,n):
        root_path = "./partial_indexes"
        parsed_files = [] # stores the file iterators
        keys = {}

        # populate the keys and generators strcuctures
        for i in range(1, n+1):
            json_file = open(f"{root_path}/index_{i}.json", 'r')
            # Stores the generator for each partial index
            parsed_files.append(ijson.kvitems(json_file, ""))

            # Handle duplicates, Append the values if it already exists to the Keys dict
            # For example if key "ai" is already in the keys dict make sure to append the new document values to that key
            # When inserting keep track of the generator index as well so we can generate new values later
            while True:
                # Get next
                temp_key, val = next(parsed_files[i-1])
                if temp_key not in keys:
                    keys[temp_key] = [i-1, val]
                    break
                else:
                    keys[temp_key][1] = keys[temp_key][1] | val


        # Main merge loop
        while True:

            # Select the key that comes alphabetically first 
            selected_key = self.alphaFirst(keys.keys())

            # Make the object to write to the combined index file
            '''
                "key": { 
                        {
                            doc_id_1: {
                                "c": count,
                                "s": score
                            },
                            doc_id_1: {
                                "c": count,
                                "s": score
                            }
                        }
            '''
            # By accessing the second value, I am retrieveing the documents object
            temp = {selected_key: keys[selected_key][1]}

            # Write to the file
            self.writeToFile(temp)

            # Gets the key's generator index
            selected_key_index = keys[selected_key][0]
            try:
                while True:
                    # New values from generator
                    temp_key, temp_val = next(parsed_files[selected_key_index])
                    if temp_key in keys:
                        # Merge the new value with existing values
                        # For example if key "ai" is already in the keys dict make sure to append the new document values to that key
                        keys[temp_key][1] = keys[temp_key][1] | temp_val
                    else:
                        # Remove the old key and add the new one with the new value
                        keys.pop(selected_key)
                        keys[temp_key] = [selected_key_index, temp_val]
                        break
            except StopIteration:
                # If a generator has completed remove it from the keys list
                keys.pop(selected_key)
                # If all generators have completed except only one, break
                if len(keys) == 1:
                    break

        # Get the straggler value left in the keys dict and write it
        selected_key = list(keys.keys())[0]
        index = keys[selected_key][0]
        self.writeToFile({selected_key :keys[selected_key][1]})

        # Clean up loop to add everything else from the reamaining generator into the file
        while True:
            try:
                # Get key and val and add it to the file
                key, val = next(parsed_files[index])
                self.writeToFile({key: val})
            
            except StopIteration:
                break
        

if __name__ == "__main__":
    indexer = Indexer()
    
    # Step 1: Create the initial index from documents
    print("Step 1: Creating initial index from documents...")
    indexer.index_all()
    
    # Step 2: Save the inverted index to a file
    print("Step 2: Saving inverted index...")
    with open('inverted_index.json', 'w') as f:
        json.dump(dict(indexer.inverted_index), f, indent=4, separators=(',', ': '), ensure_ascii=False)
    
    # Save the URL to ID mapping to a file
    with open('url_to_id.json', 'w') as f:
        json.dump(dict(indexer.url_to_id), f, indent=4, separators=(',', ': '), ensure_ascii=False)
    
    # Save the ID to URL mapping to a file
    with open('id_to_url.json', 'w') as f:
        json.dump(dict(indexer.id_to_url), f, indent=4, separators=(',', ': '), ensure_ascii=False)

    # Step 3: Generate stats
    print("Step 3: Generating statistics...")
    indexer_stats = indexer.get_stats()
    print(f"Number of unique words: {indexer_stats[0]}")
    print(f"Number of unique documents: {indexer_stats[1]}")
    print(f"Most common word: '{indexer_stats[2]}' with {indexer_stats[-1]} occurrences in {indexer_stats[1]} documents")

    # Step 4: Test some queries
    print("\nStep 4: Testing queries...")
    for query in QUERIES:
        print(f"\nCurrent query - {query}")
        res = indexer.search(query)
        for i, url in enumerate(res[:5], 1):
            print(f"#{i} url = {url}")
    
    print("\n✓ Indexing complete! Now you can run the merger and searcher.")


