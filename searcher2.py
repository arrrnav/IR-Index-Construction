import json 
import re 
from collections import defaultdict
from nltk.stem import PorterStemmer
from typing import List, Dict, Set, Tuple 
import math 
import time

A_TO_D = set("abcd")
E_TO_H = set("efgh")
I_TO_M = set("ijklm")
N_TO_P = set("nop")
Q_TO_S = set("qrs")
T_TO_Z = set("tuvwxyz")


class Searcher:
    def __init__(self):
        self.positional_indexes = {}
        self.pos_indexes_path = "./positional_indexes"
        self.split_path = "./alphabetized_indexes"
        self.id_to_url = {}
        self.stats_path = "./stats"
        self.stemmer = PorterStemmer()
        self.document_freqs = {}
        self._load_pos_indexes()
        self._calc_doc_stats()
        
        
    def _load_pos_indexes(self):
        for i in range(1, 7):
            try:
                # Fix: Use the correct path that matches merger.py output
                with open(f"{self.pos_indexes_path}/index_{i}.json", "r") as f:
                    self.positional_indexes[i] = json.load(f)
                print(f"Loaded positional index {i} with {len(self.positional_indexes[i])} tokens")
            except FileNotFoundError:
                print(f"positional index #{i} not found")
                self.positional_indexes[i] = {}
        
        with open (f"{self.stats_path}/id_to_url.json", "r") as f:
            self.id_to_url = json.loads(f.read())

    
    def _calc_doc_stats(self):
        all_docs = set() 
        for i in range(1, 7):
            with open(f"{self.split_path}/index_{i}.jsonl", "r") as f:
                for line in f:
                    posting = json.loads(line)
                    token = list(posting.keys())[0]
                    doc_list = posting[token]

                    # count unique docs for token 
                    self.document_freqs[token] = len(doc_list)
                    all_docs.update(list(doc_list.keys()))
        self.total_docs = len(all_docs)
        print(f"total docs: {self.total_docs}")

    def _get_token_partition(self, token):
        if not token: 
            return 1  # Fix: return 1 instead of 0 (partitions are 1-6)
        first = token[0].lower()
        if first in A_TO_D: return 1 
        if first in E_TO_H: return 2
        if first in I_TO_M: return 3 
        if first in N_TO_P: return 4
        if first in Q_TO_S: return 5
        if first in T_TO_Z: return 6
        return 6  # fallback for any other characters
    
    def _get_postings(self, token):
        partition = self._get_token_partition(token)

        try:

            pos = self.positional_indexes[partition][token]
            with open(f"{self.split_path}/index_{partition}.jsonl", "r") as f:
                # print(partition)
                f.seek(pos)
                line = f.readline()  # Fix: readline() not readLine()
                # print(line)
                posting = json.loads(line)
                return posting[token]
        except:
            return None


    def _get_ids_and_scores(self, query_tokens, docs):
        sims = {}

        query_vector = self._calc_query_vector(query_tokens)

        if not query_tokens:
            return
        
        for doc_id in docs:
            doc_vector = self._calc_document_vector(doc_id, query_tokens)
            sim = self._cosine_sim(query_vector, doc_vector) 

            sims[doc_id] = sim

        return sims


    def _calc_tf_idf(self, q_tokens, docs):
        scores = defaultdict(float)
        # visited = set()
        for t in q_tokens:
            postings = self._get_postings(t)
            if not postings:
                continue
            df = self.document_freqs.get(t, 1)
            idf = math.log(self.total_docs / df) 
            for id in docs:
                if id in postings:
                    tf = postings[id]["c"]
                    tf_idf = 0 
                    if tf > 0:
                        tf_idf = (1+math.log(tf)) * idf * postings[id]["s"]
                    if id in scores:
                        scores[id] = (scores[id] + tf_idf) * 4
                    else:
                        scores[id] += tf_idf

        return dict(scores)
    
    def _fetch_from_query(self, tokens: list[str], is_bool: bool) -> list[str]:
        ids_and_scores = defaultdict(int)

        docs = set()
        
        for token in tokens:
            posting = self._get_postings(token)
            if not posting:
                if is_bool:
                    return []
                continue

            if not docs:
                docs.update(posting.keys())
            else:
                if is_bool:
                    docs.intersection_update(posting.keys())
                else:
                    docs.update(posting.keys())

        if not docs:
            return []
        
        print(f"debug: {tokens}")

        ids_and_scores = self._calc_tf_idf(tokens, docs)
        ids_and_scores = sorted(ids_and_scores.items(), key=lambda x: x[1], reverse=True)

        return [self.id_to_url[doc_id] for doc_id, _ in ids_and_scores[:5]]

    def search(self, query):
        """Enhanced search method that handles both regular and boolean queries"""

        query_text = re.sub(r'[^a-zA-Z0-9\s]', '', query)
        split_text = query_text.split()

        query_tokens = [self.stemmer.stem(word.lower()) for word in split_text]

        print(query_tokens)

        if not query_tokens:
            print("No substantial query found")
            return []

        if "AND" in split_text:
            return self._fetch_from_query(query_tokens, is_bool=True)
        return self._fetch_from_query(query_tokens, is_bool=False)


if __name__ == "__main__":
    # for query in ex_queries:
        # print(f"Query: {query}")
    searcher = Searcher()
    while True:
        query = input("Enter a search query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        start_time = time.time()
        results = searcher.search(query)
        end_time = time.time()
        print(f"Results (retrieved in {(end_time - start_time)*1000} milliseconds):")
        print("Results:")
        if not results:
            print("No search results found")
        for rank, result in enumerate(results):
            print(f'#{rank+1}: {result}')
        print()