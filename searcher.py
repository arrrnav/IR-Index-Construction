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

# class Searcher:
#     def __init__(self):
#         self.stemmer = PorterStemmer()
#         self.positional_indexes = {}
#         self.document_freqs = {}
#         self.total_docs = 0
#         # Call the setup methods
#         self._load_pos_indexes()
#         self._calc_doc_stats()

    # def _load_pos_indexes(self):
    #     for i in range(1, 7):
    #         try:
    #             # Fix: Use the correct path that matches merger.py output
    #             with open(f"./positional_indexes/index_{i}.json", "r") as f:
    #                 self.positional_indexes[i] = json.load(f)
    #             print(f"Loaded positional index {i} with {len(self.positional_indexes[i])} tokens")
    #         except FileNotFoundError:
    #             print(f"positional index #{i} not found")
    #             self.positional_indexes[i] = {}
        
    # def _calc_doc_stats(self):
    #     all_docs = set() 
    #     for p_i in range(1, 7):
    #         try:
    #             # Fix: Use the correct path that matches merger.py output
    #             with open(f"./partial_indexes/index_{p_i}.jsonl", "r") as f:
    #                 for line in f:
    #                     posting = json.loads(line)
    #                     token = list(posting.keys())[0]
    #                     doc_list = posting[token]

    #                     # count unique docs for token 
    #                     unique_docs = set(doc_list.keys()) if isinstance(doc_list, dict) else set(doc_list)
    #                     self.document_freqs[token] = len(unique_docs)
    #                     all_docs.update(unique_docs)
    #         except FileNotFoundError:
    #             continue 
    #     self.total_docs = len(all_docs)
    #     print(f"total docs: {self.total_docs}")

    # def _get_token_partition(self, token):
    #     if not token: 
    #         return 1  # Fix: return 1 instead of 0 (partitions are 1-6)
    #     first = token[0].lower()
    #     if first in A_TO_D: return 1 
    #     if first in E_TO_H: return 2
    #     if first in I_TO_M: return 3 
    #     if first in N_TO_P: return 4
    #     if first in Q_TO_S: return 5
    #     if first in T_TO_Z: return 6
    #     return 6  # fallback for any other characters

#     def _preprocess_query(self, query):
#         tokens = re.findall(r'\b[a-zA-Z]+\b', query.lower())
#         stemmed = [self.stemmer.stem(token) for token in tokens]
#         filtered = [token for token in stemmed if len(token) > 1]
#         return filtered 

    # def _get_postings(self, token):
    #     partition = self._get_token_partition(token)
    #     if token not in self.positional_indexes[partition]:
    #         return {}
    #     pos = self.positional_indexes[partition][token]
    #     try:
    #         # Fix: Use correct path and method name
    #         with open(f"./partial_indexes/index_{partition}.jsonl", "r") as f:
    #             f.seek(pos)
    #             line = f.readline()  # Fix: readline() not readLine()
    #             posting = json.loads(line)
    #             return posting.get(token, {})
    #     except (FileNotFoundError, json.JSONDecodeError):
    #         return {}
    
    # def _calc_tf_idf(self, q_tokens, docs):
    #     scores = defaultdict(float)
    #     for t in q_tokens:
    #         postings = self._get_postings(t)
    #         if not postings: continue 
    #         df = self.document_freqs.get(t, 1)
    #         idf = math.log(self.total_docs / df) 
    #         for id in docs:
    #             if id in postings:
    #                 tf = postings[id] if isinstance(postings[id], (int, float)) else 1
    #                 tf_idf = 0 
    #                 if tf > 0:
    #                     tf_idf = (1+math.log(tf)) * idf
    #                 scores[id] += tf_idf
    #     return dict(scores)

#     def search(self, query, k=10, use_tfidf = True):
#         if not query.strip():
#             return [] 

#         q_tokens = self._preprocess_query(query)
#         if not q_tokens:
#             return [] 

#         print("searching for query tokens: ", q_tokens)
#         docs = None 
#         postings = {}

#         for t in q_tokens:
#             posting_list = self._get_postings(t)
#             if not posting_list:
#                 print(f"token: {t} not found")
#                 return []
        
#             postings[t] = posting_list
#             doc_set = set(posting_list.keys())
#             if docs is None:
#                 docs = doc_set 
#             else:
#                 docs = docs.intersection(doc_set)
        
#         if not docs:
#             return [] 

#         print(f"found {len(docs)} candidate docs")

#         if use_tfidf:
#             scores = self._calc_tf_idf(q_tokens, docs)
#         else:
#             # simple frequency  based scoring
#             scores = defaultdict(float)
#             for t in q_tokens:
#                 p_list = postings[t]
#                 for id in docs:
#                     if id in p_list:
#                         freq = p_list[id] if isinstance(p_list[id], (int, float)) else 1
#                         scores[id] += freq 
        
#         # sort by score and return top k 
#         sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#         return sorted_docs[:k]

#     def search_or(self, query, k=10):
#         if not query.strip():
#             return [] 
        
#         q_tokens = self._preprocess_query(query)
#         if not q_tokens:
#             return []

#         all_docs = set() 
#         t_postings = {}

#         for t in q_tokens:
#             p_list = self._get_postings(t)
#             if p_list:
#                 t_postings[t] = p_list
#                 all_docs.update(p_list.keys())
        
#         if not all_docs:
#             return []
    
#         print(f"found {len(all_docs)} docs with OR search")

#         scores = defaultdict(float)
#         for t in q_tokens:
#             if t not in t_postings:
#                 continue
            
#             p_list = t_postings[t]
#             df = self.document_freqs.get(t, 1)
#             idf = math.log(self.total_docs / df)

#             for id in p_list:
#                 tf = p_list[id] if isinstance(p_list[id], (int, float)) else 1
#                 tf_idf = 0 
#                 if tf > 0:
#                     tf_idf = (1+math.log(tf)) * idf
#                 scores[id] += tf_idf
            
#         sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#         return sorted_docs[:k]
    
#     def fetch_token_info(self, token):
#         stemmed_t = self.stemmer.stem(token.lower())
#         p_list = self._get_postings(stemmed_t)
#         if not p_list:
#             return {"token": token, "stemmed": stemmed_t, "found": False}

#         return {
#             "token": token,
#             "stemmed": stemmed_t,
#             "found": True,
#             "doc_freq": len(p_list),
#             "total_freq": sum(p_list.values()) if all(isinstance(v, (int, float)) for v in p_list.values()) else len(p_list),
#             "partition": self._get_token_partition(stemmed_t), 
#             "sample_docs": list(p_list.keys())[:5]
#         }

# # SAMPLE MAIN FUNCTION TO RUN THIS: 
# def main():
#     """Interactive search interface"""
#     searcher = Searcher()
    
#     print("=== IR Search Engine ===")
#     print("Commands:")
#     print("  search <query>     - AND search (all terms must match)")
#     print("  or <query>         - OR search (any term can match)")
#     print("  info <token>       - Get information about a specific token")
#     print("  exit               - Quit")
#     print()
    
#     while True:
#         try:
#             user_input = input("Search> ").strip()
            
#             if user_input.lower() == 'exit':
#                 break
            
#             if user_input.startswith('search '):
#                 query = user_input[7:]
#                 start_time = time.time()
#                 results = searcher.search(query)
#                 end_time = time.time()
                
#                 if results:
#                     print(f"\nFound {len(results)} results:")
#                     print(f"Results (retrieved in {end_time - start_time} seconds):")
#                     for i, (doc_id, score) in enumerate(results, 1):
#                         print(f"{i:2d}. Document {doc_id} (score: {score:.4f})")
#                 else:
#                     print("No results found.")
            
#             elif user_input.startswith('or '):
#                 query = user_input[3:]
#                 results = searcher.search_or(query)
                
#                 if results:
#                     print(f"\nFound {len(results)} results:")
#                     for i, (doc_id, score) in enumerate(results, 1):
#                         print(f"{i:2d}. Document {doc_id} (score: {score:.4f})")
#                 else:
#                     print("No results found.")
            
#             elif user_input.startswith('info '):
#                 token = user_input[5:]
#                 info = searcher.fetch_token_info(token)
                
#                 if info['found']:
#                     print(f"\nToken Information:")
#                     print(f"  Original: {info['token']}")
#                     print(f"  Stemmed: {info['stemmed']}")
#                     print(f"  Document Frequency: {info['doc_freq']}")
#                     print(f"  Total Frequency: {info['total_freq']}")
#                     print(f"  Partition: {info['partition']}")
#                     print(f"  Sample Documents: {info['sample_docs']}")
#                 else:
#                     print(f"Token '{token}' not found in index.")
            
#             else:
#                 print("Unknown command. Use 'search <query>', 'or <query>', 'info <token>', or 'exit'.")
            
#             print()
        
#         except KeyboardInterrupt:
#             print("\nstopping!")
#             break
#         except Exception as e:
#             print(f"Error: {e}")

# if __name__ == "__main__":
#     main()

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
        

    def _calc_query_vector(self, query_tokens):
        query_vector = {}

        token_counts = {}
        for token in query_tokens:
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1
        
        for token in token_counts:
            count = token_counts[token]

            tf = 1 + math.log(count)

            df = self.document_freqs.get(token, 1)
            
            if df > 0:
                idf = math.log(self.total_docs / df)
                query_vector[token] = tf * idf
        
        return query_vector
    
    def _calc_document_vector(self, doc_id, query_tokens):
        doc_vector = {}


        for token in query_tokens:
            postings = self._get_postings(token)
            if (doc_id in postings):
                tf = postings[doc_id]["c"]

                if tf > 0:
                    tf = 1 + math.log(tf)
                    df = self.document_freqs.get(token, 1)
                    idf = math.log(self.total_docs / df)

                    doc_vector[token] = tf*idf

        return doc_vector
    
    def _cosine_sim(self, query_vector, doc_vector):
        
        # get dot product
        dot_product = 0
        for token in query_vector:
            if token in doc_vector:
                dot_product += query_vector[token] * doc_vector[token]
        
        # Calculate magnitudes
        query_magnitude = math.sqrt(sum(query_vector[score]**2 for score in query_vector))
        doc_magnitude = math.sqrt(sum(doc_vector[score]**2 for score in doc_vector))

        # Return the cosin sim
        return dot_product / (query_magnitude * doc_magnitude)


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
        for t in q_tokens:
            postings = self._get_postings(t)
            df = self.document_freqs.get(t, 1)
            idf = math.log(self.total_docs / df) 
            for id in docs:
                if id in postings:
                    tf = postings[id]["c"]
                    tf_idf = 0 
                    if tf > 0:
                        tf_idf = (1+math.log(tf)) * idf * postings[id]["s"]
                    scores[id] += tf_idf
        return dict(scores)

    def search(self, query):
        """Enhanced search method that handles both regular and boolean queries"""
        # Check if query contains boolean operators
        # if self.has_boolean_operators(query):
        #     # Handle boolean query
        #     tokens = self.parse_boolean_query(query)
        #     print(f"Boolean query detected: {tokens}")
            
        #     doc_scores = self.evaluate_boolean_expression(tokens, index_f)
            
        #     if not doc_scores:
        #         return []
            
        #     # Sort by score and return top results
        #     sorted_results = sorted(doc_scores.items(), key=lambda x: x[1]['s'], reverse=True)
        #     return [self.id_to_url[doc_id] for doc_id, _ in sorted_results[:5]]
        
        # else:
        # Original search logic for regular queries
        # index_f.seek(0)

        query_text = re.sub(r'[^a-zA-Z0-9\s]', '', query)
        query_tokens = [self.stemmer.stem(word.lower()) for word in query_text.split()]
        query_tokens = sorted(set(query_tokens))
        print(query_tokens)

        if not query_tokens:
            print("shouldnt happen")
            return []

        ids_and_scores = defaultdict(int)

        postings = {}
        docs = {}
        # docs = set()

        # for token in query_tokens:
        #     posting = self._get_postings(token)
        #     if posting:
        #         docs.update(posting.keys())
            

        for token in query_tokens:
            posting = self._get_postings(token)
            if not posting:
                continue

            postings[token] = posting

            
            if not docs:
                docs = set(postings[token].keys())
            else:
                docs = docs.union(set(postings[token].keys()))
        
        if not docs:
            return []
        # # print("docs: ", docs)
        ids_and_scores = self._calc_tf_idf(query_tokens, docs)

        print("Scores:", ids_and_scores)
        
        ids_and_scores = sorted(ids_and_scores.items(), key=lambda x: x[1], reverse=True)

        
        # ids_and_scores = self._get_ids_and_scores(query_tokens, docs)
        # ids_and_scores = sorted(ids_and_scores.items(), key=lambda x: x[1], reverse=True)
        # print(ids_and_scores)
        return [self.id_to_url[doc_id] for doc_id, _ in ids_and_scores[:5]]


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
        for rank, result in enumerate(results):
            print(f'#{rank+1}: {result}')
        print()