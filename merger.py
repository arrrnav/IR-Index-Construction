import os, json, bs4
from collections import defaultdict
from nltk.stem import PorterStemmer
import re
import ijson
from urllib.parse import urlparse

URLS_PATH_A = './analyst/ANALYST'
URLS_PATH_D = './developer/DEV'
QUERIES = ["cristina lopes" ,"machine learning", "ACM", "master of software engineering"]

class Merger:
    def __init__(self):
        pass

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

    def splitAlpha(self):
        SAMPLE_INDEX_PATH = './full_index_test.jsonl'
        '''
        Subdivision | Letters             | Cumulative Percentage
        ------------|---------------------|----------------------
        1           | A, B, C, D          | 16.88%
        2           | E, F, G, H          | 17.76%
        3           | I, J, K, L, M       | 18.17%
        4           | N, O, P             | 16.38%
        5           | Q, R, S             | 16.75%
        6           | T, U, V, W, X, Y, Z | 14.06%
        '''

        a_to_d = set("abcd")
        e_to_h = set("efgh")
        i_to_m = set("ijklm")
        n_to_p = set("nop")
        q_to_s = set("qrs")
        t_to_z = set("tuvwxyz")

        with open(SAMPLE_INDEX_PATH, 'r') as f:
            for line in f:
                obj = json.loads(line)
                key = list(obj.keys())[0]
                if key[0].lower() in a_to_d:
                    with open("./partial_indexes/index_1.json", "a") as f1:
                        json.dump(obj, f1, separators=(',', ':'))
                        f1.write('\n')
                elif key[0].lower() in e_to_h:
                    with open("./partial_indexes/index_2.jsonl", "a") as f2:
                        json.dump(obj, f2, separators=(',', ':'))
                        f2.write('\n')
                elif key[0].lower() in i_to_m:
                    with open("./partial_indexes/index_3.jsonl", "a") as f3:
                        json.dump(obj, f3, separators=(',', ':'))
                        f3.write('\n')
                elif key[0].lower() in n_to_p:
                    with open("./partial_indexes/index_4.jsonl", "a") as f4:
                        json.dump(obj, f4, separators=(',', ':'))
                        f4.write('\n')
                elif key[0].lower() in q_to_s:
                    with open("./partial_indexes/index_5.jsonl", "a") as f5:
                        json.dump(obj, f5, separators=(',', ':'))
                        f5.write('\n')
                else:
                    with open("./partial_indexes/index_6.jsonl", "a") as f6:
                        json.dump(obj, f6, separators=(',', ':'))
                        f6.write('\n')

        

if __name__ == "__main__":
    merger = Merger()
    merger.splitAlpha()