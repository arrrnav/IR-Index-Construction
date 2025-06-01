from collections import defaultdict
import json, ijson



A_TO_D = set("abcd")
E_TO_H = set("efgh")
I_TO_M = set("ijklm")
N_TO_P = set("nop")
Q_TO_S = set("qrs")
T_TO_Z = set("tuvwxyz")

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
        root_path = "./partial_indexes_dev"
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
        
        pos_index_1 = defaultdict(int)
        pos_index_2 = defaultdict(int)
        pos_index_3 = defaultdict(int)
        pos_index_4 = defaultdict(int)
        pos_index_5 = defaultdict(int)
        pos_index_6 = defaultdict(int)

        positional_indexes = [pos_index_1, pos_index_2, pos_index_3, pos_index_4, pos_index_5, pos_index_6]

        pos1 = 0
        pos2 = 0
        pos3 = 0
        pos4 = 0
        pos5 = 0
        pos6 = 0

        with open(SAMPLE_INDEX_PATH, 'r') as f:
            for line in f:
                line_length = len(line.strip())
                posting = json.loads(line)
                token = list(posting.keys())[0]

                if token[0].lower() in A_TO_D:
                    with open("./partial_indexes/index_1.jsonl", "a") as f1:
                        json.dump(posting, f1, separators=(',', ':'))
                        f1.write('\n')
                        pos_index_1[token] = pos1
                    pos1 += line_length + 2

                elif token[0].lower() in E_TO_H:
                    with open("./partial_indexes/index_2.jsonl", "a") as f2:
                        json.dump(posting, f2, separators=(',', ':'))
                        f2.write('\n')
                        pos_index_2[token] = pos2
                    pos2 += line_length + 2

                elif token[0].lower() in I_TO_M:
                    with open("./partial_indexes/index_3.jsonl", "a") as f3:
                        json.dump(posting, f3, separators=(',', ':'))
                        f3.write('\n')
                        pos_index_3[token] = pos3
                    pos3 += line_length + 2

                elif token[0].lower() in N_TO_P:
                    with open("./partial_indexes/index_4.jsonl", "a") as f4:
                        json.dump(posting, f4, separators=(',', ':'))
                        f4.write('\n')
                        pos_index_4[token] = pos4
                    pos4 += line_length + 2

                elif token[0].lower() in Q_TO_S:
                    with open("./partial_indexes/index_5.jsonl", "a") as f5:
                        json.dump(posting, f5, separators=(',', ':'))
                        f5.write('\n')
                        pos_index_5[token] = pos5
                    pos5 += line_length + 2

                else:
                    with open("./partial_indexes/index_6.jsonl", "a") as f6:
                        json.dump(posting, f6, separators=(',', ':'))
                        f6.write('\n')
                        pos_index_6[token] = pos6
                    pos6 += line_length + 2


        for index, pos_index in enumerate(positional_indexes, start=1):
            with open(f"./positional_indexes/index_{index}.json", "w") as f:
                json.dump(pos_index, f)

    def posting_search(self, token: str) -> dict:
        if token[0].lower() in A_TO_D:
            pos = None
            with open("./positional_indexes/index_1.json", "r") as f1:
                data = json.load(f1)
                pos = data.get(token, None)
            
            if pos is None:
                return (f"Token '{token}' not found in index 1.")
            
            with open("partial_indexes/index_1.jsonl") as f1:
                f1.seek(pos)
                line = f1.readline()
                return json.loads(line)

        elif token[0].lower() in E_TO_H:
            pos = None
            with open("./positional_indexes/index_2.json", "r") as f2:
                data = json.load(f2)
                pos = data.get(token, None)
            
            if pos is None:
                return (f"Token '{token}' not found in index 2.")
            
            with open("partial_indexes/index_2.jsonl") as f2:
                f2.seek(pos)
                line = f2.readline()
                return json.loads(line)

        elif token[0].lower() in I_TO_M:
            pos = None
            with open("./positional_indexes/index_3.json", "r") as f3:
                data = json.load(f3)
                pos = data.get(token, None)
            
            if pos is None:
                return (f"Token '{token}' not found in index 3.")
            
            with open("partial_indexes/index_3.jsonl") as f3:
                f3.seek(pos)
                line = f3.readline()
                return json.loads(line)

        elif token[0].lower() in N_TO_P:
            pos = None
            with open("./positional_indexes/index_4.json", "r") as f4:
                data = json.load(f4)
                pos = data.get(token, None)
            
            if pos is None:
                return (f"Token '{token}' not found in index 4.")
            
            with open("partial_indexes/index_4.jsonl") as f4:
                f4.seek(pos)
                line = f4.readline()
                return json.loads(line)

        elif token[0].lower() in Q_TO_S:
            pos = None
            with open("./positional_indexes/index_5.json", "r") as f5:
                data = json.load(f5)
                pos = data.get(token, None)
            
            if pos is None:
                return (f"Token '{token}' not found in index 5.")
            
            with open("partial_indexes/index_5.jsonl") as f5:
                f5.seek(pos)
                line = f5.readline()
                return json.loads(line)

        else:
            pos = None
            with open("./positional_indexes/index_6.json", "r") as f6:
                data = json.load(f6)
                pos = data.get(token, None)
            
            if pos is None:
                return (f"Token '{token}' not found in index 6.")
            
            with open("partial_indexes/index_6.jsonl") as f6:
                f6.seek(pos)
                line = f6.readline()
                print(line)
                return json.loads(line)


        # pos = None
        # with open("./positional_indexes/index_2.json", "r") as f2:
        #     data = json.load(f2)
        #     pos = data.get(token, None)
        
        # if pos is None:
        #     return (f"Token '{token}' not found in index 2.")
        
        # print("pos", pos)
        
        # with open("partial_indexes/index_2.jsonl") as f2:
        #     f2.seek(55)
        #     line = f2.readline()
        #     print("line", line)
                # return json.loads(line)

        

if __name__ == "__main__":
    merger = Merger()
    merger.merge_files(11)
    # merger.splitAlpha()
    
    # while True:
    #     token = input("Enter a token to search for (or 'exit' to quit): ").strip()
    #     if token.lower() == 'exit':
    #         break
    #     result = merger.posting_search(token)
    #     print(result)