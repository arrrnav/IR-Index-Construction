#!/usr/bin/env python3

import json
import os
from collections import defaultdict

def create_alphabetical_partitions():
    """
    Create 8 alphabetically sorted JSON files from the existing partial index.
    Distribution:
    1. A-C
    2. D-F  
    3. G-I
    4. J-L
    5. M-O
    6. P-R
    7. S-U
    8. V-Z
    """
    
    # Define the alphabetical ranges
    partitions = {
        1: set("abc"),
        2: set("def"),
        3: set("ghi"),
        4: set("jkl"),
        5: set("mno"),
        6: set("pqr"),
        7: set("stu"),
        8: set("vwxyz")
    }
    
    # Initialize dictionaries for each partition
    partition_data = {i: {} for i in range(1, 9)}
    
    # Read the existing partial index
    input_file = "./partial_indexes/index_0.json"
    
    print("Reading existing partial index...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} tokens...")
    
    # Distribute tokens into partitions
    for token, doc_data in data.items():
        # Get the first character of the token (lowercase)
        first_char = token[0].lower() if token and token[0].isalpha() else 'z'
        
        # Find which partition this token belongs to
        partition_num = None
        for part_num, char_set in partitions.items():
            if first_char in char_set:
                partition_num = part_num
                break
        
        # If no partition found (for numbers or special chars), put in partition 8
        if partition_num is None:
            partition_num = 8
        
        partition_data[partition_num][token] = doc_data
    
    # Sort each partition alphabetically and save to files
    for part_num in range(1, 9):
        # Sort the tokens alphabetically
        sorted_data = dict(sorted(partition_data[part_num].items()))
        
        # Create output filename
        output_file = f"./partial_indexes/index_{part_num}.json"
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(sorted_data, f, indent=4, separators=(',', ': '), ensure_ascii=False)
        
        print(f"Created {output_file} with {len(sorted_data)} tokens")
    
    # Remove the original index_0.json file
    os.remove(input_file)
    print("Removed original index_0.json file")
    
    print("\n✓ Successfully created 8 alphabetically sorted JSON files!")
    
    # Print summary
    print("\nPartition Summary:")
    for part_num, char_set in partitions.items():
        char_range = ', '.join(sorted(char_set)).upper()
        token_count = len(partition_data[part_num])
        print(f"  index_{part_num}.json: {char_range} ({token_count} tokens)")

if __name__ == "__main__":
    create_alphabetical_partitions() 