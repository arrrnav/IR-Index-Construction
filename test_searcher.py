#!/usr/bin/env python3

from searcher import Searcher

def test_searcher():
    print("=== Testing Searcher ===")
    
    try:
        # Try to create a searcher instance
        print("1. Creating searcher instance...")
        searcher = Searcher()
        print("✓ Searcher created successfully!")
        
        # Test query preprocessing
        print("\n2. Testing query preprocessing...")
        test_query = "machine learning algorithms"
        processed = searcher._preprocess_query(test_query)
        print(f"Original: '{test_query}'")
        print(f"Processed: {processed}")
        
        # Test partition assignment
        print("\n3. Testing partition assignment...")
        for token in processed:
            partition = searcher._get_token_partition(token)
            print(f"Token '{token}' -> Partition {partition}")
        
        # Test a simple search
        print("\n4. Testing search...")
        results = searcher.search("machine learning", k=5)
        print(f"Search results: {results}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This means we need to create the index files first!")
        return False
    
    return True

if __name__ == "__main__":
    test_searcher() 