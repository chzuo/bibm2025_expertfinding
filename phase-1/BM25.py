import json
import bm25s
import Stemmer  # For stemming words to their root form
import argparse

def process_dataset(dataset_type, model_type):
    """
    Process specified dataset type (gold or silver) with the given model type
    
    Args:
        dataset_type: Dataset type, 'gold' or 'silver'
        model_type: Model type, 'robertson', 'bm25+', or 'bm25l'
    """
    print(f"Processing {dataset_type}-standard dataset using {model_type} model...")
    
    # Determine file names and parameters based on dataset type
    if dataset_type == 'gold':
        title_docid_file = 'title_docid_dict_test.json'
        k_value = 100
    else:  # silver
        title_docid_file = 'title_docid_dict_test_silver.json'
        k_value = 200
    
    # Build result file name
    results_file = f'results_step1_{dataset_type}_{model_type}_dict'
    if dataset_type == 'silver':
        results_file += '_silver'
    results_file += '.json'
    
    # Load data
    title_docid_dict = json.load(open(title_docid_file, 'r'))
    
    # Prepare queries
    queries = [text.lower() for text in title_docid_dict.keys()]
    query_token_ids = bm25s.tokenize(queries, stemmer=stemmer)
    
    # Retrieve results for each query
    results, scores = retriever.retrieve(query_token_ids, k=k_value)
    
    # Create dictionary mapping queries to result document IDs
    results_dict = {}
    for i, query in enumerate(title_docid_dict.keys()):
        results_dict[query] = [str(j) for j in list(results[i])]
    print(f"Number of {dataset_type}-standard queries processed: {len(results_dict)}")
    
    # Save results to JSON file
    json.dump(results_dict, open(results_file, 'w'))

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Process Gold and Silver standard datasets')
parser.add_argument('--datasets', nargs='+', choices=['gold', 'silver', 'both'], 
                    default=['both'], help='Dataset types to process: gold, silver, or both')
parser.add_argument('--model', choices=['bm25', 'bm25+', 'bm25l'], 
                    default='bm25', help='Model type to use: bm25, bm25+, or bm25l')
args = parser.parse_args()

# Load document data
docid_document_dict = json.load(open('docid_document_dict.json', 'r'))

# Create corpus by converting all documents to lowercase
corpus = [text.lower() for text in docid_document_dict.values()]

# Initialize English stemmer
stemmer = Stemmer.Stemmer("english")

# Tokenize corpus with stopword removal and stemming
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

# Initialize BM25 retrieval model
retriever = bm25s.BM25(method=args.model)
retriever.index(corpus_tokens)

# Process datasets based on command line arguments
if 'both' in args.datasets:
    process_dataset('gold', args.model)
    process_dataset('silver', args.model)
else:
    for dataset_type in args.datasets:
        process_dataset(dataset_type, args.model)