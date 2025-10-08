import json
import argparse

def get_results(results_dict, title_docid_dict):
    """
    Evaluate retrieval performance using multiple metrics.
    
    Args:
        results_dict: Dictionary mapping queries to retrieved document IDs
        title_docid_dict: Dictionary mapping queries to relevant document IDs (ground truth)
        
    Returns:
        None, prints evaluation metrics directly
    """
    # Initialize arrays to store precision and recall metrics
    result_p = [0 for i in range(2)]  # Precision at ranks 1, 10
    result_r = [0 for i in range(2)]  # Recall at ranks 10, 100
    mrr = 0  # Mean Reciprocal Rank
    map_score = 0  # Mean Average Precision
    
    # Evaluate each query
    for title in title_docid_dict:
        # Initialize metrics for current query
        r_k = [0 for i in range(3)]  # Relevance at different k values
        r_p = [0 for i in range(2)]  # Precision at different k values
        r_r = [0 for i in range(2)]  # Recall at different k values
        
        # Get retrieved documents for current query
        candidate_set = results_dict[title]
        
        # Get relevant documents for current query
        similar_list = title_docid_dict[title]
        num_rel = len(similar_list)  # Number of relevant documents
        
        # Track positions of relevant documents in results
        hit_index_all = []
        
        # Check each relevant document
        for pp in similar_list:
            if str(pp) in candidate_set:
                # Record position of relevant document in results (1-indexed)
                hit_index = candidate_set.index(str(pp)) + 1
                hit_index_all.append(hit_index)
                
                # Count relevant documents at different cutoffs
                for i, index in zip([1, 10, 100], range(3)):
                    if str(pp) in candidate_set[:i]:           
                        r_k[index] += 1
        
        # Calculate metrics if at least one relevant document was retrieved
        if len(hit_index_all) != 0:
            # Calculate Average Precision
            ap = 0
            hit_index_all.sort()
            
            # Update MRR with reciprocal of first relevant document position
            mrr += 1 / hit_index_all[0]
            
            # Calculate Average Precision
            for i in range(len(hit_index_all)):
                ap += (i + 1) / hit_index_all[i]
            ap = ap / len(hit_index_all)
            map_score += ap
            
            # Calculate precision and recall metrics
            r_p = [r_k[0], r_k[1] / 10]  # P@1, P@10
            r_r = [r_k[1] / num_rel, r_k[2] / num_rel]  # R@10, R@100
            
            # Accumulate metrics across all queries
            for i in range(2):
                result_r[i] = result_r[i] + r_r[i]
                result_p[i] = result_p[i] + r_p[i]

    # Calculate final metrics (as percentages)
    result_score_r = [round(s / float(len(title_docid_dict)) * 100, 1) for s in result_r]
    result_score_p = [round(s / float(len(title_docid_dict)) * 100, 1) for s in result_p]
    mrr = round(mrr / float(len(title_docid_dict)) * 100, 1)
    map_score = round(map_score / float(len(title_docid_dict)) * 100, 1)
    
    # Print results in a formatted table row
    print(f"{result_score_p[0]} & {result_score_p[1]} & {result_score_r[0]} & {result_score_r[1]} & {mrr} & {map_score}")


def evaluate_dataset(dataset_type, result_file):
    """
    Evaluate retrieval performance for the specified dataset type
    
    Args:
        dataset_type: Dataset type, 'gold' or 'silver'
        result_file: Result file name
    """
    print(f"Evaluating {dataset_type} dataset using {result_file}...")
    
    # Determine file names based on dataset type
    if dataset_type == 'gold':
        title_docid_file = 'title_docid_dict_test.json'
    else:  # silver
        title_docid_file = 'title_docid_dict_test_silver.json'
    
    # Load ground truth data (mapping of queries to relevant document IDs)
    title_docid_dict = json.load(open(title_docid_file, 'r'))
    
    # Load retrieval results (output from a retrieval system like BM25)
    # Format: {query_text: [doc_id1, doc_id2, ...]} where doc_ids are ranked by relevance
    results_dict = json.load(open(result_file, 'r'))
    
    # Evaluate retrieval performance
    get_results(results_dict, title_docid_dict)


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Evaluate Gold and Silver dataset retrieval performance')
    parser.add_argument('--datasets', nargs='+', choices=['gold', 'silver'], 
                        default=['gold'], help='Dataset types to evaluate: gold or silver')
    parser.add_argument('--result_file', required=True,
                        help='Result file containing retrieved document IDs')
    args = parser.parse_args()
    
    # Evaluate each dataset type specified in command line arguments
    for dataset_type in args.datasets:
        print(f"{dataset_type.capitalize()} dataset evaluation results:")
        evaluate_dataset(dataset_type, args.result_file)