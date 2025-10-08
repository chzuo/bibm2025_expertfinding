import torch
import json
from sentence_transformers import SentenceTransformer, util

# Load test data: mapping of claims to relevant document IDs
title_docid_dict = json.load(open('title_docid_dict_test.json', 'r', encoding='utf-8'))

# Load document corpus: mapping of document IDs to document text
docid_doc_dict = json.load(open('docid_document_dict_slim.json', 'r', encoding='utf-8'))

# Create a list of document IDs for indexing
corpus_index = list(docid_doc_dict.keys())[:]
print('Data loaded successfully')

# Initialize the Sentence-BERT model for semantic search
# We use a model fine-tuned on MS MARCO dataset which performs well for retrieval tasks
model_name = 'sentence-transformers/msmarco-MiniLM-L12-cos-v5'
model = SentenceTransformer(model_name)

# Move model to GPU if available for faster processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded and moved to {device}")

# Pre-compute embeddings for all documents in the corpus
print("Computing document embeddings...")
document_embedding = model.encode(list(docid_doc_dict.values())[:])
document_embedding = torch.tensor(document_embedding).to(device)
print(f"Generated embeddings for {len(document_embedding)} documents")

# Parameters for retrieval
top_k = 500  # Number of documents to retrieve per query
count = 0
title_ir_results_dict = dict()
print('Starting retrieval process...')

# Process each query (claim) and retrieve relevant documents
for query in list(title_docid_dict.keys())[:]:
    count += 1
    if count % 500 == 0:
        print(f"Processed {count} queries")
    
    # Encode the query and convert to tensor
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.to(device)
    
    # Calculate cosine similarity between query and all documents
    cos_scores = util.cos_sim(query_embedding, document_embedding)[0]
    
    # Get top-k documents with highest similarity scores
    top_results = torch.topk(cos_scores, k=top_k)
    
    # Map indices to document IDs
    candidate_set = [corpus_index[index] for index in top_results[1]]
    
    # Store results for this query
    title_ir_results_dict[query] = candidate_set

# Extract model name for the output file
model_short_name = model_name.split('/')[-1]

# Save retrieval results to file
output_file = f'results_step1_{model_short_name}_test_slim.json'
json.dump(title_ir_results_dict, open(output_file, 'w', encoding='utf-8'))
print(f"Retrieval completed and results saved to {output_file}")