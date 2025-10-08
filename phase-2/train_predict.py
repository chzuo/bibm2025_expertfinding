"""
This example demonstrates how to train a Cross-Encoder model for the MS Marco dataset
(https://github.com/microsoft/MSMARCO-Passage-Ranking).
Queries and passages are passed together to a Transformer network. The network returns
a score between 0 and 1 indicating the relevance of the passage for the given query.
The resulting Cross-Encoder can be used for passage re-ranking: For example, you can use
ElasticSearch to retrieve 100 passages for a given query, then pass the query+retrieved
passages to the CrossEncoder for scoring. The results can then be sorted based on the
CrossEncoder scores. This provides significant improvement over out-of-the-box
ElasticSearch/BM25 ranking.
"""
import os
import json
import logging
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Define experimental configurations
configs = [
    {'batch_size': 16, 'max_length': 256},
    {'batch_size': 16, 'max_length': 512},
    {'batch_size': 24, 'max_length': 256}, 
    {'batch_size': 24, 'max_length': 512}
]

# Model and training parameters
model_name = 'pritamdeka/S-PubMedBert-MS-MARCO'
num_epochs = 2
warmup_steps = 500

def load_datasets():
    """Load training, development, and test datasets"""
    train_set = json.load(open('title_docid_dict_train.json', 'r'))
    dev_set = json.load(open('title_docid_dict_dev.json', 'r'))
    corpus = json.load(open('docid_document_dict.json', 'r'))
    
    # Create query dictionary
    queries = {i: title for i, title in enumerate(train_set.keys())}
    
    return train_set, dev_set, corpus, queries

def prepare_dev_samples():
    """Prepare development samples for evaluation"""
    dev_samples = {}
    dev_data = pd.read_csv('dev_bm25_gold.csv')
    
    # Create query ID to title mapping
    dev_qid_title_dict = dict(zip(dev_data['Pid'], dev_data['Claim']))
    
    for qid in dev_data['Pid'].unique():
        dev_samples[qid] = {'query': dev_qid_title_dict[qid], 'positive': set(), 'negative': set()}
        dev_sample_df = dev_data[dev_data['Pid'] == qid]
        
        # Add positive samples
        for pp in dev_sample_df[dev_sample_df['Label'] == 1]['Passage']:
            dev_samples[qid]['positive'].add(pp)
        
        # Add negative samples
        for pp in dev_sample_df[dev_sample_df['Label'] == 0]['Passage']:
            dev_samples[qid]['negative'].add(pp)
            
    return dev_samples

def prepare_train_samples():
    """Prepare training samples"""
    train_data = pd.read_csv('train_bm25l_gold.csv')
    return [InputExample(texts=[query, passage], label=label) 
            for query, passage, label in zip(train_data['Claim'], train_data['Passage'], train_data['Label'])]

def evaluate_model(model, config):
    """Evaluate model performance"""
    dev_data = pd.read_csv('dev_bm25_gold.csv')
    pairs_list = [[x, y] for x, y in zip(dev_data['Claim'], dev_data['Passage'])]
    
    predictions = model.predict(sentences=pairs_list)
    score = sum(predictions) / len(predictions)  # Use average score as evaluation metric
    
    return score

def run_experiments():
    """Run all experimental configurations"""
    best_score = 0
    best_model = None
    best_config = None
    
    # Load data
    train_set, dev_set, corpus, queries = load_datasets()
    dev_samples = prepare_dev_samples()
    train_samples = prepare_train_samples()
    
    for config in configs:
        train_batch_size = config['batch_size']
        max_length = config['max_length']
        
        # Create model save path
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name_safe = model_name.replace("/", "-")
        model_save_path = f'output/{model_name_safe}/epoch{num_epochs}_batch{train_batch_size}_len{max_length}_{current_time}'
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        logging.info(f"Model will be saved to: {model_save_path}")
        logging.info(f"Training configuration: batch_size={train_batch_size}, max_length={max_length}, warmup_steps={warmup_steps}")
        
        # Create model
        model = CrossEncoder(model_name, num_labels=1, max_length=max_length)
        
        # Create data loader
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
        
        # Create evaluator
        evaluator = CERerankingEvaluator(dev_samples, name='train-eval')
        
        # Train model
        model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=10000,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            use_amp=True
        )
        
        # Save latest model
        model.save(model_save_path+'-latest')
        
        # Evaluate current model
        current_score = evaluate_model(model, config)
        
        if current_score > best_score:
            best_score = current_score
            best_model = model
            best_config = config
    
    # Output best results
    logging.info(f"Best configuration: batch_size={best_config['batch_size']}, max_length={best_config['max_length']}")
    logging.info(f"Best score: {best_score}")
    
    # Save best model
    best_model_path = f'output/best_model-{model_name.replace("/", "-")}'
    best_model.save(best_model_path)
    
    return best_model, best_config, best_score

# Execute experiments
best_model, best_config, best_score = run_experiments()

# Evaluate using the best model
logging.info("Performing final evaluation with the best model...")
test_data = pd.read_csv('test_bm25l_gold.csv')
test_pairs = [[x, y] for x, y in zip(test_data['Claim'], test_data['Passage'])]

# Make predictions
predictions = best_model.predict(sentences=test_pairs)

# Calculate evaluation metrics
avg_score = sum(predictions) / len(predictions)
logging.info(f"Best model average score on test set: {avg_score}")

# Save final prediction results
final_results_path = f'step2_results_{model_name.replace("/", "-")}_batch{best_config["batch_size"]}_maxlen{best_config["max_length"]}.txt'
            
with open(final_results_path, 'w') as f:
    for score in predictions:
        f.write(f"{score}\n")

logging.info(f"Final prediction results saved to: {final_results_path}")
