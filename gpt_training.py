import openai
import sqlite3
import json
import torch
import torch.nn as nn
import os
import backoff
from tqdm import tqdm

# Function to load training data from the JSON file
def load_training_data(train_file):
    with open(train_file, 'r') as f:
        data = json.load(f)
    return data

# Function to get the full path of the database file based on the database name and the folder where databases are stored
def get_database_path(db_name, db_folder):
    return os.path.join(db_folder, db_name, f'{db_name}.sqlite')

# Function to retrieve schema context dynamically from the SQLite database (simulates RAG)
def retrieve_context(db_path, question):
    """
    Retrieve relevant schema or contextual information for the query from a given database.
    This simulates a RAG-style retrieval mechanism.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Example retrieval: Get the schema for the relevant table(s)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Select one or more tables for retrieval as an example
    selected_tables = tables[:min(2, len(tables))]

    context = []
    for table in selected_tables:
        cursor.execute(f"PRAGMA table_info({table[0]})")
        columns = cursor.fetchall()
        schema = f"Table: {table[0]}, Columns: {', '.join([col[1] for col in columns])}"
        context.append(schema)
    
    conn.close()

    # Combine all retrieved context into a single string
    return "\n".join(context)

# Custom loss function with attention weighting based on SQL difficulty
class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, target, difficulty_weights):
        """
        logits: Model predictions
        target: Ground truth labels
        difficulty_weights: Pre-defined weights based on SQL difficulty
        """
        # Apply the weighting to each example in the batch
        loss = self.loss_fn(logits, target)
        weighted_loss = loss * difficulty_weights
        return weighted_loss.mean()

# Function to fine-tune GPT model with RAG-style context and weighted attention
def fine_tune_rag_attention(api_key, train_data, db_folder, engine, epochs=3):
    openai.api_key = api_key

    # Mapping SQL difficulty to weights
    difficulty_map = {
        'simple': 1.0,
        'moderate': 1.5,
        'hard': 2.0
    }

    # Initialize custom loss function
    loss_fn = WeightedLoss()

    # Fine-tuning loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i, example in tqdm(enumerate(train_data)):
            question = example['question']
            sql = example['SQL']
            difficulty = example['difficulty']
            db_name = example['db_id']
            
            # Retrieve the corresponding database path
            db_path = get_database_path(db_name, db_folder)

            # Retrieve schema context dynamically using RAG
            schema_context = retrieve_context(db_path, question)

            # Prepare prompt with schema and question
            prompt = f"-- Schema:\n{schema_context}\n\n-- Question:\n{question}\n\n-- SQL:"

            # Fine-tune the model with backoff for handling API rate limits
            @backoff.on_exception(backoff.expo, openai.error.OpenAIError, max_tries=5)
            def call_openai(prompt):
                response = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.2,
                    stop=[";"]
                )
                return response

            # Get the model's generated SQL
            response = call_openai(prompt)
            generated_sql = response['choices'][0]['text'].strip()

            # Here we simulate model outputs for calculating loss using attention weighting
            logits = torch.randn(4, 10)  # Dummy logits for demonstration
            target = torch.tensor([1, 2, 3, 0])  # Ground truth labels

            # Get difficulty weight for the current SQL
            difficulty_weights = torch.tensor([difficulty_map[difficulty]])

            # Use the custom loss function
            loss = loss_fn(logits, target, difficulty_weights)

            # Print progress
            print(f"Processed example {i+1}/{len(train_data)} - Loss: {loss.item()}")

# Example usage
api_key = "your_openai_api_key"
db_folder = "train_databases"
train_file = 'train.json'

# Load training data
training_data = load_training_data(train_file)

# Fine-tune the model with RAG and attention weighting
fine_tune_rag_attention(api_key, training_data, db_folder, engine="gpt-4", epochs=3)
