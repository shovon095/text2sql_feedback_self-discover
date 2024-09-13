import openai
import sqlite3
import json
import torch
import torch.nn as nn
import os
import backoff
from tqdm import tqdm
from transformers import GPT2Tokenizer

# Initialize the tokenizer (GPT-2 tokenizer is used here as an example, adjust based on GPT-4-O Mini)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

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
        # Apply the weighting to each example in the batch
        loss = self.loss_fn(logits, target)
        weighted_loss = loss * difficulty_weights
        return weighted_loss.mean()

# Function to provide feedback on incorrect SQL and log it
def give_feedback(generated_sql, correct_sql):
    feedback = ""
    if generated_sql != correct_sql:
        feedback = f"Your generated SQL: {generated_sql} was incorrect. Correct SQL: {correct_sql}."
    return feedback

# Function to fine-tune GPT-4-O Mini with RAG-style context, CoT prompting, and feedback-based training
def fine_tune_with_cot_and_feedback(api_key, train_data, db_folder, engine, epochs=3):
    openai.api_key = api_key

    # Mapping SQL difficulty to weights
    difficulty_map = {
        'simple': 1.0,
        'moderate': 1.5,
        'hard': 2.0
    }

    # Initialize custom loss function
    loss_fn = WeightedLoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i, example in tqdm(enumerate(train_data)):
            question = example['question']
            correct_sql = example['SQL']  # Ground truth SQL
            difficulty = example['difficulty']
            db_name = example['db_id']
            
            # Retrieve the corresponding database path and schema
            db_path = get_database_path(db_name, db_folder)
            schema_context = retrieve_context(db_path, question)

            # Create Chain of Thought prompt with schema, question, and reasoning steps
            prompt = f"""
            -- Schema:
            {schema_context}

            -- Question:
            {question}

            -- Let's think step by step:
            1. Identify the relevant tables and columns from the schema.
            2. Filter the data based on the conditions in the question.
            3. Generate the final SQL query based on the filtered data.

            -- SQL:
            """

            # Call OpenAI API to generate SQL with Chain of Thought prompting
            @backoff.on_exception(backoff.expo, openai.error.OpenAIError, max_tries=5)
            def call_openai(prompt):
                response = openai.ChatCompletion.create(
                    model=engine,
                    messages=[
                        {"role": "system", "content": "You are an SQL expert. Respond with only valid SQL queries."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.2,
                    stop=[";"]
                )
                return response

            # Get the model's generated SQL from ChatCompletion
            response = call_openai(prompt)
            generated_sql = response['choices'][0]['message']['content'].strip()

            # Tokenize the generated SQL
            input_tokens = tokenizer(generated_sql, return_tensors='pt', padding=True, truncation=True)
            logits = input_tokens.input_ids  # In this context, treat input tokens as "logits" for comparison

            # Tokenize the ground truth SQL (Target)
            target_tokens = tokenizer(correct_sql, return_tensors='pt', padding=True, truncation=True)
            target = target_tokens.input_ids

            # Get difficulty weight for the current SQL
            difficulty_weights = torch.tensor([difficulty_map[difficulty]])

            # Use the custom loss function
            loss = loss_fn(logits, target, difficulty_weights)

            # Print progress
            print(f"Processed example {i+1}/{len(train_data)} - Loss: {loss.item()}")

            # Provide feedback if SQL is incorrect
            feedback = give_feedback(generated_sql, correct_sql)
            if feedback:
                print(feedback)
                # Optionally log this for future fine-tuning or retraining

# Example usage
api_key = "your_openai_api_key"
db_folder = "train_databases"
train_file = 'train.json'

# Load training data
training_data = load_training_data(train_file)

# Fine-tune the model with RAG, CoT prompting, and feedback-based training
fine_tune_with_cot_and_feedback(api_key, training_data, db_folder, engine="gpt-4o-mini", epochs=15)

