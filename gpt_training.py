import openai
import sqlite3
import json
import torch
import torch.nn as nn
import os
import backoff
from tqdm import tqdm
from transformers import GPT2Tokenizer  # Tokenizer is used only for local tokenization, GPT-4-O Mini is accessed via API

# Initialize the tokenizer (GPT-2 tokenizer is used for preprocessing)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add a padding token if it's missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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

# Custom loss function with attention weighting based on SQL complexity
class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, target, complexity_weights):
        # Apply the weighting to each example in the batch
        loss = self.loss_fn(logits, target)
        weighted_loss = loss * complexity_weights
        return weighted_loss.mean()

# Function to calculate complexity weight based on SQL length and presence of joins or other conditions
def calculate_complexity_weight(sql):
    base_weight = 1.0
    # Calculate weight based on SQL length
    sql_length = len(sql.split())
    length_weight = sql_length / 100.0  # Example scaling, can be adjusted

    # Check for the presence of JOIN, GROUP BY, HAVING, etc.
    complexity_increase = 0.0
    if 'JOIN' in sql:
        complexity_increase += 1.0
    if 'GROUP BY' in sql:
        complexity_increase += 0.5
    if 'HAVING' in sql:
        complexity_increase += 0.5
    if 'ORDER BY' in sql:
        complexity_increase += 0.5
    
    # Final complexity weight
    complexity_weight = base_weight + length_weight + complexity_increase
    return complexity_weight

# Function to provide feedback on incorrect SQL and log it
def give_feedback(generated_sql, correct_sql):
    feedback = ""
    if generated_sql != correct_sql:
        feedback = f"Your generated SQL: {generated_sql} was incorrect. Correct SQL: {correct_sql}."
    return feedback

# Function to fine-tune GPT-4-O Mini with RAG-style context, CoT prompting, and feedback-based training
def fine_tune_with_cot_and_complexity(api_key, train_data, db_folder, engine, epochs=3):
    openai.api_key = api_key

    # Initialize custom loss function
    loss_fn = WeightedLoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i, example in tqdm(enumerate(train_data)):
            question = example['question']
            correct_sql = example['SQL']  # Ground truth SQL
            evidence = example['evidence']  # Evidence to explain query logic
            db_name = example['db_id']
            
            # Retrieve the corresponding database path and schema
            db_path = get_database_path(db_name, db_folder)
            schema_context = retrieve_context(db_path, question)

            # Create Chain of Thought prompt with schema, question, evidence, and reasoning steps
            prompt = f"""
            -- Schema:
            {schema_context}

            -- Question:
            {question}

            -- Evidence:
            {evidence}

            -- Let's think step by step:
            1. Identify the relevant tables and columns from the schema.
            2. Use the evidence to clarify the relationships between the natural language and SQL query components.
            3. Generate the final SQL query based on the filtered data.

            -- SQL:
            """

            # Call GPT-4-O Mini API to generate SQL with Chain of Thought prompting
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

            # Get the model's generated SQL from the API
            response = call_openai(prompt)
            generated_sql = response['choices'][0]['message']['content'].strip()

            # Tokenize the generated SQL
            input_tokens = tokenizer(generated_sql, return_tensors='pt', padding=True, truncation=True).input_ids.to(torch.float32)

            # Tokenize the ground truth SQL (Target)
            target_tokens = tokenizer(correct_sql, return_tensors='pt', padding=True, truncation=True).input_ids.to(torch.int64)

            # Flatten the logits and target to match the shape for CrossEntropyLoss
            logits = input_tokens.view(-1, input_tokens.size(-1))
            target = target_tokens.view(-1)

            # Ensure that the shapes match
            assert logits.size(0) == target.size(0), f"Logits and target size mismatch: {logits.size(0)} vs {target.size(0)}"

            # Calculate complexity weight based on SQL length and conditions
            complexity_weight = calculate_complexity_weight(correct_sql)

            # Use the custom loss function
            loss = loss_fn(logits, target, torch.tensor([complexity_weight], dtype=torch.float32))

            # Print progress
            print(f"Processed example {i+1}/{len(train_data)} - Loss: {loss.item()}")

            # Provide feedback if SQL is incorrect
            feedback = give_feedback(generated_sql, correct_sql)
            if feedback:
                print(feedback)

# Example usage
api_key = "your_openai_api_key"
db_folder = "train_databases"
train_file = 'train.json'

# Load training data
training_data = load_training_data(train_file)

# Fine-tune the model with RAG, CoT prompting, and complexity-based weights
fine_tune_with_cot_and_complexity(api_key, training_data, db_folder, engine="gpt-4o-mini", epochs=15)

