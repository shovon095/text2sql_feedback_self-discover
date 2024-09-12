import logging
import os
import json
import argparse
from typing import List
import torch
import datasets
import transformers
from transformers import (
    set_seed,
    EarlyStoppingCallback,
    LlamaTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Initialize tokenizer and model
model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
model = LlamaForCausalLM.from_pretrained(model_name_or_path)

def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_sql_file(sql_lst: List[str], output_file: str):
    result = {}
    for i, sql in enumerate(sql_lst):
        result[i] = sql

    # Save SQL to the output directory
    new_directory(os.path.dirname(output_file))
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    return result

def tokenize_function(examples):
    # Combine question, db_id, and evidence into a single input
    inputs = [
        f"Question: {q}\nDatabase: {d}\nEvidence: {e}\nGenerate SQL:"
        for q, d, e in zip(examples['question'], examples['db_id'], examples['evidence'])
    ]
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=512)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True, help="Path to the training dataset file.")
    parser.add_argument('--dev_path', type=str, required=True, help="Path to the dev (validation) dataset file.")
    parser.add_argument('--train_db_path', type=str, required=True, help="Path to the training database.")
    parser.add_argument('--dev_db_path', type=str, required=True, help="Path to the dev (validation) database.")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save model and results.")
    parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--per_device_train_batch_size', type=int, default=4, help="Batch size for training.")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument('--save_steps', type=int, default=1000, help="Save checkpoint every X steps.")
    parser.add_argument('--eval_steps', type=int, default=500, help="Evaluate model every X steps.")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load the training and dev (validation) datasets
    train_data = datasets.load_dataset('json', data_files=args.train_path)['train']
    dev_data = datasets.load_dataset('json', data_files=args.dev_path)['train']

    # Tokenize datasets
    tokenized_train_datasets = train_data.map(tokenize_function, batched=True)
    tokenized_dev_datasets = dev_data.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=2,
        remove_unused_columns=False,
        logging_dir='./logs',
        logging_steps=200,
        eval_accumulation_steps=10,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_dev_datasets,
        tokenizer=tokenizer,
    )

    # Train the model
    logger.info("*** Train ***")
    trainer.train()

    # Save the final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Prediction on dev data
    logger.info("*** Predict on Dev ***")
    predictions = trainer.predict(test_dataset=tokenized_dev_datasets)

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)

    # Process predictions
    final_predictions = []
    for pred, db_id in zip(decoded_preds, dev_data['db_id']):
        sql = pred.split('Generate SQL:')[-1].strip()
        sql = sql + f'\t----- bird -----\t{db_id}'
        final_predictions.append(sql)

    # Save predictions to the output directory
    output_name = os.path.join(args.output_dir, f'predict_dev.json')
    generate_sql_file(sql_lst=final_predictions, output_file=output_name)

    logger.info(f'Successfully collected results for dev evaluation')

if __name__ == "__main__":
    main()
