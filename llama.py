import logging
import os
import json
import argparse
from typing import List
import torch
import datasets
import transformers
from transformers import (
    HfArgumentParser,
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

def generate_sql_file(sql_lst: List[str], output_path: str = None):
    result = {}
    for i, sql in enumerate(sql_lst):
        result[i] = sql
    
    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)
        with open(output_path, 'w') as f:
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
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--eval_path', type=str, required=True)
    parser.add_argument('--mode', type=str, default='dev')
    parser.add_argument('--db_root_path', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_output_path', type=str, required=True)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=500)
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load the training and evaluation datasets
    train_data = datasets.load_dataset('json', data_files=args.train_path)['train']
    eval_data = datasets.load_dataset('json', data_files=args.eval_path)['train']

    # Tokenize datasets
    tokenized_train_datasets = train_data.map(tokenize_function, batched=True)
    tokenized_eval_datasets = eval_data.map(tokenize_function, batched=True)

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
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_eval_datasets,
        tokenizer=tokenizer,
    )

    # Train the model
    logger.info("*** Train ***")
    trainer.train()

    # Save the final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Prediction
    logger.info("*** Predict ***")
    predictions = trainer.predict(test_dataset=tokenized_eval_datasets)

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)

    # Process predictions
    final_predictions = []
    for pred, db_id in zip(decoded_preds, eval_data['db_id']):
        sql = pred.split('Generate SQL:')[-1].strip()
        sql = sql + f'\t----- bird -----\t{db_id}'
        final_predictions.append(sql)

    # Save predictions
    output_name = os.path.join(args.data_output_path, f'predict_{args.mode}.json')
    generate_sql_file(sql_lst=final_predictions, output_path=output_name)

    logger.info(f'Successfully collected results for {args.mode} evaluation')

if __name__ == "__main__":
    main()
