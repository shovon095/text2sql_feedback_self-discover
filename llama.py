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
from transformers import LlamaTokenizer, LlamaForCausalLM

model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
model = LlamaForCausalLM.from_pretrained(model_name_or_path)

# Now you can use the tokenizer and model for predictions


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', type=str, required=True)
    parser.add_argument('--mode', type=str, default='dev')
    parser.add_argument('--db_root_path', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_output_path', type=str, required=True)
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load the BIRD dataset
    eval_data = datasets.load_dataset('json', data_files=args.eval_path)['train']

    # Initialize tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)

    # Tokenize datasets
    def tokenize_function(examples):
        # Combine question, db_id, and evidence into a single input
        inputs = [
            f"Question: {q}\nDatabase: {d}\nEvidence: {e}\nGenerate SQL:"
            for q, d, e in zip(examples['question'], examples['db_id'], examples['evidence'])
        ]
        return tokenizer(inputs, padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = eval_data.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=4,
        prediction_loss_only=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )

    # Prediction
    logger.info("*** Predict ***")
    predictions = trainer.predict(test_dataset=tokenized_datasets)

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