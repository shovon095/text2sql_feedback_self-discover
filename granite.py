#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def new_directory(path):  
    if not os.path.exists(path):  
        os.makedirs(path)  

def get_db_schemas(bench_root: str, db_name: str) -> dict:
    asdf = 'database' if bench_root == 'spider' else 'databases'
    with sqlite3.connect(f'file:{bench_root}/{asdf}/{db_name}/{db_name}.sqlite?mode=ro', uri=True) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schemas = {}
        for table in tables:
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
            schemas[table[0]] = cursor.fetchone()[0]
        return schemas

def nice_look_table(column_names: list, values: list):
    rows = []
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + '\n' + rows
    return final_output

def generate_schema_prompt(db_path, num_rows=None):
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}
    for table in tables:
        if table == 'sqlite_sequence':
            continue
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt
        if num_rows:
            cur_table = table[0]
            if cur_table in ['order', 'by', 'group']:
                cur_table = "`{}`".format(cur_table)

            cursor.execute("SELECT * FROM {} LIMIT {}".format(cur_table, num_rows))
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(num_rows, cur_table, num_rows, rows_prompt)
            schemas[table[0]] = "{} \n {}".format(create_prompt, verbose_prompt)

    for k, v in schemas.items():
        full_schema_prompt_list.append(v)

    schema_prompt = "\n\n".join(full_schema_prompt_list)

    return schema_prompt

def generate_comment_prompt(question, knowledge=None):
    pattern_prompt_no_kg = "-- Using valid SQLite, answer the following questions for the tables provided above."
    pattern_prompt_kg = "-- Using valid SQLite and understanding External Knowledge, answer the following questions for the tables provided above."
    question_prompt = "-- {}".format(question)
    knowledge_prompt = "-- External Knowledge: {}".format(knowledge) if knowledge else ""

    if not knowledge:
        result_prompt = pattern_prompt_no_kg + '\n' + question_prompt
    else:
        result_prompt = knowledge_prompt + '\n' + pattern_prompt_kg + '\n' + question_prompt

    return result_prompt

def cot_wizard():
    return "\nGenerate the SQL after thinking step by step: "

def generate_combined_prompts_one(db_path, question, knowledge=None):
    schema_prompt = generate_schema_prompt(db_path, num_rows=None)
    comment_prompt = generate_comment_prompt(question, knowledge)

    combined_prompts = schema_prompt + '\n\n' + comment_prompt + cot_wizard() + '\nSELECT '
    return combined_prompts

def generate_sql_with_granite(model, tokenizer, prompt, device):
    input_tokens = tokenizer(prompt, return_tensors="pt")
    input_tokens = {k: v.to(device) for k, v in input_tokens.items()}
    output_tokens = model.generate(**input_tokens)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text

def collect_response_from_granite(model, tokenizer, db_path_list, question_list, device, knowledge_list=None):
    responses_dict = {}
    response_list = []
    
    for i, question in tqdm(enumerate(question_list)):
        print('--------------------- processing {}th question ---------------------'.format(i))
        print('the question is: {}'.format(question))
        
        if knowledge_list:
            cur_prompt = generate_combined_prompts_one(db_path=db_path_list[i], question=question, knowledge=knowledge_list[i])
        else:
            cur_prompt = generate_combined_prompts_one(db_path=db_path_list[i], question=question)
        
        sql = generate_sql_with_granite(model=model, tokenizer=tokenizer, prompt=cur_prompt, device=device)
        
        db_id = db_path_list[i].split('/')[-1].split('.sqlite')[0]
        sql = sql + '\t----- bird -----\t' + db_id
        response_list.append(sql)

    return response_list

def decouple_question_schema(datasets, db_root_path):
    question_list, db_path_list, knowledge_list = [], [], []
    for i, data in datasets:
        question_list.append(data['question'])
        cur_db_path = os.path.join(db_root_path, data['db_id'], f"{data['db_id']}.sqlite")
        db_path_list.append(cur_db_path)
        knowledge_list.append(data.get('evidence'))
    
    return question_list, db_path_list, knowledge_list

def generate_sql_file(sql_lst, output_path=None):
    result = {i: sql for i, sql in enumerate(sql_lst)}
    
    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
    
    return result    

def prepare_dataset_for_dolomite(data_path, db_root_path, output_dir):
    with open(data_path, 'r') as f:
        eval_data = json.load(f)

    train_data = []
    for data in eval_data:
        question = data['question']
        db_path = os.path.join(db_root_path, data['db_id'], f"{data['db_id']}.sqlite")

        # Debugging step: Print the db_path
        print(f"Attempting to open database: {db_path}")
        
        # Check if the database file exists
        if not os.path.exists(db_path):
            print(f"Error: Database file not found at {db_path}")
            continue  # Skip to the next item
        
        schema_prompt = generate_schema_prompt(db_path, num_rows=None)
        prompt = schema_prompt + '\n\n' + question
        
        # Check if 'SQL' key exists
        if 'SQL' in data:
            sql = data['SQL']  # Correct key for the SQL query
        else:
            print(f"Warning: 'SQL' key not found in entry with question: {question}")
            sql = "NO_SQL_PROVIDED"  # Handle missing SQL keys

        train_data.append({"prompt": prompt, "completion": sql})
    
    os.makedirs(output_dir, exist_ok=True)
    train_data_path = os.path.join(output_dir, "train_data.jsonl")
    with open(train_data_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    
    return train_data_path

def fine_tune_with_dolomite(train_data_path, model_name_or_path, output_dir):
    # Check if dolomite-engine directory exists
    if not os.path.exists("dolomite-engine"):
        # Clone Dolomite Engine repo only if it doesn't exist
        subprocess.run(["git", "clone", "https://github.com/ibm-granite/dolomite-engine.git"])
    
    # Change directory to dolomite-engine
    os.chdir("dolomite-engine")
    
    # Prepare the training config
    config_path = "configs/granite-example/training.yml"
    with open(config_path, 'r') as file:
        config = file.read()

    config = config.replace("path/to/your/data", train_data_path)
    config = config.replace("output/model/path", output_dir)
    config = config.replace("model/to/fine/tune", model_name_or_path)

    with open(config_path, 'w') as file:
        file.write(config)

    # Run the fine-tuning script using bash instead of sh
    subprocess.run(["bash", "scripts/finetune.sh", config_path])

    # Export the fine-tuned model using bash
    subprocess.run(["bash", "scripts/export.sh", "configs/granite-example/export.yml"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune IBM Granite model on BIRD dataset using Dolomite Engine and evaluate on dev data.')
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training data.')
    parser.add_argument('--eval_data_path', type=str, required=True, help='Path to the evaluation data.')
    parser.add_argument('--db_root_path', type=str, required=True, help='Path to the database root.')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Model name or path to fine-tune.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the fine-tuned model.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation.')
    parser.add_argument('--chain_of_thought', type=str, default='False', help='Whether to use Chain of Thought prompting.')
    parser.add_argument('--use_knowledge', type=str, default='False', help='Whether to use external knowledge.')

    args = parser.parse_args()

    # Step 1: Prepare the dataset for Dolomite Engine
    train_data_path = prepare_dataset_for_dolomite(args.train_data_path, args.db_root_path, args.output_dir)

    # Step 2: Fine-tune the model using Dolomite Engine
    fine_tune_with_dolomite(train_data_path, args.model_name_or_path, args.output_dir)

    # Step 3: Evaluate the fine-tuned model on dev data
    eval_data = json.load(open(args.eval_data_path, 'r'))
    question_list, db_path_list, knowledge_list = decouple_question_schema(datasets=eval_data, db_root_path=args.db_root_path)
    
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    model = AutoModelForCausalLM.from_pretrained(args.output_dir).to(args.device)

    responses = collect_response_from_granite(model=model, tokenizer=tokenizer, db_path_list=db_path_list, question_list=question_list, device=args.device, knowledge_list=knowledge_list if args.use_knowledge == 'True' else None)
    
    output_name = args.output_dir + ('/predict_' + args.chain_of_thought + '_cot.json' if args.chain_of_thought == 'True' else '/predict_' + args.chain_of_thought + '.json')
    generate_sql_file(sql_lst=responses, output_path=output_name)

    print('Successfully fine-tuned and evaluated the Granite model using Dolomite Engine.')
