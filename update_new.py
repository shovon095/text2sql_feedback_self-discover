#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
from typing import Dict, List, Any, Tuple
import openai
import backoff
import sqlparse
from tqdm import tqdm
import spacy
import re
openai.debug = True

# Initialize SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

import logging
from contextlib import contextmanager
from threading import Timer

logging.basicConfig(level=logging.INFO)

import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

@contextmanager
def time_limit(seconds):
    """
    Context manager to limit the execution time of a block of code.
    It raises a TimeoutException if the block takes longer than 'seconds'.
    """
    # Register the signal function handler
    signal.signal(signal.SIGALRM, timeout_handler)
    # Schedule the signal after `seconds` seconds
    signal.alarm(seconds)
    
    try:
        yield
    except TimeoutException:
        logging.error("Operation timed out")
        raise
    finally:
        # Cancel the alarm
        signal.alarm(0)

def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def question_package(data_json, knowledge=False):
    question_list = []
    for data in data_json:
        question_list.append(data['question'])

    return question_list

def knowledge_package(data_json, knowledge=False):
    knowledge_list = []
    for data in data_json:
        knowledge_list.append(data['evidence'])

    return knowledge_list

def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    for i, data in enumerate(datasets):
        question_list.append(data['question'])
        cur_db_path = f"{db_root_path}/{data['db_id']}/{data['db_id']}.sqlite"  # Adjusted here
        db_path_list.append(cur_db_path)
        knowledge_list.append(data['evidence'])
    
    return question_list, db_path_list, knowledge_list


def clean_sql_query(sql: str) -> str:
    sql = re.sub(r'^```sql|```$', '', sql).strip()
    if not re.match(r'^(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b', sql, re.IGNORECASE):
        sql = 'SELECT ' + sql
    return sql

def get_db_schemas(db_path: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
        cursor = conn.cursor()
        
        # Get all the table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schemas = {}
        all_columns = {}
        
        for table in tables:
            table_name = table[0]
            
            # Get the CREATE statement for each table
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
            schema_result = cursor.fetchone()
            if schema_result:
                schemas[table_name] = schema_result[0]
                
                # Get the columns for the table
                cursor.execute(f'PRAGMA table_info("[{table_name}]")')
                columns = cursor.fetchall()
                all_columns[table_name] = [col[1] for col in columns]  # Extract column names

        return schemas, all_columns

def generate_schema_prompt(db_path, relevant_tables=None, relevant_columns=None, attention_weights=None, num_rows=None):
    """
    Generates the schema prompt with CREATE statements, example rows, and attention weights
    for each table and column in the database.
    
    :param db_path: Path to the database file
    :param relevant_tables: List of relevant tables based on question entities
    :param relevant_columns: List of relevant columns based on question entities
    :param attention_weights: Dictionary of attention weights for tables and columns
    :param num_rows: Number of example rows to include in the prompt
    :return: A string representing the schema prompt with attention weights
    """
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}

    # Loop through tables in the schema
    for table in tables:
        if table[0] == 'sqlite_sequence':
            continue
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table[0]}'")
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt

        # Add attention weight to the table (if applicable)
        if relevant_tables and attention_weights:
            table_weight = attention_weights.get(table[0], 0.5)  # Default weight of 0.5 for non-relevant tables
            create_prompt += f" -- Attention Weight: {table_weight}"
        else:
            table_weight = 0.5  # Default attention weight for non-relevant tables

        if num_rows:
            cur_table = table[0]
            if cur_table in ['order', 'by', 'group']:
                cur_table = f"`{cur_table}`"
            cursor.execute(f"SELECT * FROM {cur_table} LIMIT {num_rows}")
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = f"/* \n {num_rows} example rows: \n SELECT * FROM {cur_table} LIMIT {num_rows}; \n {rows_prompt} \n */"
            schemas[table[0]] = f"{create_prompt} \n {verbose_prompt}"

        # Add attention weight to columns (if applicable)
        if relevant_columns and attention_weights:
            for column in relevant_columns:
                if column[0] == table[0]:  # Check if the column belongs to the current table
                    column_weight = attention_weights.get(column[1], 0.5)  # Default weight of 0.5 for non-relevant columns
                    create_prompt += f"\n    {column[1]} -- Attention Weight: {column_weight}"

    # Format and combine all table schemas into a single prompt
    for k, v in schemas.items():
        full_schema_prompt_list.append(v)

    schema_prompt = "\n\n".join(full_schema_prompt_list)
    return schema_prompt


def generate_sql_file(sql_lst, output_path=None):
    result = {}
    for i, sql in enumerate(sql_lst):
        result[i] = sql
    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)
        json.dump(result, open(output_path, 'w'), indent=4)
    return result

def generate_comment_prompt(question, knowledge=None):
    """
    Generates the comment prompt for SQL query generation based on the question and optional external knowledge.
    """
    pattern_prompt_no_kg = "-- Using valid SQLite, answer the following questions for the tables provided above."
    pattern_prompt_kg = "-- Using valid SQLite and understanding External Knowledge, answer the following questions for the tables provided above."
    question_prompt = f"-- {question}"
    knowledge_prompt = f"-- External Knowledge: {knowledge}" if knowledge else ""

    return pattern_prompt_kg + '\n' + question_prompt if knowledge else pattern_prompt_no_kg + '\n' + question_prompt


def cot_wizard():
    """
    Generates the Chain-of-Thought prompting.
    """
    cot = "\nGenerate the SQL after thinking step by step: "
    return cot

def nice_look_table(column_names: list, values: list):
    """
    Create a nicely formatted table for example rows in the schema.
    """
    rows = []
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + '\n' + rows
    return final_output

def extract_entities_and_relationships(question: str):
    """
    Extracts key entities and their relationships from the question using SpaCy.
    """
    doc = nlp(question)
    entities = [ent.text for ent in doc.ents]  # Extract named entities
    relationships = []
    for token in doc:
        if token.dep_ in ('nsubj', 'dobj'):  # Subject-object dependencies
            relationships.append((token.text, token.head.text))
    return entities, relationships


def map_entities_to_schema(entities: List[str], relationships: List[tuple], schema: Dict[str, str]):
    """
    Maps extracted entities and their relationships to schema tables and columns.
    Returns a list of relevant tables, columns, and attention weights.
    
    :param entities: List of entities extracted from the question.
    :param relationships: List of tuples representing relationships between entities.
    :param schema: Dictionary representing the schema (tables and columns).
    :return: Tuple of relevant tables, relevant columns, and attention weights.
    """
    relevant_tables = []
    relevant_columns = []
    attention_weights = {}

    # Map entities to tables and columns
    for entity in entities:
        entity_lower = entity.lower()
        for table, create_stmt in schema.items():
            if entity_lower in table.lower():
                relevant_tables.append(table)
                attention_weights[table] = attention_weights.get(table, 1.0) + 1.0
            else:
                # Look for columns containing the entity
                for column in create_stmt.split('(')[1].split(')')[0].split(','):
                    column_name = column.strip().split()[0]
                    if entity_lower in column_name.lower():
                        relevant_columns.append((table, column_name))
                        attention_weights[column_name] = attention_weights.get(column_name, 1.0) + 1.0

    # Handle relationships (if necessary)
    for rel in relationships:
        subject, obj = rel  # Extract subject and object of the relationship
        subject_lower, obj_lower = subject.lower(), obj.lower()

        # If both subject and object are found in schema, increase attention weight for related tables/columns
        for table, create_stmt in schema.items():
            if subject_lower in table.lower() or obj_lower in table.lower():
                attention_weights[table] = attention_weights.get(table, 1.0) + 0.5  # Increase weight for table

            for column in create_stmt.split('(')[1].split(')')[0].split(','):
                column_name = column.strip().split()[0]
                if subject_lower in column_name.lower() or obj_lower in column_name.lower():
                    attention_weights[column_name] = attention_weights.get(column_name, 1.0) + 0.5  # Increase weight for columns

    return relevant_tables, relevant_columns, attention_weights

def create_attention_mask(schema: Dict[str, str], relevant_tables: List[str], relevant_columns: List[tuple], attention_weights: Dict[str, float]):
    """
    Creates an attention mask that assigns higher weights to relevant schema elements.
    Returns a formatted string for inclusion in the prompt.
    """
    attention_info = {}

    for table, create_stmt in schema.items():
        table_weight = attention_weights.get(table, 0.5)
        attention_info[table] = table_weight

        columns = create_stmt.split('(')[1].split(')')[0].split(',')
        for column in columns:
            column_name = column.strip().split()[0]
            full_column = f"{table}.{column_name}"
            if (table, column_name) in relevant_columns:
                attention_info[full_column] = attention_weights.get(column_name, 1.0)
            else:
                attention_info[full_column] = 0.3  # Lower weight for non-relevant columns

    attention_mask_str = "### Schema Attention Weights ###\n"
    for element, weight in attention_info.items():
        attention_mask_str += f"{element}: {weight}\n"

    return attention_mask_str

def few_shot():
    ini_table = "CREATE TABLE singer\n(\n    singer_id         TEXT not null\n        primary key,\n    nation       TEXT  not null,\n    sname       TEXT null,\n    dname       TEXT null,\n    cname       TEXT null,\n    age    INTEGER         not null,\n    year  INTEGER          not null,\n    birth_year  INTEGER          null,\n    salary  REAL          null,\n    city TEXT          null,\n    phone_number   INTEGER          null,\n--     tax   REAL      null,\n)"
    
    ini_prompt = "-- External Knowledge: age = year - birth_year;\n-- Using valid SQLite and understanding External Knowledge, answer the following questions for the tables provided above.\n-- How many singers in USA who are older than 27?"
    
    # Direct final SQL query without step-by-step explanation
    final_sql = "SELECT COUNT(*) FROM singer WHERE year - birth_year > 27 AND nation = 'US';"
    
    one_shot_demo = ini_table + '\n' + ini_prompt + '\n' + final_sql
    
    return one_shot_demo

def few_shot_no_kg():
    ini_table = "CREATE TABLE singer\n(\n    singer_id         TEXT not null\n        primary key,\n    nation       TEXT  not null,\n    sname       TEXT null,\n    dname       TEXT null,\n    cname       TEXT null,\n    age    INTEGER         not null,\n    year  INTEGER          not null,\n    age  INTEGER          null,\n    salary  REAL          null,\n    city TEXT          null,\n    phone_number   INTEGER          null,\n--     tax   REAL      null,\n)"
    
    ini_prompt = "-- Using valid SQLite and understanding External Knowledge, answer the following questions for the tables provided above.\n-- How many singers in USA who are older than 27?"
    
    # Direct final SQL query without step-by-step explanation
    final_sql = "SELECT COUNT(*) FROM singer WHERE age > 27 AND nation = 'US';"
    
    one_shot_demo = ini_table + '\n' + ini_prompt + '\n' + final_sql
    
    return one_shot_demo

def generate_difficulty_based_prompt(db_path, question, knowledge=None):
    """
    Generates a prompt incorporating schema details and the question.
    Uses few-shot examples and schema without Chain-of-Thought (CoT).
    """
    # Fetch schema directly without caching
    schema_dict, all_columns = get_db_schemas(db_path)

    # Pre-process question to extract entities and relationships
    entities, relationships = extract_entities_and_relationships(question)

    # Map entities to schema elements
    relevant_tables, relevant_columns, attention_weights = map_entities_to_schema(entities, relationships, schema_dict)

    # Generate schema prompt
    schema = generate_schema_prompt(db_path, relevant_tables, relevant_columns, attention_weights)

    # Generate attention mask
    attention_mask = create_attention_mask(schema_dict, relevant_tables, relevant_columns, attention_weights)

    # Generate few-shot examples based on whether external knowledge is used
    if knowledge:
        few_shot_example = few_shot()  # With external knowledge
    else:
        few_shot_example = few_shot_no_kg()  # Without external knowledge

    # Generate the main question prompt
    comment_prompt = generate_comment_prompt(question, knowledge)

    final_instruction = "Write the SQL query to answer the question, prioritizing schema elements based on the attention weights provided."

    # Combine all elements into the final prompt
    prompt = (
        f"{few_shot_example}\n\n"  # Add the few-shot example
        f"{schema}\n\n"
        f"{comment_prompt}\n\n"
        f"{attention_mask}\n\n"
        f"{final_instruction}\n"
    )

    return prompt

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)

def connect_gpt(engine, prompt, max_tokens, temperature, stop):
    try:
        # Call OpenAI's ChatCompletion.create() directly
        result = openai.ChatCompletion.create(
            model=engine,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": "You are an SQL expert. Respond only with valid SQL queries, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            stop=stop  # Ensure stop is a list of strings
        )

        # Extract and return the raw SQL from the response
        raw_sql = result['choices'][0]['message']['content'].strip()
        return raw_sql

    except Exception as e:
        # Return the error message for debugging
        return f'error: {str(e)}'


import time
@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=2)
def retry_connect_gpt(*args, **kwargs):
    return connect_gpt(*args, **kwargs)

def collect_response_from_gpt(db_path_list, question_list, api_key, engine, knowledge_list=None):
    openai.api_key = api_key
    response_list = []

    for i, question in enumerate(question_list):
        logging.info(f"Processing {i + 1}/{len(question_list)}: {question}")
        start_time = time.time()

        try:
            with time_limit(120):  # 2-minute timeout for the entire question processing
                # Fetch schema directly without caching
                schema_dict, all_columns = get_db_schemas(db_path_list[i])
                logging.info(f"Schema loaded for question {i + 1}")

                # Generate the prompt
                cur_prompt = generate_difficulty_based_prompt(
                    db_path=db_path_list[i],
                    question=question,
                    knowledge=knowledge_list[i] if knowledge_list else None
                )

                # Call GPT once
                result = retry_connect_gpt(engine=engine, prompt=cur_prompt, max_tokens=256, temperature=0.5, stop=[';'])
                sql = clean_sql_query(result.strip())

                # Append the generated SQL to response_list
                response_list.append(sql)

        except TimeoutException:
            logging.error(f"Processing timed out for question: {question}")
            response_list.append(None)  # Append None if the request times out
            continue  # Skip to the next question if there's a timeout

        except Exception as e:
            logging.error(f"Error processing question: {question}")
            logging.error(f"Error details: {str(e)}")
            response_list.append(None)  # Append None if there's an error
            continue  # Skip to the next question if there's an exception

        end_time = time.time()
        logging.info(f"Question {i + 1} processed in {end_time - start_time:.2f} seconds")

    return response_list

if __name__ == '__main__':
    # Argument parsing
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--eval_path', type=str, default='')
    args_parser.add_argument('--mode', type=str, default='dev')
    args_parser.add_argument('--use_knowledge', type=str, default='False')
    args_parser.add_argument('--db_root_path', type=str, default='')
    args_parser.add_argument('--api_key', type=str, required=True)
    args_parser.add_argument('--engine', type=str, required=True, default='code-davinci-002')
    args_parser.add_argument('--data_output_path', type=str)
    args_parser.add_argument('--chain_of_thought', type=str, default='False', help="Enable chain-of-thought prompting (True/False)")  # New argument

    args = args_parser.parse_args()

    # Load evaluation data
    eval_data = json.load(open(args.eval_path, 'r'))

    # Decouple question and schema
    question_list, db_path_list, knowledge_list = decouple_question_schema(datasets=eval_data, db_root_path=args.db_root_path)
    if args.use_knowledge == 'True':
        responses = collect_response_from_gpt(
            db_path_list=db_path_list,
            question_list=question_list,
            api_key=args.api_key,
            engine=args.engine,
            #force_reload=True  # Enable force reload
    )
    
    else:
        responses = collect_response_from_gpt(
        db_path_list=db_path_list,
        question_list=question_list,
        api_key=args.api_key,
        engine=args.engine,
        #force_reload=True 
    )
    response_list = [clean_sql_query(sql) for sql in responses]  # Clean all SQL queries here
    output_name = args.data_output_path + 'predict_' + args.mode + '.json'
    generate_sql_file(sql_lst=response_list, output_path=output_name)
    print(f'Successfully collected results from {args.engine} for {args.mode} evaluation.')

