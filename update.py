#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
from typing import Dict, List, Any
import openai
import backoff
import sqlparse
from tqdm import tqdm
import spacy


openai.debug = True

# Initialize SpaCy NLP model
nlp = spacy.load("en_core_web_sm")


def new_directory(path):  
    if not os.path.exists(path):  
        os.makedirs(path)  
        
def get_db_schemas(bench_root: str, db_name: str) -> Dict[str, str]:
    asdf = 'database' if bench_root == 'spider' else 'databases'
    db_path = os.path.join(bench_root, asdf, db_name, db_name + '.sqlite')
    
    # Check if the file exists before attempting to connect
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file does not exist: {db_path}")
    
    print(f"Opening database at: {db_path}")  # Debugging statement
    
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schemas = {}
        for table in tables:
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table[0]}'")
            schemas[table[0]] = cursor.fetchone()[0]
        return schemas


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

def execute_and_validate_query(db_path: str, sql_query: str, question: str) -> Dict[str, Any]:
    """
    Executes the generated SQL query and validates its correctness based on the number of results,
    potential issues, and comparison with expected results.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        
        validation_result = {
            "execution_success": True,
            "row_count": len(results),
            "column_count": len(column_names),
            "sample_data": results[:5] if results else [],
            "column_names": column_names,
            "potential_issues": []
        }
        
        # Check for empty results
        if len(results) == 0:
            validation_result["potential_issues"].append("Query returned no results")
        
        # Check for excessive results
        if len(results) > 1000:
            validation_result["potential_issues"].append("Query returned an unusually large number of rows")
        
        # Check if all columns are the same value (might indicate a mistake in JOIN conditions)
        if results and all(len(set(row)) == 1 for row in results):
            validation_result["potential_issues"].append("All columns have the same value, possible JOIN issue")
        
        return validation_result
    
    except sqlite3.Error as e:
        return {
            "execution_success": False,
            "error_message": str(e)
        }
    finally:
        if conn:
            conn.close()

def generate_feedback_from_validation(validation_result: Dict[str, Any]) -> str:
    """
    Generates feedback based on query execution validation.
    """
    if not validation_result["execution_success"]:
        return f"The query failed to execute. Error: {validation_result['error_message']}"
    
    feedback = []
    for issue in validation_result["potential_issues"]:
        feedback.append(f"- {issue}")
    
    if validation_result["row_count"] == 0:
        feedback.append("- The query returned no results. Consider relaxing conditions or checking table/column names.")
    elif validation_result["row_count"] > 1000:
        feedback.append("- The query returned a large number of rows. Consider adding more specific conditions.")
    
    return "\n".join(feedback)

def regenerate_sql_with_feedback(question: str, db_path: str, feedback: str, attempts_history: List[Dict]) -> str:
    """
    Regenerates the SQL query with feedback provided using GPT-4.
    """
    schema = generate_schema_prompt(db_path)
    prompt_content = f"""Given the following question, database schema, and feedback, generate an improved SQL query:

Question: {question}

Schema:
{schema}

Previous attempt feedback:
{feedback}

Attempts history:
{attempts_history}

Improved SQL query:"""

    response = openai.ChatCompletion.create(
        model="gpt-4",  # Change to GPT-4 model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates SQL queries."},
            {"role": "user", "content": prompt_content}
        ],
        max_tokens=200,
        temperature=0.3
    )
    
    return response['choices'][0]['message']['content'].strip()


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
        cur_db_path = db_root_path + data['db_id'] + '/' + data['db_id'] + '.sqlite'
        db_path_list.append(cur_db_path)
        knowledge_list.append(data['evidence'])
    
    return question_list, db_path_list, knowledge_list


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


def map_entities_to_schema(entities: List[str], schema: Dict[str, str]):
    """
    Maps extracted entities to schema tables and columns.
    Returns a list of relevant tables and columns with assigned weights.
    """
    relevant_tables = []
    relevant_columns = []
    attention_weights = {}

    for entity in entities:
        entity_lower = entity.lower()
        for table, create_stmt in schema.items():
            if entity_lower in table.lower():
                relevant_tables.append(table)
                attention_weights[table] = attention_weights.get(table, 1.0) + 1.0
            else:
                for column in create_stmt.split('(')[1].split(')')[0].split(','):
                    column_name = column.strip().split()[0]
                    if entity_lower in column_name.lower():
                        relevant_columns.append((table, column_name))
                        attention_weights[column_name] = attention_weights.get(column_name, 1.0) + 1.0

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


def generate_difficulty_based_prompt(db_path, question, knowledge=None, difficulty='simple', attempts_history=[], schema=None, chain_of_thought='False'):
    """
    Generates a prompt based on the difficulty level of the question, incorporating schema details,
    external knowledge, and specific guidance for handling the question based on its difficulty.
    """
    if schema is None:
        schema = generate_schema_prompt(db_path, relevant_tables, relevant_columns, attention_weights)  # Using updated schema with attention

    comment_prompt = generate_comment_prompt(question, knowledge)
    history_summary = "\n\nPrevious Attempts Summary:\n" if attempts_history else ""

    for idx, attempt in enumerate(attempts_history, start=1):
        history_summary += f"Attempt {idx}: Generated SQL: {attempt['generated_sql']}, Success: {attempt['is_successful']}\n"

    cot_prompt = cot_wizard() if chain_of_thought == 'True' else ""  # Include chain-of-thought prompting only if enabled

    if difficulty == 'simple':
        detailed_steps = detailed_self_discovery_guide()
    else:
        detailed_steps = detailed_self_discovery_guide() if difficulty in ['moderate', 'challenging'] else ""

    # Pre-process question to extract entities and relationships
    entities, relationships = extract_entities_and_relationships(question)

    # Map entities to schema elements
    relevant_tables, relevant_columns, attention_weights = map_entities_to_schema(entities, schema)

    # Create attention mask
    attention_mask = create_attention_mask(schema, relevant_tables, relevant_columns, attention_weights)

    final_instruction = "Write the SQL query to answer the question, prioritizing schema elements based on the attention weights provided."

    prompt = (
        f"{schema}\n\n"
        f"{comment_prompt}\n\n"
        f"{cot_prompt}\n\n"  # Chain-of-thought section (included if enabled)
        f"{detailed_steps}\n"
        f"{history_summary}\n"
        f"{attention_mask}\n\n"
        f"{final_instruction}\n"
    )

    return prompt




def detailed_self_discovery_guide():
    """
    Provides detailed step-by-step guidance for constructing SQL queries.
    """
    steps = [
        "-- Step 1: Analyze the schema and identify all tables and columns relevant to the question.",
        "-- Step 2: Determine how these tables are related. Identify the type of joins needed to connect these tables.",
        "-- Step 3: Outline what information you need to extract from each table. Consider what columns need to be selected or calculated.",
        "-- Step 4: Identify any filters or conditions that need to be applied to the data. This includes WHERE clauses and any necessary data transformations.",
        "-- Step 5: Consider if any data needs to be aggregated. Determine what GROUP BY or HAVING clauses might be necessary.",
        "-- Step 6: Plan how the results should be ordered. Decide on the ORDER BY clauses.",
        "-- Step 7: Construct the SQL query step by step, incorporating the elements identified in the previous steps."
    ]
    return '\n'.join(steps)


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)

def connect_gpt(engine, prompt, max_tokens, temperature, stop):
    #print("Prompt to GPT:")
    #print(prompt)
    #print("-" * 50)

    client = openai.ChatCompletion()
    try:
        result = client.create( model=engine, max_tokens=max_tokens, messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": prompt}],temperature=temperature, stop=stop)
    except Exception as e:
        result = 'error:{}'.format(e)
    return result

def collect_response_from_gpt_with_retry(db_path_list, question_list, api_key, engine, knowledge_list=None):
    openai.api_key = api_key
    response_list = []
    feedback_results = {}

    for i, question in enumerate(question_list):
        print(f"Processing {i + 1}/{len(question_list)}: {question}")

        # Extract schema as a dictionary
        schema_dict = get_db_schemas(args.db_root_path, db_path_list[i])

        # Initial prompt and query generation
        attempts_history = []
        attempt = 0
        is_successful = False

        while attempt < 3 and not is_successful:
            print(f"Attempt {attempt + 1} for question: {question}")
            
            # Generate the initial prompt
            cur_prompt = generate_difficulty_based_prompt(
                db_path=db_path_list[i],
                question=question,
                knowledge=knowledge_list[i] if knowledge_list else None,
                schema=schema_dict
            )

            # Get the response from GPT-4 API
            result = connect_gpt(engine=engine, prompt=cur_prompt, max_tokens=256, temperature=0.5, stop=['--', '\n\n', ';', '#'])
            
            if isinstance(result, str):  # If the result is already a string (error handling)
                sql = result
            else:
                sql = 'SELECT ' + result['choices'][0]['message']['content'].strip()

            # Validate the SQL query by executing it
            validation_result = execute_and_validate_query(db_path_list[i], sql, question)

            if validation_result["execution_success"] and not validation_result["potential_issues"]:
                print("Query successful!")
                is_successful = True
            else:
                # Generate feedback and retry
                feedback = generate_feedback_from_validation(validation_result)
                sql = regenerate_sql_with_feedback(question, db_path_list[i], feedback, attempts_history)

            # Save the attempt details in the history
            attempts_history.append({
                "generated_sql": sql,
                "is_successful": validation_result["execution_success"],
                "feedback": feedback
            })

            attempt += 1

        # Save the final successful (or last attempted) SQL query
        response_list.append(sql)

        # Store the feedback results for this question
        feedback_results[i] = {
            "question": question,
            "generated_sql": sql,
            "is_successful": is_successful,
            "attempts": attempt,
            "feedback_history": attempts_history  # Including the feedback for each attempt
        }

    return response_list, feedback_results


def save_feedback(feedback_results, feedback_output_path):
    """
    Saves the feedback results to a specified file.
    """
    if feedback_output_path:
        directory_path = os.path.dirname(feedback_output_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(feedback_output_path, 'w') as json_file:
            json.dump(feedback_results, json_file, indent=4)

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
    args_parser.add_argument('--feedback_output_path', type=str, help="Path to store feedback output", default=None)
    args_parser.add_argument('--chain_of_thought', type=str, default='False', help="Enable chain-of-thought prompting (True/False)")  # New argument

    args = args_parser.parse_args()

    # Load evaluation data
    eval_data = json.load(open(args.eval_path, 'r'))

    # Decouple question and schema
    question_list, db_path_list, knowledge_list = decouple_question_schema(datasets=eval_data, db_root_path=args.db_root_path)

    # Collect responses from GPT
    if args.use_knowledge == 'True':
        responses, feedback_results = collect_response_from_gpt_with_retry(
            db_path_list=db_path_list, 
            question_list=question_list, 
            api_key=args.api_key, 
            engine=args.engine, 
            knowledge_list=knowledge_list,
            chain_of_thought=args.chain_of_thought  # Pass chain-of-thought argument
        )
    else:
        responses, feedback_results = collect_response_from_gpt_with_retry(
            db_path_list=db_path_list, 
            question_list=question_list, 
            api_key=args.api_key, 
            engine=args.engine, 
            chain_of_thought=args.chain_of_thought  # Pass chain-of-thought argument
        )

    # Save SQL queries
    output_name = args.data_output_path + 'predict_' + args.mode + '.json'
    generate_sql_file(sql_lst=responses, output_path=output_name)

    # Save feedback results if feedback_output_path is provided
    save_feedback(feedback_results, args.feedback_output_path)

    print(f'Successfully collected results from {args.engine} for {args.mode} evaluation.')

