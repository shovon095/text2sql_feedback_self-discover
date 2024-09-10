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
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

openai.debug = True
import concurrent.futures
import threading
# Initialize SpaCy NLP model
nlp = spacy.load("en_core_web_sm")


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

def clean_sql_query(generated_sql: str) -> str:
    """
    Cleans the generated SQL query by removing duplicate SELECT statements,
    unnecessary newlines, and ensuring proper capitalization.
    """
    # Remove any markdown-style code block markers
    cleaned_sql = generated_sql.strip().replace("```sql", "").replace("```", "")
    
    # Remove any remaining backticks
    cleaned_sql = cleaned_sql.replace("`", "")
    
    # Remove duplicate SELECT statements
    cleaned_sql = re.sub(r'(?i)(\s*SELECT\s+)+', ' SELECT ', cleaned_sql)
    
    # Normalize whitespace
    cleaned_sql = " ".join(cleaned_sql.split())
    
    # Ensure the query starts with SELECT
    if not cleaned_sql.upper().startswith("SELECT"):
        cleaned_sql = "SELECT " + cleaned_sql
    
    return cleaned_sql.strip()



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

class SchemaCache:
    def __init__(self, cache_dir: str = "./schema_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def get_cache_file_path(self, db_path: str) -> str:
        """Generate a cache file path based on the database file path."""
        cache_file_name = os.path.basename(db_path).replace('.sqlite', '.json')
        return os.path.join(self.cache_dir, cache_file_name)

    def load_schema_from_cache(self, db_path: str) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """Load schema from cache if it exists."""
        cache_file = self.get_cache_file_path(db_path)
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
            return data['schemas'], data['all_columns']
        return None, None

    def save_schema_to_cache(self, db_path: str, schemas: Dict[str, Any], all_columns: Dict[str, List[str]]):
        """Save the schema to the cache."""
        cache_file = self.get_cache_file_path(db_path)
        data = {
            'schemas': schemas,
            'all_columns': all_columns
        }
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    def get_or_fetch_schema(self, db_path: str) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """Return the schema from cache or fetch from database if not cached."""
        # Try loading from cache first
        schemas, all_columns = self.load_schema_from_cache(db_path)
        if schemas and all_columns:
            print(f"Loaded schema from cache for {db_path}")
            return schemas, all_columns

        # If not cached, fetch from database and save to cache
        schemas, all_columns = get_db_schemas(db_path)
        self.save_schema_to_cache(db_path, schemas, all_columns)
        return schemas, all_columns


schema_cache = SchemaCache()

def is_subselect(parsed):
    if not parsed.is_group:
        return False
    for item in parsed.tokens:
        if item.ttype is DML and item.value.upper() == 'SELECT':
            return True
    return False

def extract_identifiers(token_stream):
    """Extract column and table identifiers from a parsed SQL query."""
    identifiers = []
    for item in token_stream:
        if isinstance(item, IdentifierList):
            for identifier in item.get_identifiers():
                identifiers.append(identifier.get_real_name())
        elif isinstance(item, Identifier):
            identifiers.append(item.get_real_name())
        elif item.ttype is Keyword:
            continue
        elif is_subselect(item):
            identifiers.extend(extract_identifiers(item.tokens))
    return identifiers

def analyze_query(sql_query: str, all_columns: Dict[str, List[str]]) -> Dict[str, Any]:
    issues = []
    parsed = sqlparse.parse(sql_query)[0]
    
    # Check if it's a SELECT query
    if parsed.get_type() != 'SELECT':
        issues.append("Query is not a SELECT statement")
    
    # Extract column and table identifiers
    used_columns = extract_identifiers(parsed.tokens)
    
    # Check for existence of columns and tables
    for col in used_columns:
        if '.' in col:
            table, column = col.split('.')
            if table not in all_columns or column not in all_columns[table]:
                issues.append(f"Column {col} not found in schema")
        else:
            # If the column isn't part of a table.column format, check all columns
            found = False
            for table, columns in all_columns.items():
                if col in columns:
                    found = True
                    break
            if not found:
                issues.append(f"Column {col} not found in schema")
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues
    }

def execute_with_adaptive_timeout(cursor, query, initial_timeout=5, max_timeout=30):
    timeout = initial_timeout
    while timeout <= max_timeout:
        try:
            cursor.execute(f"pragma query_timeout={timeout * 1000}")
            return cursor.execute(query).fetchall()
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e).lower():
                timeout *= 2
            else:
                raise
    raise TimeoutError(f"Query execution exceeded maximum timeout of {max_timeout} seconds")



def escape_sql_identifier(identifier: str) -> str:
    """
    Escapes SQL identifiers (e.g., table names, column names) by wrapping them in double quotes.
    This prevents issues when reserved keywords are used as identifiers.
    """
    return f'"{identifier}"'  # Wrap in double quotes to escape

def get_db_schemas(db_path: str) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """
    Reads an SQLite file, returns the CREATE commands for each of the tables in the database,
    and retrieves the list of columns in each table, with proper escaping of table and column names.
    """
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schemas = {}
        all_columns = {}

        for table in tables:
            table_name = table[0]

            # Escape the table name using the helper function
            table_name_escaped = escape_sql_identifier(table_name)

            # Retrieve the CREATE statement for the table
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name={table_name_escaped}")
            schemas[table_name] = cursor.fetchone()[0]

            # Get columns for the table using PRAGMA and escape table name
            cursor.execute(f"PRAGMA table_info({table_name_escaped})")
            columns = cursor.fetchall()
            all_columns[table_name] = [escape_sql_identifier(col[1]) for col in columns]  # Escape column names

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

def generate_sql_file(sql_lst, db_path_list, output_path=None):
    result = {}
    for i, sql in enumerate(sql_lst):
        db_id = db_path_list[i].split('/')[-1].split('.sqlite')[0]
        sql_with_db_id = sql + f'\t----- bird -----\t{db_id}'
        result[i] = sql_with_db_id  # Save the SQL query with db_id
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


def escape_column_name(column_name: str) -> str:
    if not column_name.startswith('"') and not column_name.endswith('"'):
        return f'"{column_name}"'
    return column_name


def escape_all_column_names(sql_query: str, all_columns: Dict[str, List[str]]) -> str:
    tokens = re.split(r'(\W+)', sql_query)
    all_columns_flat = {col for columns in all_columns.values() for col in columns}
    escaped_tokens = [escape_column_name(token) if token in all_columns_flat else token for token in tokens]
    escaped_sql_query = ''.join(escaped_tokens)
    return escaped_sql_query


def is_valid_sql(sql_query):
    try:
        parsed = sqlparse.parse(sql_query)
        if len(parsed) > 0 and parsed[0].get_type() == 'SELECT':
            return True, None  # Valid SQL, no error message
        else:
            return False, "SQL query does not start with a valid SELECT statement."
    except Exception as e:
        return False, str(e)  # Return the exception message as the error


def execute_and_validate_query(db_path: str, sql_query: str, question: str, all_columns: Dict[str, List[str]]) -> Dict[str, Any]:
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        cursor = conn.cursor()
        
        # Fetch the schema (this should already be cached)
        _, all_columns = schema_cache.get_or_fetch_schema(db_path)
        
        # Analyze query before execution
        analysis_result = analyze_query(sql_query, all_columns)
        if not analysis_result["is_valid"]:
            return {
                "execution_success": False,
                "error_message": "Invalid query structure",
                "potential_issues": analysis_result["issues"]
            }
        
        # Escape all column names in the SQL query
        escaped_sql_query = escape_all_column_names(sql_query, all_columns)
        
        print("Executing SQL query with adaptive timeout...")
        # Execute the SQL query with adaptive timeout
        results = execute_with_adaptive_timeout(cursor, escaped_sql_query)
        print("SQL query executed successfully")
        
        if results is None:
            return {
                "execution_success": False,
                "error_message": "SQL query returned no results."
            }
        
        # Get column names from the query result
        column_names = [description[0] for description in cursor.description]
        
        return {
            "execution_success": True,
            "row_count": len(results),
            "column_count": len(column_names),
            "sample_data": results[:5] if results else [],
            "column_names": column_names,
            "potential_issues": []
        }

    except sqlite3.Error as e:
        print(f"SQLite error occurred: {e}")
        return {
            "execution_success": False,
            "error_message": f"SQLite Error: {str(e)}",
            "potential_issues": [f"SQLite Error: {str(e)}"]
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


def regenerate_sql_with_feedback(question: str, db_path: str, feedback: str, attempts_history: List[Dict], all_columns: Dict[str, List[str]]) -> str:
    schema, _ = schema_cache.get_or_fetch_schema(db_path)

    prompt_content = f"""Given the following question, database schema, and feedback, generate an improved SQL query:

    Question: {question}

    Schema:
    {schema}

    Previous attempt feedback:
    {feedback}

    Attempts history:
    {attempts_history}

    Improved SQL query (no explanations or formatting):
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an SQL expert. Respond only with valid SQL queries, no explanations."},
            {"role": "user", "content": prompt_content}
        ],
        max_tokens=200,
        temperature=0.3
    )

    generated_sql = response['choices'][0]['message']['content'].strip()

    # Clean the SQL query to remove markdown formatting
    cleaned_sql = clean_sql_query(generated_sql)

    # Escape all column names in the cleaned SQL
    escaped_sql = escape_all_column_names(cleaned_sql, all_columns)

    return escaped_sql



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


def generate_difficulty_based_prompt(db_path, question, knowledge=None, difficulty='simple', attempts_history=[], schema=None, chain_of_thought='False'):
    """
    Generates a prompt based on the difficulty level of the question, incorporating schema details,
    external knowledge, and specific guidance for handling the question based on its difficulty.
    """
    # Use the cache to fetch the schema
    schema_dict, all_columns = schema_cache.get_or_fetch_schema(db_path)

    # Pre-process question to extract entities and relationships
    entities, relationships = extract_entities_and_relationships(question)

    # Map entities to schema elements
    relevant_tables, relevant_columns, attention_weights = map_entities_to_schema(entities, relationships, schema_dict)

    # Generate schema prompt
    schema = generate_schema_prompt(db_path, relevant_tables, relevant_columns, attention_weights)

    # Generate attention mask
    attention_mask = create_attention_mask(schema_dict, relevant_tables, relevant_columns, attention_weights)

    comment_prompt = generate_comment_prompt(question, knowledge)
    history_summary = "\n\nPrevious Attempts Summary:\n" if attempts_history else ""

    for idx, attempt in enumerate(attempts_history, start=1):
        history_summary += f"Attempt {idx}: Generated SQL: {attempt['generated_sql']}, Success: {attempt['is_successful']}\n"

    cot_prompt = cot_wizard() if chain_of_thought == 'True' else ""  # Include chain-of-thought prompting only if enabled

    if difficulty == 'simple':
        detailed_steps = detailed_self_discovery_guide()
    else:
        detailed_steps = detailed_self_discovery_guide() if difficulty in ['moderate', 'challenging'] else ""

    # Strict instruction to only return SQL query
    final_instruction = """
    Generate only a valid SQL query that answers the question. Do not include any explanations, comments, or formatting. 
    The output should be executable SQL code only.
    """

    # Construct the final prompt for GPT
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




@backoff.on_exception(backoff.expo, openai.error.RateLimitError)

def connect_gpt(engine, prompt, max_tokens, temperature, stop):
    client = openai.ChatCompletion()
    try:
        result = client.create(
            model=engine,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": "You are an SQL expert. Respond only with valid SQL queries, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            stop=stop
        )
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f'error:{e}'


def calculate_confidence(validation_result: Dict[str, Any], analysis_result: Dict[str, Any]) -> float:
    confidence = 0.0
    
    # Execution success increases confidence
    if validation_result["execution_success"]:
        confidence += 0.6  # Execution success provides the highest boost
    
    # Row count feedback
    row_count = validation_result["row_count"]
    if 1 <= row_count <= 1000:
        confidence += 0.2
    elif row_count > 1000:
        confidence -= 0.1
    
    # No issues in query analysis
    if analysis_result["is_valid"]:
        confidence += 0.2
    
    # Reduce confidence for each potential issue
    confidence -= 0.05 * len(validation_result["potential_issues"])  # Reduce confidence gradually
    
    return max(0.0, min(confidence, 1.0))



def collect_response_from_gpt_with_retry(db_path_list, question_list, api_key, engine, knowledge_list=None):
    openai.api_key = api_key
    response_list = []
    feedback_results = {}

    # Initialize schema cache
    schema_cache = SchemaCache()

    for i, question in enumerate(question_list):
        print(f"Processing {i + 1}/{len(question_list)}: {question}")

        # Use the cache to fetch schema
        schema_dict, all_columns = schema_cache.get_or_fetch_schema(db_path_list[i])

        # Initial prompt and query generation
        attempts_history = []
        attempt = 0
        is_successful = False
        best_confidence_score = 0.0
        best_sql_query = None

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
            result = connect_gpt(engine=engine, prompt=cur_prompt, max_tokens=256, temperature=0.5, stop=[';'])

            # Ensure the result is a valid SQL query
            sql = result.strip()
            if not sql.upper().startswith('SELECT'):
                sql = f'SELECT {sql}'

            # Clean the SQL query to remove markdown-style code blocks
            cleaned_sql = clean_sql_query(sql)

            # Analyze the query before validation
            analysis_result = analyze_query(cleaned_sql, all_columns)

            # Validate the cleaned SQL query by executing it
            validation_result = execute_and_validate_query(db_path_list[i], cleaned_sql, question, all_columns)

            # Calculate confidence score based on both validation and analysis results
            confidence_score = calculate_confidence(validation_result, analysis_result)

            # Generate feedback if the query was not successful
            if not validation_result["execution_success"]:
                feedback = generate_feedback_from_validation(validation_result)
                
                # Attempt to regenerate the SQL query with feedback
                regenerated_sql = regenerate_sql_with_feedback(question, db_path_list[i], feedback, attempts_history, all_columns)

                # Clean the regenerated SQL query
                cleaned_regenerated_sql = clean_sql_query(regenerated_sql)

                # Validate the cleaned regenerated query
                validation_result = execute_and_validate_query(db_path_list[i], cleaned_regenerated_sql, question, all_columns)

                # Analyze the regenerated SQL query
                analysis_result = analyze_query(cleaned_regenerated_sql, all_columns)

                # Calculate the confidence score again based on the new analysis
                confidence_score = calculate_confidence(validation_result, analysis_result)

            # Save the attempt details in the history
            attempts_history.append({
                "generated_sql": cleaned_sql,
                "is_successful": validation_result["execution_success"],
                "feedback": validation_result.get("potential_issues", []),
                "confidence_score": confidence_score
            })

            # Update the best SQL query based on confidence
            if confidence_score > best_confidence_score:
                best_confidence_score = confidence_score
                best_sql_query = cleaned_sql

            # Check if the query is successful and meets the confidence threshold
            if validation_result["execution_success"] and confidence_score >= 0.5:
                is_successful = True

            attempt += 1

        # Save the final best SQL query with the highest confidence
        response_list.append(best_sql_query if best_sql_query else cleaned_sql)

        # Store the feedback results for this question
        feedback_results[i] = {
            "question": question,
            "best_sql_query": best_sql_query,
            "best_confidence_score": best_confidence_score,
            "attempts": attempt,
            "feedback_history": attempts_history
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
            #chain_of_thought=args.chain_of_thought  # Pass chain-of-thought argument
        )
    else:
        responses, feedback_results = collect_response_from_gpt_with_retry(
            db_path_list=db_path_list,
            question_list=question_list,
            api_key=args.api_key,
            engine=args.engine,
            #chain_of_thought=args.chain_of_thought  # Pass chain-of-thought argument
        )

    # Save SQL queries
    output_name = args.data_output_path + 'predict_' + args.mode + '.json'
    generate_sql_file(sql_lst=responses, db_path_list=db_path_list, output_path=output_name)
    # Save feedback results if feedback_output_path is provided
    save_feedback(feedback_results, args.feedback_output_path)

    print(f'Successfully collected results from {args.engine} for {args.mode} evaluation.')

