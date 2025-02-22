from flask import Flask, request, render_template, send_file, redirect, url_for, current_app, jsonify
import requests
import io
import json
import csv
import os
import logging
from werkzeug.utils import secure_filename
import re
import pandas as pd
from faker import Faker
from openai import OpenAI
from openai import AzureOpenAI
import faker_commerce
import sys
import io
import boto3

app = Flask(__name__)
faker = Faker()  # mentioned locale i.e., language en_US, it_IT

api_key = "your_api_key"
azure_endpoint = 'your_azure_endpoint'
api_version = 'your_api_version'

rows_per_cycle = 500  # Define rows per cycle

# Configure AWS credentials and region
aws_access_key_id = 'your_access_key_id'
aws_secret_access_key = 'your_secret_access_key'
aws_region = 'your_region'  

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/process_schema', methods=['POST'])
def process_schema():
    try:
        data = request.get_json()
        schema = data['schema']
        num_rows = int(data['num_rows'])
        instructions = data['instructions']

        # Calculate the number of cycles required
        num_cycles = (num_rows + rows_per_cycle - 1) // rows_per_cycle

        if isinstance(schema, str):
            schema_json = json.loads(schema)
        else:
            schema_json = schema

        # Extract table names and relationships from the schema
        table_names = [table['name'] for table in schema_json['tables']]
        relationships = schema_json.get('relationships', [])

        # Determine the order of table generation based on relationships
        table_order = determine_table_order(table_names, relationships)

        # Generate data for each table in the determined order
        generated_data = {}
        for table_name in table_order:
            table_schema = next(table for table in schema_json['tables'] if table['name'] == table_name)

            # Extract fields with primary key, rule constraints, or instructions, foreign keys
            constrained_or_instructed_fields = [
                field for field in table_schema['fields']
                if ('primary_key' in field.get('constraints', []))  # Primary key fields
                   or ('foreign_key' in field.get('constraints', []))  # Foreign key fields
                   or ('rules' in field and field['rules'])  # Fields with rules
            ]

            # Extract remaining fields (excluding foreign keys)
            remaining_fields = [
                field for field in table_schema['fields']
                if field not in constrained_or_instructed_fields
                and 'foreign_key' not in field.get('constraints', [])
            ]

            # Generate data for constrained or instructed fields using the LLM
            constrained_or_instructed_data = {}
            for field in constrained_or_instructed_fields:
                field_data = ""
                last_rows = None
                for cycle in range(num_cycles):
                    start_row = cycle * rows_per_cycle + 1
                    end_row = min((cycle + 1) * rows_per_cycle, num_rows)
                    cycle_num_rows = end_row - start_row + 1
                    field_cycle_data, last_rows = generate_constrained_or_instructed_data(table_schema, field, relationships, generated_data, instructions, cycle_num_rows, last_rows, start_row, end_row, cycle + 1)
                    field_data += field_cycle_data + '\n'
                constrained_or_instructed_data[field['field_name']] = field_data.strip()

            # Generate data for the remaining fields using Python libraries
            remaining_data = generate_remaining_data(table_schema, remaining_fields, num_rows)

            # Combine the constrained or instructed data and remaining data
            csv_data = combine_data(constrained_or_instructed_data, {}, remaining_data, num_rows, table_schema, relationships, generated_data)
            generated_data[table_name] = csv_data

        s3_bucket_name = 'centelon-generative-ai'
        s3_subfolder = 'test_data_generation'
        files_info, message = process_csv_response(generated_data, s3_bucket_name, s3_subfolder)

        return jsonify(message="Data generated successfully for all tables.", files_info=files_info, timer=50000)
    except IndexError as e:
        logging.error(f"Error in process_schema: {e}")
        # Log the traceback for more detailed information
        logging.exception("Traceback:")
        return jsonify(message=str(e), timer=5000), 500
    except Exception as e:
        logging.error(f"Error in process_schema: {e}")
        return jsonify(message=str(e), timer=5000), 500

def determine_table_order(table_names, relationships):
    table_order = []
    visited = set()

    def dfs(table_name):
        if table_name in visited:
            return
        visited.add(table_name)

        # Visit the referenced tables first
        for rel in relationships:
            if rel['from'].split('.')[0] == table_name or rel['to'].split('.')[0] == table_name:
                referenced_table = rel['from'].split('.')[0] if rel['to'].split('.')[0] == table_name else rel['to'].split('.')[0]
                dfs(referenced_table)

        table_order.append(table_name)

    for table_name in table_names:
        dfs(table_name)

    return table_order

def process_csv_response(csv_response, s3_bucket_name, s3_subfolder):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)
    files_info = {}

    for table_name, csv_data in csv_response.items():
        s3_key = f"{s3_subfolder}/{table_name}.csv"
        s3.put_object(Body=csv_data, Bucket=s3_bucket_name, Key=s3_key)
        files_info[table_name] = f"s3://{s3_bucket_name}/{s3_key}"

    return files_info, "CSV files uploaded to S3 successfully."

def generate_constrained_or_instructed_data(table_schema, field, relationships, generated_data, instructions, cycle_num_rows, last_rows=None, start_row=1, end_row=None, cycle_number=1):
    field_data = ""

    # Check if the field is a primary key or foreign key
    if 'primary_key' in field.get('constraints', []) or 'foreign_key' in field.get('constraints', []):
        # Create prompt for the field, relationships, generated data, and example output
        prompt = create_prompt_for_field(table_schema, field, relationships, generated_data, instructions, cycle_num_rows, last_rows, start_row, end_row, cycle_number)
        print(f"Sending the following prompt to the model for field '{field['field_name']}' in table '{table_schema['name']}':")
        print(prompt)

        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version)

        response = client.chat.completions.create(
            model="omni_model",
            temperature=0,
            top_p=1,
            messages=[{"role": "system", "content": prompt}],
            seed=8627401
        )
        print(f"Raw response from the model for field '{field['field_name']}' in table '{table_schema['name']}':")
        print(response)

        if response:
            model_response = response.choices[0].message.content
            print(f"Model's response to process for field '{field['field_name']}' in table '{table_schema['name']}':")
            print(model_response)

            # Clean up the model's response by removing code block formatting and unwanted lines
            model_response = model_response.strip().replace('```csv', '').replace('```', '').strip()
            model_response_lines = model_response.strip().split('\n')
            filtered_lines = [line for line in model_response_lines if not line.startswith(('Sure,', 'Here is', 'The generated', 'Please find'))]
            model_response = '\n'.join(filtered_lines)

            # Truncate the generated data to the desired number of rows
            truncated_response = '\n'.join(filtered_lines[:cycle_num_rows])

            if truncated_response:
                # Extract the last 2 rows from the truncated response
                last_rows = extract_last_rows(truncated_response, 2)
            else:
                last_rows = []

            field_data += truncated_response + '\n'
        else:
            print(f"API Error: Unable to get response from the model for field '{field['field_name']}' in table '{table_schema['name']}'.")
            field_data += f"API Error: Unable to get response from the model for field '{field['field_name']}' in table '{table_schema['name']}'." + '\n'

    # Check if the field has rules
    elif 'rules' in field and field['rules']:
        # Create prompt for the field, relationships, generated data, and example output
        prompt = create_prompt_for_field(table_schema, field, relationships, generated_data, instructions, cycle_num_rows, last_rows, start_row, end_row, cycle_number)
        print(f"Sending the following prompt to the model for field '{field['field_name']}' in table '{table_schema['name']}':")
        print(prompt)

        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version)

        response = client.chat.completions.create(
            model="omni_model",
            temperature=0,
            top_p=1,
            messages=[{"role": "system", "content": prompt}],
            seed=8627401
        )
        print(f"Raw response from the model for field '{field['field_name']}' in table '{table_schema['name']}':")
        print(response)

        if response:
            model_response = response.choices[0].message.content
            print(f"Model's response to process for field '{field['field_name']}' in table '{table_schema['name']}':")
            print(model_response)

            # Clean up the model's response by removing code block formatting and unwanted lines
            model_response = model_response.strip().replace('```csv', '').replace('```', '').strip()
            model_response_lines = model_response.strip().split('\n')
            filtered_lines = [line for line in model_response_lines if not line.startswith(('Sure,', 'Here is', 'The generated', 'Please find'))]
            model_response = '\n'.join(filtered_lines)

            # Extract the generated rows from the model's response
            generated_rows = model_response.strip().split('\n')

            # Ensure the correct number of rows is generated
            while len(generated_rows) != cycle_num_rows:
                if len(generated_rows) < cycle_num_rows:
                    # Generate additional rows to meet the required count
                    additional_rows = cycle_num_rows - len(generated_rows)
                    prompt = create_prompt_for_field(table_schema, field, relationships, generated_data, instructions, additional_rows, last_rows, start_row, end_row, cycle_number)
                    additional_response = client.chat.completions.create(
                        model="omni_model",
                        temperature=0,
                        top_p=1,
                        messages=[{"role": "system", "content": prompt}],
                        seed=8627401
                    )
                    if additional_response:
                        additional_model_response = additional_response.choices[0].message.content
                        additional_model_response = additional_model_response.strip().replace('```csv', '').replace('```', '').strip()
                        additional_rows_lines = additional_model_response.strip().split('\n')
                        filtered_additional_rows = [line for line in additional_rows_lines if not line.startswith(('Sure,', 'Here is', 'The generated', 'Please find'))]
                        generated_rows.extend(filtered_additional_rows)
                else:
                    # Truncate the generated rows to match the requested count
                    generated_rows = generated_rows[:cycle_num_rows]

            truncated_response = '\n'.join(generated_rows)

            if truncated_response:
                # Extract the last 2 rows from the truncated response
                last_rows = extract_last_rows(truncated_response, 2)
            else:
                last_rows = []

            field_data += truncated_response + '\n'
        else:
            print(f"API Error: Unable to get response from the model for field '{field['field_name']}' in table '{table_schema['name']}'.")
            field_data += f"API Error: Unable to get response from the model for field '{field['field_name']}' in table '{table_schema['name']}'." + '\n'

    return field_data.strip(), last_rows

def create_prompt_for_field(table_schema, field, relationships, generated_data, instructions, cycle_num_rows, last_rows=None, start_row=1, end_row=None, cycle_number=1):
    schema_str = f"Table: {table_schema['name']}\nField: {field['field_name']} ({field['data_type']})\n"
    if 'constraints' in field:
        schema_str += f"Constraints: {', '.join(field['constraints'])}\n"
    if 'rule' in field:
        schema_str += f"Rule: {field['rule']}\n"
 
    relationship_str = ""
    table_relationships = [r for r in relationships if r['from'].split('.')[0] == table_schema['name'] or r['to'].split('.')[0] == table_schema['name']]
    if table_relationships:
        relationship_str = "Relationships:\n"
        for rel in table_relationships:
            if rel['from'].split('.')[0] == table_schema['name']:
                relationship_str += f"- {rel['from']} references {rel['to']}\n"
            else:
                relationship_str += f"- {rel['to']} is referenced by {rel['from']}\n"
 
    instructions_str = ""
    if 'rules' in field and field['rules']:
        instructions_str = f"{table_schema['name']} table {field['field_name']} column {field['rules']}\n"
 
    last_rows_str = ""
    if last_rows:
        last_rows_str = "Last Rows:\n" + "\n".join(last_rows)
    else:
        last_rows_str = "Last Rows: None"
 
    rules_instructions = ""
    if 'rules' in field and field['rules']:
        rules_instructions = f"For the rules field '{field['field_name']}', it is crucial to generate exactly {cycle_num_rows} rows of data in this cycle. The generated data must strictly adhere to the specified rules and instructions. Please ensure that the number of rows generated matches the requested count precisely."
 
    prompt = f"""
Given the following information:
Schema:
{schema_str}
{relationship_str}
 
Instructions:
{instructions_str}
{last_rows_str}
{rules_instructions}
 
Cycle Number: {cycle_number}
Please generate {cycle_num_rows} rows of test data in CSV format for the '{field['field_name']}' field of the '{table_schema['name']}' table, ensuring the data complies with the field's data type, constraints, rules, relationships, and instructions.
 
Important notes:
- For primary key fields, generate unique and sequential values, continuing from the last generated value if applicable.
- For foreign key fields, generate unique and sequential values, continuing from the last generated value if applicable.
- For rules fields, generate unique and sequential values, continuing from the last generated value if applicable, and ensure that the number of rows generated matches the requested count precisely.
- Ensure that all generated values are not null and every row is populated with values.
- The number of rows generated should exactly match the requested number of rows.
- The rows should start from {start_row} and continue sequentially till {end_row}.
 
 
Please provide only the generated test data for the '{field['field_name']}' field in CSV format without any headers or additional text.
"""
    return prompt
 
from faker import Faker

faker = Faker()

def generate_remaining_data(table_schema, remaining_fields, num_rows):
    data_type_map = {
        'string': lambda: faker.word(),
        'integer': lambda: faker.random_int(),
        'bigint': lambda: faker.random_int(),
        'boolean': lambda: faker.boolean(),
        'date': lambda: faker.date(),
        'email': lambda: faker.email(),
        'url': lambda: faker.url(),
        'address': lambda: faker.address().replace('\n', ', '),
        'name': lambda: faker.name(),
        'phone number': lambda: faker.phone_number(),
        'uuid': lambda: faker.uuid4(),
        'first name': lambda: faker.first_name(),
        'last name': lambda: faker.last_name(),
        'middle name': lambda: faker.first_name(),
        'decimal': lambda: faker.pyfloat(),
        'alphanumeric': lambda: faker.pystr(),
        'job': lambda: faker.job().replace(',', ''),
        'year': lambda: faker.year(),
        'full name': lambda: faker.name(),
        'time': lambda: faker.time(),
        'street address': lambda: faker.street_address().replace('\n', ', '),
        'city': lambda: faker.city(),
        'zip code': lambda: faker.postcode(),
        'region': lambda: faker.state(),
        'country': lambda: faker.country(),
        'latitude': lambda: faker.latitude(),
        'longitude': lambda: faker.longitude(),
        'smallint': lambda: faker.random_int(min=0, max=32767),
        'array<string>': lambda: [faker.word() for _ in range(faker.random_int(min=1, max=5))],
        'company': lambda: faker.company().replace(',', ''),
        'company suffix': lambda: faker.company_suffix(),
        'first name female': lambda: faker.first_name_female(),
        'first name male': lambda: faker.first_name_male(),
        'prefix': lambda: faker.prefix(),
        'prefix female': lambda: faker.prefix_female(),
        'prefix male': lambda: faker.prefix_male(),
        'building number': lambda: faker.building_number(),
        'country code': lambda: faker.country_code(),
        'street name': lambda: faker.street_name(),
        'license plate': lambda: faker.license_plate(),
        'basic bank account number': lambda: faker.bban(), ##Basic Bank Account Number (BBAN)
        'international bank account number': lambda: faker.iban(), ##International Bank Account Number (BBAN).
        'credit card expire': lambda: faker.credit_card_expire(),
        'credit card full details': lambda: faker.credit_card_full(),
        'credit card number': lambda: faker.credit_card_number(),
        'credit card provider': lambda: faker.credit_card_provider(),
        'credit card security code': lambda: faker.credit_card_security_code(),
        'cryptocurrency': lambda: faker.cryptocurrency(),
        'cryptocurrency code': lambda: faker.cryptocurrency_code(),
        'currency': lambda: faker.currency(),
        'currency code': lambda: faker.currency_code(),
        'currency name': lambda: faker.currency_name(),
        'currency symbol': lambda: faker.currency_symbol(),
        'pricetag': lambda: faker.pricetag(),
        'am pm': lambda: faker.am_pm(),
        'century': lambda: faker.century(),
        'date of birth': lambda: faker.date_of_birth(),
        'month': lambda: faker.month(),
        'month name': lambda: faker.month_name(),
        'date of month': lambda: faker.day_of_month(),
        'date of week': lambda: faker.day_of_week(),
        'time zone': lambda: faker.timezone(),
        'file extension': lambda: faker.file_extension(),
        'file name': lambda: faker.file_name(),
        'file path':lambda: faker.file_path(),
        'coordinate': lambda: faker.coordinate(),
        'latlng': lambda: faker.latlng(),
        'location on land': lambda: faker.location_on_land(),
        'local latlng': lambda: faker.local_latlng(),
        'domain name': lambda: faker.domain_name(),
        'image url': lambda: faker.image_url(),
        'user name': lambda: faker.user_name(),
        'uri': lambda: faker.uri(),
        'paragraph': lambda: faker.paragraph(),
        'sentence': lambda: faker.sentence(),
        'sentences': lambda: faker.sentences(),
        'text': lambda: faker.text(),
        'texts': lambda: faker.texts(),
        'word': lambda: faker.word(),
        'words': lambda: faker.words(),
        'boolean': lambda: faker.boolean(),
        'null boolean': lambda: faker.null_boolean(),
        'password': lambda: faker.password(),
        'uuid4': lambda: faker.uuid4(),
        #'passport_number': lambda: faker.passport_number(),
        #'passport_owner': lambda: faker.passport_owner(),
        'country calling code': lambda: faker.country_calling_code(),
    }

    field_names = [field['field_name'] for field in remaining_fields]
    data = []

    for _ in range(num_rows):
        row = {}
        for field in remaining_fields:
            field_name = field['field_name']
            data_type = field['data_type'].lower()  # Convert data type to lowercase

            if data_type in data_type_map:
                faker_func = data_type_map[data_type]
                value = str(faker_func())
                # Handle Unicode characters
                value = value.encode('utf-8').decode('utf-8')
                # Remove commas and newline characters from the generated value
                value = value.replace(',', '').replace('\n', ' ')
                row[field_name] = value
            else:
                row[field_name] = ''
                print(f"Warning: No generator for data type: {data_type}")  # Debug print

        data.append(row)

    csv_data = ','.join(field_names) + '\n'
    for row in data:
        csv_data += ','.join([f'"{str(row.get(field_name, ""))}"' for field_name in field_names]) + '\n'

    return csv_data
 
def combine_data(constrained_data, instructed_data, remaining_data, num_rows, table_schema, relationships, generated_data):
    all_data = {}

    if isinstance(constrained_data, dict):
        all_data.update(constrained_data)
    else:
        all_data['constrained_data'] = constrained_data

    if isinstance(instructed_data, dict):
        all_data.update(instructed_data)
    else:
        all_data['instructed_data'] = instructed_data

    if isinstance(remaining_data, dict):
        all_data.update(remaining_data)
    else:
        all_data['remaining_data'] = remaining_data

    # Extract column names and values from remaining_data
    remaining_data_lines = remaining_data.strip().split('\n')
    remaining_data_header = remaining_data_lines[0].split(',')
    remaining_data_values = [line.split(',') for line in remaining_data_lines[1:]]

    # Create a list to store the rows
    rows = []

    for i in range(num_rows):
        row = []
        for field in table_schema['fields']:
            field_name = field['field_name']
            if field_name in constrained_data:
                constrained_values = constrained_data[field_name].split('\n')
                if i < len(constrained_values):
                    row.append(constrained_values[i])
                else:
                    # Check if the field is a foreign key
                    if is_foreign_key(field_name, table_schema):
                        # Get the referenced table and primary key column
                        referenced_table, referenced_column = get_referenced_table_and_column(field_name, relationships)
                        if referenced_table and referenced_column:
                            # Copy the generated value from the referenced primary key column
                            referenced_value = get_referenced_values(referenced_table, referenced_column, i, generated_data)
                            row.append(referenced_value)
                        else:
                            row.append('')
                    else:
                        row.append('')
            elif field_name in instructed_data:
                instructed_values = instructed_data[field_name].split('\n')
                if i < len(instructed_values):
                    row.append(instructed_values[i])
                else:
                    row.append('')
            else:
                if field_name in remaining_data_header:
                    field_index = remaining_data_header.index(field_name)
                    if i < len(remaining_data_values) and field_index < len(remaining_data_values[i]):
                        row.append(remaining_data_values[i][field_index])
                    else:
                        row.append('')
                else:
                    row.append('')
        rows.append(row)

    # Convert the list of rows to a CSV string
    csv_data = ','.join([field['field_name'] for field in table_schema['fields']]) + '\n'
    csv_data += '\n'.join([','.join(row) for row in rows])

    return csv_data
 
def extract_last_rows(text, num_rows):
    if text is None:
        return []
    lines = text.strip().split('\n')
    return lines[-num_rows:]
 
def is_foreign_key(field_name, table_schema):
    for field in table_schema['fields']:
        if field['field_name'] == field_name and 'foreign_key' in field.get('constraints', []):
            return True
    return False
 
def get_referenced_table_and_column(field_name, relationships):
    for rel in relationships:
        if rel['from'].split('.')[1] == field_name:
            return rel['to'].split('.')[0], rel['to'].split('.')[1]
    return None, None
 
def get_referenced_values(referenced_table, referenced_column, generated_data, num_rows):
    if referenced_table in generated_data:
        referenced_data = generated_data[referenced_table]
        referenced_rows = referenced_data.split('\n')
        referenced_column_index = get_column_index(referenced_table, referenced_column, generated_data)
        if referenced_column_index is not None:
            referenced_values = [row.split(',')[referenced_column_index] for row in referenced_rows[:num_rows]]
            return referenced_values
    return []
 
def get_column_index(table_name, column_name, generated_data):
    if table_name in generated_data:
        table_data = generated_data[table_name]
        header_row = table_data.split('\n')[0]
        columns = header_row.split(',')
        if column_name in columns:
            return columns.index(column_name)
    return None
 
if __name__ == '__main__':
    app.run()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501)
