import re
import json
import argparse
import os

def extract_plan(text):
    start_match = re.search(r'<plan>', text)
    if start_match:
        start_index = start_match.end()
        end_match = re.search(r'</plan>', text[start_index:])
        
        if end_match:
            end_index = start_index + end_match.start()
            plan = text[start_index:end_index].strip()
        else:
            # If no closing tag, include everything after <plan>
            plan = text[start_index:].strip()
        
        return plan
    else:
        print("No plan found in text")
    return ""

def insert_plan_into_docstring(prompt, plan):
    # Find the end of the docstring
    docstring_start = prompt.find('"""')
    docstring_end = prompt.find('"""', docstring_start + 3)
    
    # If there's no docstring or it's malformed, return the original prompt
    if docstring_end == -1:
        return prompt
    
    # Get the indentation of the closing docstring
    closing_indent = prompt.rfind('\n', 0, docstring_end) + 1
    indent = prompt[closing_indent:docstring_end].replace('"""', '')
    
    # Append the plan text as is, without modifying indentation or header
    plan_text = "\n\n" + plan
    
    return prompt[:docstring_end] + plan_text + prompt[docstring_end:]

def process_item(item, col_to_clean, col_cleaned_name):
    plan = extract_plan(item[col_to_clean])
    new_prompt = insert_plan_into_docstring(item['prompt'], plan)
    item[col_cleaned_name] = new_prompt  # Add the new column to the item
    return item  # Return the entire item with the new column

def main(input_file_path):
    COL_TO_CLEAN = "output"
    COL_CLEANED_NAME = "cleaned-" + COL_TO_CLEAN

    # Load input JSON file
    with open(input_file_path, 'r') as f:
        data = json.load(f)

    # Process each item in the dataset
    cleaned_data = [process_item(item, COL_TO_CLEAN, COL_CLEANED_NAME) for item in data]

    # Create the output file path
    dir_name = os.path.dirname(input_file_path)
    base_name = os.path.basename(input_file_path)
    output_file_path = os.path.join(dir_name, "cleaned-" + base_name)

    # Save cleaned data to new JSON file as a list
    with open(output_file_path, 'w') as f:
        json.dump(cleaned_data, f, indent=2)

    print(f"Cleaned data saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON file and insert plans into docstrings.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file")
    args = parser.parse_args()
    
    main(args.input_file)