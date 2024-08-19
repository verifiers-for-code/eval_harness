import os
import subprocess
import argparse
from vllm import LLM, SamplingParams
import json
from tabulate import tabulate
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# ==== CONFIG ==== #
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
NUM_GPUS = 1
MAX_TOKENS = 2048
FORCE_PLANS = False
# ==== END CONFIG ==== #

__MAGIC_SPLITTER__ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    MODEL = args.model
    MODEL_NAME = MODEL.split('/')[-1]
    OUTPUT_DIR = MODEL_NAME + "-output"
    print(OUTPUT_DIR)

    response = f"""
Below is a self-contained Python script that solves the problem: 
```python 
{__MAGIC_SPLITTER__}
```
"""

    llm = LLM(model=MODEL, 
              tensor_parallel_size=NUM_GPUS, 
              enable_prefix_caching=False, 
              gpu_memory_utilization=0.90, 
              max_model_len=4096, 
              trust_remote_code=True,
              max_num_seqs=16
            )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0, top_p=0.95, max_tokens=MAX_TOKENS,
    )

    process_mode(args.json_file, llm, tokenizer, sampling_params, response, OUTPUT_DIR, args.key)
    
    # Run evaluation and visualization after processing
    run_evaluation_and_visualize(OUTPUT_DIR, args.sort_eval_plus)

def process_mode(json_rules_path, llm, tokenizer, sampling_params, response, output_dir, key):
    # Load the JSON file
    with open(json_rules_path, 'r') as f:
        content = f.read().strip()
        if content.startswith('[') and content.endswith(']'):
            # It's a JSON array
            rules = json.loads(content)
        else:
            # Try parsing as JSONL
            try:
                rules = [json.loads(line) for line in content.split('\n') if line.strip()]
            except json.JSONDecodeError:
                # If JSONL parsing fails, try parsing as a single JSON object
                rules = [json.loads(content)]
    
    # Extract 'cleaned-output' from each JSON object
    cleaned_outputs = [rule[key] for rule in rules]
    
    # Create prompts using the cleaned outputs
    prompts = [create_prompts(cleaned_output, tokenizer, response, __MAGIC_SPLITTER__) 
               for cleaned_output in cleaned_outputs]

    print(f"Sample prompt:")
    print(prompts[0])

    generated_solutions = get_vllm_code(llm, prompts, sampling_params)

    clean_solutions = [extract_clean_code(code) for code in generated_solutions]
    mode_output_dir = os.path.join(output_dir, args.subdir)  # Use args.subdir instead of subdir
    print(mode_output_dir)
    for index, solution in enumerate(clean_solutions):
        name = f"HumanEval_{index}"
        os.makedirs(os.path.join(mode_output_dir, name), exist_ok=True)
        with open(os.path.join(mode_output_dir, name, '0.py'), 'w', encoding='utf-8') as f:
            f.write(solution)

    convert_to_jsonl(clean_solutions, os.path.join(mode_output_dir, 'solutions.jsonl'))

    print(f"Generation completed")
    print(f"Output directory: {output_dir}")


def create_prompts(prompt, tokenizer, response, __MAGIC_SPLITTER__):
    if FORCE_PLANS:
        prompt = f"Please provide a self-contained Python script that solves the following problem in a markdown code block. Follow the given Action Plan, and ALWAYS HAVE THE DOCSTRING with the Action Plan in your answer.\n```\n{prompt.strip()}\n```\n"
    else:
        prompt = f"Please provide a self-contained Python script that solves the following problem in a markdown code block. \n```\n{prompt.strip()}\n```\n"
    x = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
        add_generation_prompt=True).split(__MAGIC_SPLITTER__)[0]
    return x

def get_vllm_code(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    return [x.outputs[0].text for x in outputs]

def extract_clean_code(text):
    index = text.find("```")
    if index != -1:
        text = text[:index]
    lines = text.splitlines()
    for i, line in enumerate(reversed(lines)):
        if "return" in line:
            last_return_index = len(lines) - i - 1
            return '\n'.join(lines[:last_return_index+1])
    return text

def convert_to_jsonl(solutions, output_file):
    with open(output_file, 'w') as f:
        for i, solution in enumerate(solutions):
            json_line = json.dumps({"task_id": f"HumanEval/{i}", "completion": solution})
            f.write(json_line + '\n')

def run_evaluation_and_visualize(output_dir, sort_eval_plus=True):
    results = []
    for subdir in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir)
        if os.path.isdir(subdir_path):
            
            command = f"evalplus.evaluate --dataset humaneval --samples {subdir_path}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            humaneval_score, humaneval_plus_score = parse_scores(result.stdout)
            results.append((subdir, humaneval_score, humaneval_plus_score))
    
    # Sort results based on the selected score
    if sort_eval_plus:
        results.sort(key=lambda x: x[2], reverse=True)  # Sort by HumanEval+ score
    else:
        results.sort(key=lambda x: x[1], reverse=True)  # Sort by HumanEval score
    
    # Apply coloring after sorting
    #colored_results = [color_row(*result) for result in results]
    
    # Change the order of columns based on sorting method
    if sort_eval_plus:
        headers = ["Subdir", "HumanEval+ (%)", "HumanEval (%)"]
        results = [[row[0], row[2], row[1]] for row in results]
    else:
        headers = ["Subdir", "HumanEval (%)", "HumanEval+ (%)"]
    
    # Visualize the results
    print("\nEvaluation Results:")
    print(tabulate(results, headers=headers, tablefmt="grid"))

def parse_scores(output):
    lines = output.split('\n')
    humaneval_score = None
    humaneval_plus_score = None
    for line in lines:
        if "humaneval (base tests)" in line:
            humaneval_score = float(lines[lines.index(line) + 1].split(':')[1].strip()) * 100
        elif "humaneval+ (base + extra tests)" in line:
            humaneval_plus_score = float(lines[lines.index(line) + 1].split(':')[1].strip()) * 100
    return humaneval_score, humaneval_plus_score

def color_row(subdir, humaneval_score, humaneval_plus_score):
    if subdir == "gold_plan":
        color = Fore.GREEN
    elif subdir == "none":
        color = Fore.BLUE
    else:
        color = ""
    
    return [
        f"{color}{subdir}{Style.RESET_ALL}",
        f"{color}{humaneval_score:.3f}{Style.RESET_ALL}",
        f"{color}{humaneval_plus_score:.3f}{Style.RESET_ALL}"
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate code and evaluate HumanEval results.")
    parser.add_argument("subdir", type=str, nargs='?', help="Name of the subdirectory for output")
    parser.add_argument("gpu", type=str, nargs='?', help="GPU to use (e.g., '0' or '0,1')")
    parser.add_argument("model", type=str, nargs='?', help="Model to use")
    
    parser.add_argument("-s", "--subdir", dest="subdir_flag", type=str, help="Name of the subdirectory for output")
    parser.add_argument("-g", "--gpu", dest="gpu_flag", type=str, help="GPU to use (e.g., '0' or '0,1')")
    parser.add_argument("-m", "--model", dest="model_flag", type=str, help="Model to use")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the JSON file containing prompts")
    parser.add_argument("--key", type=str, required=True, help="Key to read in the JSON for prompts")
    parser.add_argument("--sort-eval-plus", action="store_true", help="Sort by HumanEval+ score when specified")
    
    args = parser.parse_args()
    
    # Use flag versions if provided, otherwise use positional arguments
    subdir = args.subdir_flag if args.subdir_flag is not None else args.subdir
    gpu = args.gpu_flag if args.gpu_flag is not None else args.gpu
    model = args.model_flag if args.model_flag is not None else args.model
    
    if subdir is None or gpu is None or model is None:
        parser.error("Please provide values for subdir, gpu, and model either as positional arguments or with flags.")
    
    args.subdir = subdir
    args.gpu = gpu
    args.model = model
    
    main(args)
