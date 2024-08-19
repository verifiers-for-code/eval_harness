import os
import subprocess
import argparse
from tabulate import tabulate
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def run_evaluation(output_dir, subdir, dataset):
    command = f"evalplus.evaluate --dataset {dataset} --samples {os.path.join(output_dir, subdir)}"
    if dataset == "mbpp":
        command += "/samples.jsonl"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

def parse_scores(output, dataset):
    lines = output.split('\n')
    base_score = None
    plus_score = None
    for line in lines:
        if f"{dataset} (base tests)" in line:
            base_score = float(lines[lines.index(line) + 1].split(':')[1].strip()) * 100
        elif f"{dataset}+ (base + extra tests)" in line:
            plus_score = float(lines[lines.index(line) + 1].split(':')[1].strip()) * 100
    return base_score, plus_score

def color_row(subdir, base_score, plus_score):
    if subdir == "gold_plan":
        color = Fore.GREEN
    elif subdir == "none":
        color = Fore.BLUE
    else:
        color = ""
    
    return [
        f"{color}{subdir}{Style.RESET_ALL}",
        f"{color}{base_score:.3f}{Style.RESET_ALL}",
        f"{color}{plus_score:.3f}{Style.RESET_ALL}"
    ]

def main(model, sort_eval_plus, dataset):
    output_dir = f"{model}-output" if dataset == "humaneval" else f"{model}-output-mbpp"
    results = []
    for subdir in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir)
        if os.path.isdir(subdir_path):
            eval_output = run_evaluation(output_dir, subdir, dataset)
            base_score, plus_score = parse_scores(eval_output, dataset)
            results.append((subdir, base_score, plus_score))
    
    # Sort results based on the selected score
    if sort_eval_plus:
        results.sort(key=lambda x: x[2], reverse=True)  # Sort by plus score
    else:
        results.sort(key=lambda x: x[1], reverse=True)  # Sort by base score
    
    # Apply coloring after sorting
    colored_results = [color_row(*result) for result in results]
    
    # Change the order of columns based on sorting method
    if sort_eval_plus:
        headers = ["Subdir", f"{dataset.upper()}+ (%)", f"{dataset.upper()} (%)"]
        colored_results = [[row[0], row[2], row[1]] for row in colored_results]
    else:
        headers = ["Subdir", f"{dataset.upper()} (%)", f"{dataset.upper()}+ (%)"]
    
    print(tabulate(colored_results, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare HumanEval or MBPP results.")
    parser.add_argument("--model", required=True, help="Model name (e.g., Meta-llama-3-8b-instruct)")
    parser.add_argument("--sort-eval-plus", action="store_true", help="Sort by plus score when specified")
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval", help="Choose dataset to evaluate (default: humaneval)")
    args = parser.parse_args()

    main(args.model, args.sort_eval_plus, args.dataset)