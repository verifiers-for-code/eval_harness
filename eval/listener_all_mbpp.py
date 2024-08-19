import os
import time
import subprocess
import re
import argparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from queue import Queue
from threading import Thread

class CheckpointHandler(FileSystemEventHandler):
    def __init__(self, run_script_path, watch_dir, gpu_id, adapter_dir):
        self.run_script_path = run_script_path
        self.watch_dir = watch_dir
        self.gpu_id = gpu_id
        self.adapter_dir = adapter_dir
        self.processed_checkpoints = set()
        self.checkpoint_queue = Queue()
        self.processing_thread = Thread(target=self.process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def on_created(self, event):
        if event.is_directory and "checkpoint-" in event.src_path:
            self.checkpoint_queue.put(event.src_path)

    def on_moved(self, event):
        if event.is_directory and "checkpoint-" in event.dest_path:
            self.checkpoint_queue.put(event.dest_path)

    def process_queue(self):
        while True:
            checkpoint_path = self.checkpoint_queue.get()
            self.process_checkpoint(checkpoint_path)
            self.checkpoint_queue.task_done()

    def process_checkpoint(self, checkpoint_path):
        checkpoint_name = os.path.basename(checkpoint_path)
        parent_dir = os.path.basename(os.path.dirname(checkpoint_path))
        match = re.search(r'checkpoint-(\d+)(.*)', checkpoint_name)
        if match:
            checkpoint_num = int(match.group(1))
            checkpoint_suffix = match.group(2)
            
            full_checkpoint_name = f"{parent_dir}/{checkpoint_name}"
            if full_checkpoint_name not in self.processed_checkpoints:
                print(f"Processing new checkpoint: {full_checkpoint_name}")
                self.run_script_for_checkpoint(parent_dir, checkpoint_name, checkpoint_num, checkpoint_suffix)
                self.processed_checkpoints.add(full_checkpoint_name)
            else:
                print(f"Skipping {full_checkpoint_name} as it has already been processed.")
        else:
            print(f"Invalid checkpoint directory name: {checkpoint_name}")

    def run_script_for_checkpoint(self, parent_dir, checkpoint_name, checkpoint_num, checkpoint_suffix):
        model_name = f"{parent_dir}/{checkpoint_name}"
        rank = self.get_rank_from_parent_dir(parent_dir)
        subdir_name = f"{parent_dir}_phi3_mini_a64_{checkpoint_name}_planner_mbpp"
        code_gen = "meta-llama/Meta-Llama-3-8B-Instruct"

        print(f"Running script for checkpoint: {model_name}")
        subprocess.run([self.run_script_path, model_name, subdir_name, code_gen, self.gpu_id, self.adapter_dir])

    def get_rank_from_parent_dir(self, parent_dir):
        match = re.search(r'r(\d+)', parent_dir)
        if match:
            n = match.group(1)
            return f"{n}"
        else:
            print(f"Warning: Could not determine rank from parent directory: {parent_dir}")
            return "unknown_rank"

def monitor_directory(path_to_watch, run_script_path, gpu_id, adapter_dir):
    event_handler = CheckpointHandler(run_script_path, path_to_watch, gpu_id, adapter_dir)
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor directory for new checkpoints and process them for MBPP.")
    parser.add_argument("gpu_id", type=str, help="GPU ID to use for processing")
    parser.add_argument("watch_dir", type=str, help="Directory to watch for new checkpoints")
    parser.add_argument("adapter_dir", type=str, help="Directory for adapter_name_or_path")
    args = parser.parse_args()

    WATCH_DIR = f"/shared/model_outputs/{args.watch_dir}"
    RUN_SCRIPT = "./run_all_mbpp.sh"
    
    print(f"Monitoring directory: {WATCH_DIR}")
    print(f"Using GPU ID: {args.gpu_id}")
    print(f"Adapter directory: {args.adapter_dir}")
    monitor_directory(WATCH_DIR, RUN_SCRIPT, args.gpu_id, args.adapter_dir)