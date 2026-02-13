"""
LMDeploy Distributed Inference Evaluation Script
Supports multi-GPU data parallel inference with checkpoint resume
Applicable for mathematical problems, code generation, and other tasks
"""

# ============================================================================
# Environment Configuration
# ============================================================================
import os as _os
import re
import json
import random
import time
from datetime import timedelta
from jinja2 import Template
from omegaconf import OmegaConf, MISSING

_os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# ============================================================================
# Configuration Loading
# ============================================================================
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


# ============================================================================
# Prompt and Answer Extraction Tools
# ============================================================================
def get_prompt(data_i, system_prompts):
    return Template(system_prompts).render(problem=data_i["question"])


def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output."
    return code_output


# ============================================================================
# Data Sharding
# ============================================================================
def get_data_chunk(data, num_nodes, node_idx):
    total = len(data)
    start = (total * node_idx) // num_nodes
    end = (total * (node_idx + 1)) // num_nodes
    return data[start:end]


# ============================================================================
# Time Formatting Utility
# ============================================================================
def format_time(seconds):
    """Format seconds into human-readable time string"""
    if seconds < 0:
        return "N/A"
    return str(timedelta(seconds=int(seconds)))


# ============================================================================
# Checkpoint Resume Functions
# ============================================================================
def load_checkpoint(output_file):
    """Load existing inference results"""
    if _os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ Loaded from checkpoint: {output_file}", flush=True)
            return data
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}", flush=True)
            return None
    return None


def save_checkpoint(data, output_file):
    """Save current inference results"""
    try:
        # Write to temp file first, then rename (atomic operation to prevent corruption)
        temp_file = output_file + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        _os.replace(temp_file, output_file)
    except Exception as e:
        print(f"✗ Failed to save: {e}", flush=True)


def is_question_completed(data_item, k_sample):
    """Check if a question has completed all k_sample responses"""
    return (
        "full_output" in data_item and 
        len(data_item["full_output"]) >= k_sample
    )


# ============================================================================
# Main Program
# ============================================================================
if __name__ == "__main__":
    
    # ------------------------------------------------------------------------
    # Step 1: Load Configuration
    # ------------------------------------------------------------------------
    config = get_config()
    
    print("="*70)
    print("【Configuration Check】")
    print(f"  max_token: {config.rollout.max_token}")
    print(f"  block_size: {config.rollout.block_size}")
    print(f"  denoising_steps: {config.rollout.denoising_steps_per_block}")
    print(f"  eval_dataset: {config.dataset.eval_dataset}")
    print(f"  result_base: {config.paths.result_base}")
    print("="*70)
    
    # ------------------------------------------------------------------------
    # Step 2: Import LMDeploy
    # ------------------------------------------------------------------------
    import torch
    from transformers import AutoTokenizer
    from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
    
    # ------------------------------------------------------------------------
    # Step 3: Calculate GPU Count and Data Parallelism
    # ------------------------------------------------------------------------
    cvd = _os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        visible_gpus = [x.strip() for x in cvd.split(",") if x.strip() != ""]
        gpu_num = len(visible_gpus)
    else:
        gpu_num = torch.cuda.device_count()
    
    print(f"[GPU Config] Detected {gpu_num} available GPUs")
    
    # Use data parallelism
    tp = config.rollout.tensor_parallel_size
    dp = gpu_num
    
    print(f"[Parallel Strategy] tp={tp}, dp={dp} (Data Parallel Mode)")
    
    # ------------------------------------------------------------------------
    # Step 4: Prepare Prompt Templates
    # ------------------------------------------------------------------------
    k_sample = config.rollout.num_response_per_task
    code_eval = False
    
    # Default math problem prompt
    system_prompts = '''<|im_start|>user\n{{problem}}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'''
    
    if config.rollout.start_with_think:
        system_prompts = '''<|im_start|>user\nYou need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|im_end|>\n<|im_start|>assistant\n<think>\n'''
    
    # Adjust prompt based on data type
    dataset = config.dataset.eval_dataset
    pretrained_model = config.model
    
    if config.dataset.data_type == "code":
        code_eval = True
        system_prompts = """<|im_start|>user\n{{problem}}\nYou are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.<|im_end|>\n<|im_start|>assistant\n"""

        if config.rollout.start_with_think:
            system_prompts = """<|im_start|>user\n{{problem}}\nYou are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.<|im_end|>\n<|im_start|>assistant\n<think>\n"""
    
    elif config.dataset.data_type == "option":
        system_prompts = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'''
        if config.rollout.start_with_think:
            system_prompts = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n'''
    
    # ------------------------------------------------------------------------
    # Step 5: Determine Output File Path
    # ------------------------------------------------------------------------
    output_dir = config.paths.result_base
    _os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(config.experiment, 'name') and config.experiment.name:
        exp_name = config.experiment.name
    else:
        if hasattr(config.experiment, 'model_name') and config.experiment.model_name:
            model_name = config.experiment.model_name
        else:
            model_name = _os.path.basename(_os.path.normpath(pretrained_model))
        
        block_size = config.rollout.block_size
        exp_name = (f"eval-{model_name}-{dataset}-"
                   f"mt{config.rollout.max_token}-"
                   f"bs{block_size}-"
                   f"ds{config.rollout.denoising_steps_per_block}")
    
    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    
    if num_node > 1:
        output_file_name = _os.path.join(output_dir, f"outputs-{node_index}-{exp_name}.json")
    else:
        output_file_name = _os.path.join(output_dir, f"outputs-{exp_name}.json")
    
    # ------------------------------------------------------------------------
    # Step 6: Load Dataset and Checkpoint
    # ------------------------------------------------------------------------
    dataset_base_path = config.paths.dataset_base
    dataset_file = _os.path.join(dataset_base_path, dataset, "test", f"{dataset}.json")
    
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    # Multi-node distributed support
    if num_node > 1:
        data = get_data_chunk(data, num_node, node_index)
    
    num = len(data)
    print(f"Current node {node_index}/{num_node} will process {num} questions", flush=True)
    
    # Try to load checkpoint
    checkpoint_data = load_checkpoint(output_file_name)
    if checkpoint_data is not None:
        # Merge checkpoint data
        for i, item in enumerate(checkpoint_data):
            if i < len(data):
                # Keep original question, restore completed responses
                for key in ["full_output", "extracted_output", "response_length", "finish_reason", "prompt", "correctness"]:
                    if key in item:
                        data[i][key] = item[key]
    
    # ------------------------------------------------------------------------
    # Step 7: Initialize Data Structure and Count Completed Items
    # ------------------------------------------------------------------------
    model_path = _os.path.expanduser(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    completed_count = 0
    for i in range(num):
        # Initialize fields if not exist
        if "full_output" not in data[i]:
            data[i]["full_output"] = []
        if "extracted_output" not in data[i]:
            data[i]["extracted_output"] = []
        if "response_length" not in data[i]:
            data[i]["response_length"] = []
        if "finish_reason" not in data[i]:
            data[i]["finish_reason"] = []
        if "correctness" not in data[i]:
            data[i]["correctness"] = []
        
        # Generate prompt if not exist
        if "prompt" not in data[i]:
            current_prompt = system_prompts 
            data[i]["prompt"] = get_prompt(data[i], current_prompt)
        
        # Count completed questions
        if is_question_completed(data[i], k_sample):
            completed_count += 1
    
    remaining_count = num - completed_count
    print(f"✓ Completed: {completed_count}/{num} | Remaining: {remaining_count}/{num}", flush=True)
    
    if remaining_count == 0:
        print("🎉 All data points completed!", flush=True)
        exit(0)
    
    # ------------------------------------------------------------------------
    # Step 8: Configure LMDeploy Engine
    # ------------------------------------------------------------------------
   
    backend_config = PytorchEngineConfig(
        tp=tp,
        dp=dp,
        dtype="bfloat16",
        max_prefill_token_num=4096,
        cache_max_entry_count=0.8,
        enable_prefix_caching=True,
        block_size=64,
        session_len=config.rollout.max_token + 4096,
        
        # Block Diffusion model parameters
        dllm_block_length=config.rollout.block_size,
        dllm_denoising_steps=config.rollout.denoising_steps_per_block,
        dllm_unmasking_strategy=config.rollout.remasking_strategy,
        dllm_confidence_threshold=config.rollout.dynamic_threshold,
        dllm_confidence_upper_threshold=config.rollout.confidence_upper_threshold,
        dllm_confidence_lower_threshold=config.rollout.confidence_lower_threshold
    )
    
    # ------------------------------------------------------------------------
    # Step 9: Create Pipeline
    # ------------------------------------------------------------------------
    print("Loading model...", flush=True)
    pipe = pipeline(
        model_path, 
        backend_config=backend_config,
        chat_template_config=None
    )
    print("Model loaded successfully!", flush=True)
    
    # ------------------------------------------------------------------------
    # Step 10: Configure Generation Parameters
    # ------------------------------------------------------------------------
    stop_token_ids = []
    stop_words = [] 
    
    if OmegaConf.select(config, "rollout.stop_token_list", default=MISSING) is not MISSING:
        for s in config.rollout.stop_token_list:
            if isinstance(s, int):
                stop_token_ids.append(s)
            elif isinstance(s, str):
                stop_words.append(s)
                tid = tokenizer.encode(s, add_special_tokens=False)
                if len(tid) == 1:
                    stop_token_ids.append(tid[0])
    
    stop_words = list(set(stop_words)) if stop_words else None
    stop_token_ids = list(set(stop_token_ids)) if stop_token_ids else None
    
    print(f"[Stop Words Config] stop_words: {stop_words}")
    print(f"[Stop Words Config] stop_token_ids: {stop_token_ids}")
    
    gen_config = GenerationConfig(
        top_p=config.rollout.top_p,
        top_k=config.rollout.top_k,
        temperature=config.rollout.temperature,
        do_sample=True,
        max_new_tokens=config.rollout.max_token,
        stop_words=stop_words,
        stop_token_ids=stop_token_ids,
        ignore_eos=False,
    )
    
    # ------------------------------------------------------------------------
    # Step 11: Question-by-Question Inference (with checkpoint resume support)
    # ------------------------------------------------------------------------
    print(f"\nStarting inference (saving checkpoint after each question)...\n", flush=True)
    
    # Time statistics initialization
    start_time = time.time()
    processed_questions = 0
    total_questions_to_process = remaining_count
    
    try:
        for i in range(num):
            # Skip completed questions
            if is_question_completed(data[i], k_sample):
                continue
            
            # Calculate how many more responses needed
            already_done = len(data[i]["full_output"])
            need_generate = k_sample - already_done
            
            if need_generate <= 0:
                continue
            
            # Calculate progress and time
            processed_questions += 1
            current_progress = (processed_questions / total_questions_to_process) * 100
            elapsed_time = time.time() - start_time
            
            # Estimate remaining time
            if processed_questions > 0:
                avg_time_per_question = elapsed_time / processed_questions
                remaining_questions = total_questions_to_process - processed_questions
                estimated_remaining_time = avg_time_per_question * remaining_questions
            else:
                estimated_remaining_time = -1
            
            # Prepare prompts for this question (repeat need_generate times)
            prompts_for_this_question = [data[i]["prompt"]] * need_generate
            
            # Inference
            question_start_time = time.time()
            try:              
                outputs = pipe(prompts_for_this_question, gen_config=gen_config)
                
                # Process outputs
                for output in outputs:
                    # Extract answer
                    if code_eval:
                        extracted_output = extract_code(output.text)
                    else:
                        extracted_output = extract_final_boxed_answer(output.text)
                    
                    # Store results
                    data[i]["full_output"].append(output.text)
                    data[i]["extracted_output"].append(extracted_output)
                    data[i]["response_length"].append(output.generate_token_len)
                    data[i]["finish_reason"].append(output.finish_reason)
                    
                    # Calculate correctness if ground_truth available
                    if "ground_truth_answer" in data[i]:
                        is_correct = (extracted_output == data[i]["ground_truth_answer"])
                        data[i]["correctness"].append(is_correct)
                
                question_elapsed = time.time() - question_start_time
                
                # Count successful extractions
                success_count = sum(
                    1 for ext in data[i]["extracted_output"][-need_generate:]
                    if ext and not ext.startswith("Can not extract") and not ext.startswith("We can not extract")
                )
                
                # Save checkpoint
                save_checkpoint(data, output_file_name)
                
                # Progress output: [question#] progress | time | estimated remaining | success rate | saved
                print(f"[{i+1}/{num}] {processed_questions}/{total_questions_to_process} ({current_progress:.1f}%) | "
                      f"Time:{format_time(question_elapsed)} | Elapsed:{format_time(elapsed_time)} | "
                      f"ETA:{format_time(estimated_remaining_time)} | "
                      f"Extracted:{success_count}/{need_generate} | Saved", flush=True)
                
            except Exception as e:
                print(f"[{i+1}/{num}] ✗ Inference failed: {e}", flush=True)
                # Save progress even on failure
                save_checkpoint(data, output_file_name)
                # Continue to next question
                continue
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted, progress saved. Resume by running again.", flush=True)
        save_checkpoint(data, output_file_name)
        exit(0)
    
    # ------------------------------------------------------------------------
    # Step 12: Final Save
    # ------------------------------------------------------------------------
    save_checkpoint(data, output_file_name)
    total_elapsed_time = time.time() - start_time
    
    # ------------------------------------------------------------------------
    # Step 13: Print Statistics
    # ------------------------------------------------------------------------
    total_responses = sum(len(d["full_output"]) for d in data)
    total_tokens = sum(sum(d["response_length"]) for d in data)
    mean_response_length = total_tokens / total_responses if total_responses else 0
    
    successful_extractions = sum(
        1 for d in data 
        for ext in d["extracted_output"] 
        if ext and not ext.startswith("Can not extract") and not ext.startswith("We can not extract")
    )
    
    # Count finish_reason
    finish_reasons = {}
    for d in data:
        for reason in d.get("finish_reason", []):
            finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    
    print("\n" + "="*70, flush=True)
    print("Evaluation Complete! Statistics:", flush=True)
    print("="*70, flush=True)
    print(f"  📊 Dataset: {dataset}")
    print(f"  🤖 Model: {model_name if 'model_name' in locals() else pretrained_model}")
    print(f"  🏷️  Experiment Name: {exp_name}")
    print(f"  📝 Number of Questions: {num}")
    print(f"  🔢 Responses per Question: {k_sample}")
    print(f"  ✅ Total Responses: {total_responses}")
    print(f"  📏 Average Response Length: {mean_response_length:.2f} tokens")
    print(f"  ✨ Successful Extractions: {successful_extractions}/{total_responses} "
          f"({100*successful_extractions/max(1, total_responses):.1f}%)")
    print(f"  🏁 Finish Reason Statistics:")
    for reason, count in finish_reasons.items():
        print(f"     - {reason}: {count} ({100*count/max(1,total_responses):.1f}%)")
    print(f"  ⏱️  Total Runtime: {format_time(total_elapsed_time)}")
    if processed_questions > 0:
        avg_time = total_elapsed_time / processed_questions
        print(f"  ⚡ Average Time per Question: {format_time(avg_time)}")
    print(f"  💾 Results Saved to: {output_file_name}")
    print(f"  🎮 GPU Configuration: {dp} GPUs in parallel (DP mode)")
    print("="*70, flush=True)
    
    # Example output
    if len(data) > 0 and data[0].get("full_output"):
        print("\nExample Output (first question):", flush=True)
        print(f"  Question: {data[0]['question'][:100]}...")
        print(f"  Generated: {data[0]['full_output'][0][:200]}...")
        print(f"  Extracted Answer: {data[0]['extracted_output'][0]}")
        print(f"  Response Length: {data[0]['response_length'][0]} tokens")
        print(f"  Finish Reason: {data[0]['finish_reason'][0]}")
        if data[0].get("correctness"):
            print(f"  Correctness: {data[0]['correctness'][0]}")
    
    print("\n🎉 All tasks completed!", flush=True)