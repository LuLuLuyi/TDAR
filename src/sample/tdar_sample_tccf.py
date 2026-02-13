"""
LMDeploy Distributed Inference Evaluation Script - Two-Stage Sequential Inference
Stage 1: Use large block length to generate until </think>
Stage 2: Use small block length to continue generating the remaining part
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


def get_think_end_token_ids(tokenizer):
    """
    Get token ID sequence for </think> tag
    """
    think_end = "</think>"
    token_ids = tokenizer.encode(think_end, add_special_tokens=False)
    return token_ids


# ============================================================================
# Main Program - Two-Stage Sequential Inference
# ============================================================================
if __name__ == "__main__":
    
    # ------------------------------------------------------------------------
    # Step 1: Load Configuration
    # ------------------------------------------------------------------------
    config = get_config()
    
    print("="*70)
    print("🎯 Two-Stage Sequential Inference Configuration")
    print("="*70)
    print(f"  Think Stage - block_size: {config.rollout.think_block_size}")
    print(f"  Think Stage - denoising_steps: {config.rollout.think_denoising_steps}")
    print(f"  Final Stage - block_size: {config.rollout.final_block_size}") 
    print(f"  Final Stage - denoising_steps: {config.rollout.final_denoising_steps}")
    print(f"  max_token: {config.rollout.max_token}")
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
    
    # Default math problem prompt - with think tag
    system_prompts = '''<|im_start|>user\n{{problem}}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>'''
    
    if not config.rollout.start_with_think:
        system_prompts = '''<|im_start|>user\n{{problem}}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'''
    
    # Adjust prompt based on data type
    dataset = config.dataset.eval_dataset
    pretrained_model = config.model
    
    # Code evaluation prompt
    if config.dataset.data_type == "code":
        code_eval = True
        system_prompts = """<|im_start|>user\n{{problem}}\nYou are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.<|im_end|>\n<|im_start|>assistant\n"""
        
        if config.rollout.start_with_think:
            system_prompts = """<|im_start|>user\n{{problem}}\nYou are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.<|im_end|>\n<|im_start|>assistant\n<think>\n"""
    
    # Multiple choice evaluation prompt
    elif config.dataset.data_type == "option":
        system_prompts = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'''
        if config.rollout.start_with_think:
            system_prompts = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>'''
    
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
        
        exp_name = (f"eval-{model_name}-{dataset}-"
                   f"think{config.rollout.think_block_size}-"
                   f"final{config.rollout.final_block_size}-"
                   f"mt{config.rollout.max_token}")
    
    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    
    if num_node > 1:
        output_file_name = _os.path.join(output_dir, f"outputs-{node_index}-{exp_name}.json")
    else:
        output_file_name = _os.path.join(output_dir, f"outputs-{exp_name}.json")
    
    # ------------------------------------------------------------------------
    # Step 6: Load Dataset
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
    
    # ------------------------------------------------------------------------
    # Step 7: Initialize Data Structure and Tokenizer
    # ------------------------------------------------------------------------
    model_path = _os.path.expanduser(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Get </think> token IDs for stopping condition
    think_end_token_ids = get_think_end_token_ids(tokenizer)
    print(f"</think> token IDs: {think_end_token_ids}")
    
    # Initialize data structure
    for i in range(num):
        data[i]["full_output"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        data[i]["finish_reason"] = []
        data[i]["used_final_stage"] = []
        data[i]["correctness"] = []

        # Add stage-wise outputs
        data[i]["stage1_output"] = []  # Stage 1 output (thinking part)
        data[i]["stage2_output"] = []  # Stage 2 output (final answer part)

        # Add stage token statistics
        data[i]["stage1_tokens"] = []  # Stage 1 token count
        data[i]["stage2_tokens"] = []  # Stage 2 token count
        
        # Use unified system_prompts
        data[i]["prompt"] = get_prompt(data[i], system_prompts)
    
    print(f"Total questions: {num}, generating {k_sample} samples per question")
    
    # ------------------------------------------------------------------------
    # Step 8: Configure Stop Words
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
    
    # Think stage: add </think> as stop word
    think_stop_words = stop_words + ["</think>"] if stop_words else ["</think>"]
    think_stop_token_ids = stop_token_ids + think_end_token_ids if stop_token_ids else think_end_token_ids
    
    think_stop_words = list(set(think_stop_words)) if think_stop_words else None
    think_stop_token_ids = list(set(think_stop_token_ids)) if think_stop_token_ids else None
    
    # ------------------------------------------------------------------------
    # Stage 1: Think Stage - Large block length inference until </think>
    # ------------------------------------------------------------------------
    print("\n" + "="*70)
    print("🧠 Stage 1: Think Stage - Using large block length to generate until </think>")
    print("="*70)
    
    # Think stage engine config (large block length)
    think_backend_config = PytorchEngineConfig(
        tp=tp,
        dp=dp,
        dtype="bfloat16",
        max_prefill_token_num=4096,
        cache_max_entry_count=0.8,
        enable_prefix_caching=True,
        block_size=64,
        session_len=config.rollout.max_token + 4096,
        
        # Block Diffusion model parameters - think stage uses large block
        dllm_block_length=config.rollout.think_block_size,
        dllm_denoising_steps=config.rollout.think_denoising_steps,
        dllm_unmasking_strategy=config.rollout.remasking_strategy,
        dllm_confidence_threshold=config.rollout.dynamic_threshold,
        dllm_confidence_upper_threshold=config.rollout.confidence_upper_threshold,
        dllm_confidence_lower_threshold=config.rollout.confidence_lower_threshold
    )
    
    # Think stage generation config
    think_gen_config = GenerationConfig(
        top_p=config.rollout.top_p,
        top_k=config.rollout.top_k,
        temperature=config.rollout.temperature,
        do_sample=True,
        repetition_penalty=config.rollout.repetition_penalty,
        max_new_tokens=config.rollout.max_token,
        stop_words=think_stop_words,
        stop_token_ids=think_stop_token_ids,
        ignore_eos=False,
    )
    
    print("Loading think stage model...", flush=True)
    think_pipe = pipeline(
        model_path, 
        backend_config=think_backend_config,
        chat_template_config=None
    )
    print("Think stage model loaded successfully!", flush=True)
    
    # Execute think stage inference - loop by question
    stage1_start_time = time.time()
    print(f"Starting think stage inference, processing {num} questions, {k_sample} samples per question...\n", flush=True)
    
    # Store all think stage results
    all_think_results = []
    
    for i in range(num):
        # Prepare prompts for this question (repeat k_sample times)
        prompts_for_this_question = [data[i]["prompt"]] * k_sample
        
        # Inference
        question_start_time = time.time()
        try:
            outputs = think_pipe(prompts_for_this_question, gen_config=think_gen_config)
            
            # Process outputs
            for output in outputs:
                text = output.text + "Let's check if there are any mistakes and give the final answer."
                
                all_think_results.append({
                    'text': text,
                    'think_token_len': output.generate_token_len,
                    'finish_reason': output.finish_reason,
                    'original_index': i  # Record which question it belongs to
                })
            
            question_elapsed = time.time() - question_start_time
            
            # Simple progress output
            print(f"[{i+1}/{num}] Think stage completed | Time:{format_time(question_elapsed)} | "
                  f"Generated samples:{len(outputs)}", flush=True)
            
        except Exception as e:
            print(f"[{i+1}/{num}] ✗ Think stage failed: {e}", flush=True)
            # Create empty results for failure case
            for _ in range(k_sample):
                all_think_results.append({
                    'text': "",
                    'think_token_len': 0,
                    'finish_reason': "error",
                    'original_index': i
                })
            continue
    
    stage1_time = time.time() - stage1_start_time
    print(f"\n✅ Think stage completed, time elapsed: {format_time(stage1_time)}", flush=True)
    print(f"Total generated samples: {len(all_think_results)}", flush=True)
    
    # Clean up think stage model
    think_pipe.close()
    del think_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ------------------------------------------------------------------------
    # Stage 2: Final Stage - Small block length to continue generating
    # ------------------------------------------------------------------------
    print("\n" + "="*70)
    print("✍️  Stage 2: Final Stage - Using small block length to continue generating")
    print("="*70)
    
    # Final stage engine config (small block length)
    final_backend_config = PytorchEngineConfig(
        tp=tp,
        dp=dp,
        dtype="bfloat16",
        max_prefill_token_num=4096,
        cache_max_entry_count=0.8,
        enable_prefix_caching=True,
        block_size=64,
        session_len=config.rollout.max_token + 16384,
        
        # Block Diffusion model parameters - final stage uses small block
        dllm_block_length=config.rollout.final_block_size,
        dllm_denoising_steps=config.rollout.final_denoising_steps,
        dllm_unmasking_strategy=config.rollout.remasking_strategy,
        dllm_confidence_threshold=config.rollout.dynamic_threshold,
        dllm_confidence_upper_threshold=config.rollout.confidence_upper_threshold,
        dllm_confidence_lower_threshold=config.rollout.confidence_lower_threshold
    )
    
    # Final stage generation config
    final_gen_config = GenerationConfig(
        top_p=config.rollout.top_p,
        top_k=config.rollout.top_k,
        temperature=config.rollout.temperature,
        do_sample=True,
        repetition_penalty=config.rollout.repetition_penalty,
        max_new_tokens=4096,  # Token budget for final stage
        stop_words=stop_words,
        stop_token_ids=stop_token_ids,
        ignore_eos=False,
    )
    
    print("Loading final stage model...", flush=True)
    final_pipe = pipeline(
        model_path, 
        backend_config=final_backend_config,
        chat_template_config=None
    )
    print("Final stage model loaded successfully!", flush=True)
    
    # Execute final stage inference - loop by question
    stage2_start_time = time.time()
    print(f"Starting final stage continuation, processing {num} questions...\n", flush=True)
    
    # Store all final stage results
    all_final_results = []
    
    # Process by question
    current_idx = 0
    for i in range(num):
        # Get all think results for this question (k_sample results)
        think_results_for_question = all_think_results[current_idx:current_idx + k_sample]
        
        # Prepare continuation prompts
        continuation_prompts = []
        for think_result in think_results_for_question:
            think_output = think_result['text']
            # For continuation, use original prompt plus think stage output
            continuation_prompts.append(data[i]["prompt"] + think_output)
        
        # Inference
        question_start_time = time.time()
        try:
            outputs = final_pipe(continuation_prompts, gen_config=final_gen_config)
            
            # Process outputs
            for output in outputs:
                all_final_results.append({
                    'text': output.text,
                    'final_token_len': output.generate_token_len,
                    'finish_reason': output.finish_reason,
                    'original_index': i
                })
            
            question_elapsed = time.time() - question_start_time
            
            # Simple progress output
            print(f"[{i+1}/{num}] Final stage completed | Time:{format_time(question_elapsed)} | "
                  f"Generated samples:{len(outputs)}", flush=True)
            
        except Exception as e:
            print(f"[{i+1}/{num}] ✗ Final stage failed: {e}", flush=True)
            # Create empty results for failure case
            for _ in range(k_sample):
                all_final_results.append({
                    'text': "",
                    'final_token_len': 0,
                    'finish_reason': "error",
                    'original_index': i
                })
            continue
        
        current_idx += k_sample
    
    stage2_time = time.time() - stage2_start_time
    print(f"\n✅ Final stage completed, time elapsed: {format_time(stage2_time)}", flush=True)
    
    # Clean up final stage model
    final_pipe.close()
    del final_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ------------------------------------------------------------------------
    # Step 9: Merge Results and Save
    # ------------------------------------------------------------------------
    print("\nMerging two-stage results...", flush=True)
    
    for result_idx, (think_result, final_result) in enumerate(zip(all_think_results, all_final_results)):
        idx = think_result['original_index']
        
        # Stage 1 output (thinking part)
        stage1_output = think_result['text']
        stage1_tokens = think_result['think_token_len']
        
        # Stage 2 output
        stage2_output = final_result['text']
        stage2_tokens = final_result['final_token_len']
        
        # Complete output
        full_output = stage2_output  # Final stage already contains complete text
        
        # Extract answer
        if code_eval:
            extracted_output = extract_code(full_output)
        else:
            extracted_output = extract_final_boxed_answer(full_output)
        
        # Store results
        data[idx]["full_output"].append(full_output)
        data[idx]["stage1_output"].append(stage1_output)
        data[idx]["stage2_output"].append(stage2_output)
        data[idx]["extracted_output"].append(extracted_output)
        data[idx]["response_length"].append(stage1_tokens + stage2_tokens)
        data[idx]["finish_reason"].append(final_result['finish_reason'])
        data[idx]["used_final_stage"].append(True)
        data[idx]["stage1_tokens"].append(stage1_tokens)
        data[idx]["stage2_tokens"].append(stage2_tokens)
        
        # Calculate correctness if ground_truth available
        if "ground_truth_answer" in data[idx]:
            is_correct = (extracted_output == data[idx]["ground_truth_answer"])
            data[idx]["correctness"].append(is_correct)
    
    # ------------------------------------------------------------------------
    # Step 10: Save Final Results
    # ------------------------------------------------------------------------
    print("Saving final results...", flush=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # ------------------------------------------------------------------------
    # Step 11: Print Statistics
    # ------------------------------------------------------------------------
    total_responses = sum(len(d["full_output"]) for d in data)
    total_tokens = sum(sum(d["response_length"]) for d in data)
    mean_response_length = total_tokens / total_responses if total_responses else 0

    successful_extractions = sum(
        1 for d in data 
        for ext in d["extracted_output"] 
        if ext and not ext.startswith("Can not extract") and not ext.startswith("We can not extract")
    )

    # Statistics for stage tokens
    all_stage1_tokens = [token for d in data for token in d.get("stage1_tokens", [])]
    all_stage2_tokens = [token for d in data for token in d.get("stage2_tokens", [])]

    avg_stage1_tokens = sum(all_stage1_tokens) / len(all_stage1_tokens) if all_stage1_tokens else 0
    avg_stage2_tokens = sum(all_stage2_tokens) / len(all_stage2_tokens) if all_stage2_tokens else 0

    # Count finish_reason
    finish_reasons = {}
    for d in data:
        for reason in d.get("finish_reason", []):
            finish_reasons[reason] = finish_reasons.get(reason, 0) + 1

    total_time = stage1_time + stage2_time

    print("\n" + "="*70, flush=True)
    print("🎯 Two-Stage Sequential Evaluation Complete! Statistics:", flush=True)
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
    print(f"  📊 Stage Token Statistics:")
    print(f"     - Stage 1 Average: {avg_stage1_tokens:.2f} tokens")
    print(f"     - Stage 2 Average: {avg_stage2_tokens:.2f} tokens")
    print(f"  🏁 Finish Reason Statistics:")
    for reason, count in finish_reasons.items():
        print(f"     - {reason}: {count} ({100*count/max(1,total_responses):.1f}%)")
    print(f"  ⏱️  Total Runtime: {format_time(total_time)}")
    print(f"     - Think Stage: {format_time(stage1_time)}")
    print(f"     - Final Stage: {format_time(stage2_time)}")
    print(f"  💾 Results Saved to: {output_file_name}")
    print(f"  🎮 GPU Configuration: {dp} GPUs in parallel (DP mode)")
    print(f"  🔧 Two-Stage Configuration:")
    print(f"     - Think Stage: block={config.rollout.think_block_size}, denoise={config.rollout.think_denoising_steps}")
    print(f"     - Final Stage: block={config.rollout.final_block_size}, denoise={config.rollout.final_denoising_steps}")
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
    
    print("\n🎉 All two-stage sequential inference tasks completed!", flush=True)