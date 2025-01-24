import argparse
import gzip
import os
import random
import timeit


# from datatrove.pipeline.readers import JsonlReader, ParquetReader
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.doc_utils import execute_meta_operations
from utils.jsonl_rw import JsonlReader, JsonlWriter
from vllm import LLM, SamplingParams

random.seed(42)


# dummy env constants for multi-gpu & multi-node
NODE_GPUS = int(os.environ.get("NODE_GPUS", 8))
NODE_RANK = int(os.environ.get("NODE_RANK", 0))
CUDA_DEVICE = int(os.environ["CUDA_VISIBLE_DEVICES"])
TOTAL_SPLIT = int(os.environ["TOTAL_SPLIT"])
DEST_PATH = "/mbz/users/yuqi.wang/datasets/prox/txt360"
PROGRESS_PATH = "/mbz/users/yuqi.wang/ProX/prox_progress"



def main(args):
    start_time = timeit.default_timer()
    tokenizer = AutoTokenizer.from_pretrained(args.token_template, use_fast=False)
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.bos_token = base_tokenizer.bos_token
    tokenizer.eos_token = base_tokenizer.eos_token
    
    # prepare output dirs
    keep_dir = os.path.join(DEST_PATH, "keep")
    drop_dir = os.path.join(DEST_PATH, "drop")
    os.makedirs(keep_dir, exist_ok=True)
    os.makedirs(drop_dir, exist_ok=True)
    org_file_path = args.data_path.split(",")[CUDA_DEVICE]
    dir_name = os.sep.join(org_file_path.split(os.sep)[-3:-1])
    file_name = os.path.basename(org_file_path)
    os.makedirs(os.path.join(keep_dir, dir_name), exist_ok=True)
    os.makedirs(os.path.join(drop_dir, dir_name), exist_ok=True)
    dest_keep_gz_path = os.path.join(keep_dir, dir_name, file_name)
    dest_drop_gz_path = os.path.join(drop_dir, dir_name, file_name)
    
    # setup vllm
    tp_size = 1
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=2000)
    engine = LLM(
        model=args.model_path,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tp_size
    )

    c, l, drop = 0, 0, 0
    with JsonlWriter(output_filename=dest_keep_gz_path, compression="gzip") as keep_writer:
        with JsonlWriter(output_filename=dest_drop_gz_path, compression="gzip") as drop_writer:
            with JsonlReader(org_file_path, compression="gzip", batch_size=args.batch_size) as reader:
                for bi, batch in enumerate(tqdm(reader, desc="Reading data")):
                    rets, texts = [], []
                    for sample in tqdm(batch, total=len(batch), unit="tokenizing"):
                        user_msg = sample["text"]
                        l += len(user_msg)
                        total_msg = tokenizer.apply_chat_template(
                            [
                                {
                                    "role": "system",
                                    "content": "You are a helpful, respectful and honest assistant.",
                                },
                                {"role": "user", "content": user_msg},
                            ],
                            add_generation_prompt=True,
                            # tokenize=False,
                            truncation=True,
                            max_length=2000,
                        )
                        texts.append(user_msg)
                        rets.append(total_msg)        
                    outputs = engine.generate(
                        sampling_params=sampling_params, prompt_token_ids=rets
                    )
                    outputs = [item.outputs[0].text.strip(" ") for item in outputs]
                    assert len(outputs) == len(texts)
                    c += len(outputs)
                    for i, output in enumerate(outputs):
                        text = execute_meta_operations(texts[i], output)
                        if not text:
                            drop += 1
                            drop_writer.write_batch(texts[i] + "\n")
                        else:
                            keep_writer.write_batch(texts[i] + "\n")
    with open(PROGRESS_PATH, "a") as f:
        f.write(org_file_path + "\n")
    elapsed = timeit.default_timer() - start_time
    if c != 0:
        print(l)
        print(f"refining time elapsed for {org_file_path}: {elapsed:.4f}s")
        print(f"{c} docs in total, dropped {drop/c:.2%} documents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--token_template",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
    )
    parser.add_argument("--batch_size", type=int, default=1000000)
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit the number of samples to process, for debugging.",
    )
    parser.add_argument("--data_format", type=str, default="parquet")
    parser.add_argument("--dataset_name", type=str, default="redpajama-v2")
    args = parser.parse_args()
    main(args)
