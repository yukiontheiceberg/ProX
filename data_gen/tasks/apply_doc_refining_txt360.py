import argparse
import gzip
import os
import random
import shutil

from datasets import Dataset
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.data_utils import get_adapter_func, normalize_lined_text, split_into_batches
from utils.doc_utils import execute_meta_operations
from vllm import LLM, SamplingParams

from data_gen.configs import GentaskConfig

random.seed(42)


# dummy env constants for multi-gpu & multi-node
NODE_GPUS = int(os.environ.get("NODE_GPUS", 8))
NODE_RANK = int(os.environ.get("NODE_RANK", 0))
CUDA_DEVICE = int(os.environ["CUDA_VISIBLE_DEVICES"])
TOTAL_SPLIT = int(os.environ["TOTAL_SPLIT"])
DEST_PATH = "/mbz/shared/yuqi.wang/TxT360/common-crawl/prox"


def main(args):
    # load config
    # config = GentaskConfig().from_yaml(args.config_path)
    tokenizer = AutoTokenizer.from_pretrained(args.token_template, use_fast=False)
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.bos_token = base_tokenizer.bos_token
    tokenizer.eos_token = base_tokenizer.eos_token
    keep_dir = os.path.join(DEST_PATH, "keep")
    drop_dir = os.path.join(DEST_PATH, "drop")
    os.makedirs(keep_dir, exist_ok=True)
    os.makedirs(drop_dir, exist_ok=True)

    # prepare data
    if args.data_format == "parquet":
        data_reader = ParquetReader(
            data_folder=args.data_path,
            file_progress=True,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    elif args.data_format == "jsonl.gz":
        data_path = args.data_path.split(",")[CUDA_DEVICE]
        print(f"data_path {data_path}")
        data_reader = JsonlReader(
            data_folder=data_path,
            file_progress=True,
            # adapter=get_adapter_func(args.dataset_name),
            limit=args.limit,
        )

    arguments, org_file_path = [], None
    for idx, doc in enumerate(
        data_reader.run(
            rank=CUDA_DEVICE + NODE_RANK * NODE_GPUS, world_size=TOTAL_SPLIT
        )
    ):
        arguments.append({"text": doc.text})
        org_file_path = doc.metadata["file_path"]
        assert doc.metadata["file_path"] == org_file_path

    tp_size = 1
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=2000)
    engine = LLM(
        model=args.model_path,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tp_size
    )
    for i, batch in enumerate(tqdm(arguments)):
        rets = []
        user_msg = batch["text"]
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
        rets.append(total_msg)
        outputs = engine.generate(
            sampling_params=sampling_params, prompt_token_ids=rets
        )
        outputs = [item.outputs[0].text.strip(" ") for item in outputs]
        dir_name = os.sep.join(org_file_path.split(os.sep)[-3:-1])
        file_name = os.path.basename(org_file_path)
        keep_any, drop_any = False, False
        keep_lines, drop_lines = [], []
        for output in outputs:
            text = execute_meta_operations(batch["text"], output)
            if not text:
                drop_any = True
                drop_lines.append(batch["text"] + "\n")
            else:
                keep_any = True
                keep_lines.append(batch["text"] + "\n")
        if keep_any:
            os.makedirs(os.path.join(keep_dir, dir_name), exist_ok=True)
            dest_keep_gz_path = os.path.join(keep_dir, dir_name, file_name)
            # write to keep folder
            print("write to keep")
            with gzip.open(dest_keep_gz_path, "at") as drop_gz:
                drop_gz.writelines(keep_lines)
        if drop_any:
            os.makedirs(os.path.join(drop_dir, dir_name), exist_ok=True)
            dest_drop_gz_path = os.path.join(drop_dir, dir_name, file_name)
            # write to drop folder
            print("write to drop")
            with gzip.open(dest_drop_gz_path, "at") as drop_gz:
                drop_gz.writelines(drop_lines)


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
