import os
import subprocess


SOURCE_PATH = "/vast/shared/pretraining-data/raw_data/TxT360_Final/common-crawl/"
DEST_PATH = "/mbz/shared/yuqi.wang/TxT360/common-crawl/prox"
PROGRESS_PATH = "/mbz/users/yuqi.wang/ProX/prox_progress"

with open(PROGRESS_PATH, "r") as f:
    processed = f.read().splitlines()


doc_list = []
for d1 in os.listdir(SOURCE_PATH):
    p1 = os.path.join(SOURCE_PATH, d1)
    for d2 in os.listdir(p1):
        p2 = os.path.join(p1, d2)
        print(f" current dir {p2}")
        if p2 in processed:
            continue
        doc_list.append(p2)
        if len(doc_list) == 8:
            # print(doc_list)
            subprocess.run(["sbatch", "./scripts/data_gen/doc_refining_txt360.sh", ",".join(doc_list)], check=True)
            with open(PROGRESS_PATH, "a") as f:
                f.write(p2 + "\n")
            doc_list = []
            exit()
