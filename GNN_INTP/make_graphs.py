import subprocess
import os
import json
import time
import support_functions


datasets = ["ABO_res250_reg4c", "SAQN_res250_reg4c", ]
folds = [4, ]
houldouts = [0, 1, 2, 3, ]
mds = [0, 20, 50, ]

hpc_header = \
f"""#!/bin/bash
#SBATCH --job-name=data
#SBATCH --error=./Datasets/slurm/logs/%x.%j.err
#SBATCH --output=./Datasets/slurm/logs/%x.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anonym
#SBATCH --export=ALL

#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=180000mb
#SBATCH --time=72:00:00

eval \"$(conda shell.bash hook)\"
conda activate $Path to your conda environment$
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$Path to your conda environment$/lib

job_id=$SLURM_JOB_ID
"""

ws_path = '$Path to your workspace$'

orders = []
for d in datasets:
    for f in folds:
        for h in houldouts:
            orders.append({"dataset": d, "fold": f, "holdout": h, "call_name": "train", "mask_distance": -1})
            orders.append({"dataset": d, "fold": f, "holdout": h, "call_name": "test", "mask_distance": 0})
            orders.append({"dataset": d, "fold": f, "holdout": h, "call_name": "eval", "mask_distance": 0})
            orders.append({"dataset": d, "fold": f, "holdout": h, "call_name": "eval", "mask_distance": 20})
            orders.append({"dataset": d, "fold": f, "holdout": h, "call_name": "eval", "mask_distance": 50})


for order in orders:
    # folder cleanup
    support_functions.build_folder_and_clean(f"{ws_path}/{order['dataset']}/pkl/{order['fold']}_{order['holdout']}_{order['call_name']}_{order['mask_distance']}/")

    time.sleep(5)
    print(f"Cleaned folder {ws_path}/{order['dataset']}/pkl/{order['fold']}_{order['holdout']}_{order['call_name']}_{order['mask_distance']}/")


for order in orders:
    while True:
        cmd = f"squeue"
        status = subprocess.check_output(cmd, shell=True).decode()
        lines = status.split('\n')[1:-1]
        if len(lines) <= 49:
            break
        else:
            time.sleep(60)

    # then build up the slurm script
    job_script = f"{hpc_header}\npython make_g.py '{json.dumps(order)}'"

    # Write job submission script to a file
    sb_path = f"./Datasets/slurm/{order['dataset']}_{order['fold']}_{order['holdout']}_{order['call_name']}_{order['mask_distance']}.sbatch"
    with open(sb_path, "w") as f:
        f.write(job_script)

    # Submit job to Slurm system and get job ID
    cmd = "sbatch " + sb_path
    output = subprocess.check_output(cmd, shell=True).decode().strip()
    job_id = output.split()[-1]

    time.sleep(5)

    print(f"Launch {job_id}")
