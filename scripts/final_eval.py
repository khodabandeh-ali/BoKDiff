import argparse
import os
import pickle
import shutil
import time
import concurrent.futures

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum
from tqdm.auto import tqdm
import subprocess
import time
import random


def aggregate_eval(batch_size, ckpt_path, pre_ckpt):
    fixed_seed = 7
    random.seed(fixed_seed)

    # Delete previously created files and data
    outdir_filename = './outputs_final_evals'
    # List items in the output directory
    for item in os.listdir(outdir_filename):
        item_path = os.path.join(outdir_filename, item)  # Join path to get the full path
        # Check if the item is a directory and starts with "out_temp"
        if os.path.isdir(item_path):
            shutil.rmtree(item_path, ignore_errors=True)
    
    try:
        result_path = "./final_evals"
        shutil.copytree(result_path, result_path +'_history/'+ pre_ckpt.split('/')[-1][:-3], dirs_exist_ok=True)
        shutil.rmtree(result_path)
    except:
        pass


    # Sample a batch for generating new dataset
    indices = random.sample(range(0, 100), batch_size)
    print(indices)

    i = 0
    for data_index in indices:
        start_time = time.time()
        i += 1

        # Generating 10 samples for each index
        sampling_args = ["--ckpt_path", ckpt_path, "--outdir", outdir_filename, "-i", str(data_index), "--prior_mode", "ref_prior"]
        sampling = subprocess.run(["python", "scripts/sample_diffusion_decomp.py", "configs/sampling_drift.yml"] + sampling_args)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f'-------------------------- data: {i} | index: {data_index} | duration: {duration:.0f} (s) -------------------------')

    # Evaluating the result in an aggregated manner
    evaluation_args = ["--docking_mode", "vina", "--aggregate_meta", "True", "--result_path", result_path]
    data_points = subprocess.run(["python", "scripts/evaluate_mol_from_meta_full.py", outdir_filename] + evaluation_args)


if __name__ == '__main__':
    start_time_g = time.time()

    # Evaluation
    batch_size = 30
    # ckpt_path = './pretrained_models/uni_o2_bond.pt'
    pre_ckpt = './pretrained_models/al_inter2-chert-W-radius.pt'
    ckpt_path = './pretrained_models/1000.pt'
    aggregate_eval(batch_size, ckpt_path, pre_ckpt)

    end_time_g = time.time()
    duration = end_time_g - start_time_g
    print(f'--------------------------  | Overall duration: {duration:.0f} (s) |  -------------------------')
