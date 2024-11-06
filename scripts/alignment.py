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

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def high_reward_dataset(batch_size, ckpt_path):
    # Delete previously created files and data
    output_dir = './outputs'
    # List items in the output directory
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)  # Join path to get the full path
        # Check if the item is a directory and starts with "out_temp"
        if os.path.isdir(item_path) and item.startswith("out_temp"):
            shutil.rmtree(item_path, ignore_errors=True)
    shutil.rmtree("./new_data_temp", ignore_errors=True)
    shutil.rmtree("./tmp", ignore_errors=True)
    shutil.rmtree("./data/crossdocked_samples_processed", ignore_errors=True)
    remove_file("./data/crossdocked_samples_processed_full_ref_prior_aromatic_name2id.pt")
    remove_file("./data/crossdocked_samples_processed_full_ref_prior_aromatic.lmdb")
    remove_file("./data/crossdocked_samples_processed_full_ref_prior_aromatic.lmdb-lock")

    # Sample a batch for generating new dataset
    indices = random.sample(range(1, 99165), batch_size)
    with open('./data/train_index.pkl', 'rb') as f:
        train_index = pickle.load(f)

    i = 0
    for data_index in indices:
        start_time = time.time()
        outdir_filename = "./outputs/out_temp"+str(data_index)
        i += 1

        # Generating 10 samples for each index
        sampling_args = ["--ckpt_path", ckpt_path, "--outdir", outdir_filename, "-i", str(data_index), "--prior_mode", "ref_prior"]
        sampling = subprocess.run(["python", "scripts/sample_modified.py", "configs/sampling_drift.yml"] + sampling_args)

        # Evaluating and picking best sample based on reward
        # Change the reward function weights with --weights "qed,sa,vina" (int weights)
        # Generating new folder with target protein and best sdf file in ./new_data_temp
        protein_src_file = train_index[data_index]['src_protein_filename']
        ligand_src_file = train_index[data_index]['src_ligand_filename']
        evaluation_args = ["--docking_mode", "vina", "--aggregate_meta", "True", "--result_path", "./eval_temp",
        "--protein_path", protein_src_file, "--ligand_path", ligand_src_file, "--weights", "1,1,2"] #qed,sa,vina
        data_points = subprocess.run(["python", "scripts/evaluate_mol_modified.py", outdir_filename] + evaluation_args)

        end_time = time.time()
        duration = end_time - start_time
        print(f'-------------------------- data: {i} | index: {data_index} | duration: {duration:.0f} (s) -------------------------')


if __name__ == '__main__':
    alignment_iter = 2
    random.seed(alignment_iter + 94)

    if alignment_iter == 1:
        ckpt_path = 'pretrained_models/uni_o2_bond.pt'
    else:
        ckpt_name = 'al_iter' + str(alignment_iter-1) + '.pt'
        ckpt_path = 'pretrained_models/' + ckpt_name

    # Data collection
    batch_size = 128
    high_reward_dataset(batch_size, ckpt_path)
    
    # Pre-processing the Data
    preprocessing_args = ["--dest", "./data/crossdocked_samples_processed"]
    preprocess = subprocess.run(["python", "scripts/preprocess_new_data.py", "configs/preprocessing/crossdocked_samples.yml"] + preprocessing_args)

    # Training
    # training_args = ["--alignment_iter", str(alignment_iter)]
    # training = subprocess.run(["python", "scripts/train_aligned_decomp.py", "configs/alignment_training.yml"] + training_args)