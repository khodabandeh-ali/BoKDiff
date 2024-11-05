# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


import argparse
import os
import shutil

import numpy as np
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from copy import deepcopy

import sys
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils import misc
from utils.evaluation import scoring_func
from utils.evaluation.docking import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask
from multiprocessing import Pool
from functools import partial
from glob import glob
from utils.evaluation import eval_bond_length
from utils.transforms import get_atomic_number_from_index
import utils.transforms as trans
from rdkit import Chem
from utils.reconstruct import reconstruct_from_generated_with_bond
from openbabel import pybel
from openbabel import openbabel as ob

def mol_to_sdf(mol, file_path):
    """
    Convert an RDKit Mol object to an SDF file.
    
    Args:
    mol (rdkit.Chem.rdchem.Mol): The molecule to convert
    file_path (str): The path where the SDF file should be saved
    """
    sdf_string = Chem.MolToMolBlock(mol)
    
    with open(file_path, 'w') as f:
        f.write(sdf_string)
        f.write('$$$$\n')  # SDF file separator


# Function to clean and write the PDB file without Vina metadata lines
def clean_pdb(input_pdb, output_pdb):
    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        for line in infile:
            # Exclude non-atom metadata lines
            if line.startswith("ATOM") or line.startswith("HETATM"):
                outfile.write(line)

def get_pose_sdf(pdb_file = "docked_pose.pdb", sdf_file = "final_docked_pose.sdf"):
    # Clean the PDB file
    clean_pdb(pdb_file, "cleaned_docked_pose.pdb")

    # Load the PDB file
    pdb_file2 = "cleaned_docked_pose.pdb"
    mol = next(pybel.readfile("pdb", pdb_file2))
    docked_mol = mol.OBMol

    # Remove atoms without bonds and save the updated docked pose
    # docked_mol.DeleteHydrogens()
    # flag = False
    # while not flag:
    #     i = 0
    #     for atom in ob.OBMolAtomIter(docked_mol):
    #         a = pybel.Atom(atom)
    #         if a.degree == 0:
    #             docked_mol.DeleteAtom(atom)
    #             i += 1
    #     if i == 0:
    #         flag = True

    # Save the modified docked pose to SDF format
    mol.write("sdf", sdf_file, overwrite=True)

    if os.path.exists(pdb_file):
        os.remove(pdb_file)

    if os.path.exists("cleaned_docked_pose.pdb"):
        os.remove("cleaned_docked_pose.pdb")


def transform_to_original(original_sdf, docked_sdf, aligned_sdf):
    # Load the original ligand and docked ligand
    original_ligand = Chem.MolFromMolFile(original_sdf, removeHs=False)
    docked_ligand = Chem.MolFromMolFile(docked_sdf, removeHs=False)

    # Ensure both ligands are loaded correctly
    if original_ligand is None or docked_ligand is None:
        raise ValueError("Failed to load one of the ligand structures.")

    # Remove standalone hydrogens in both molecules
    def remove_dangling_hydrogens(mol):
        """Remove hydrogens without neighbors."""
        atoms_to_remove = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0:
                atoms_to_remove.append(atom.GetIdx())
        mol = Chem.RWMol(mol)
        for idx in reversed(atoms_to_remove):  # Remove from end to maintain indexing
            mol.RemoveAtom(idx)
        return mol

    docked_ligand = remove_dangling_hydrogens(docked_ligand)
    docked_ligand = Chem.RemoveHs(docked_ligand)

    # Create a mapping of original ligand atom indices to docked ligand atom indices
    original_atom_indices = {atom.GetIdx(): atom.GetAtomicNum() for atom in original_ligand.GetAtoms()}
    docked_atom_indices = {atom.GetIdx(): atom.GetSymbol() for atom in docked_ligand.GetAtoms()}

    # This dictionary will hold the replacements
    replacement_mapping = {}

    for docked_idx, docked_symbol in docked_atom_indices.items():
        if docked_symbol == '*':
            # Find the closest corresponding atom in the original ligand
            # You may need to implement a logic here to find the best match
            # For simplicity, let's assume we're replacing with a Carbon atom for demonstration
            if docked_idx in original_atom_indices:
                replacement_mapping[docked_idx] = original_atom_indices[docked_idx]

    for atom in docked_ligand.GetAtoms():
        if atom.GetSymbol() == '*':
            # Get the index of the current docked atom
            docked_idx = atom.GetIdx()
            # Replace `*` with the corresponding atom type from the original ligand
            if docked_idx in replacement_mapping:
                # Set the atomic number based on the atom type from the mapping
                atom.SetAtomicNum(replacement_mapping[docked_idx])

    if original_ligand.GetNumAtoms() != docked_ligand.GetNumAtoms():
        print(original_ligand.GetNumAtoms())
        print(docked_ligand.GetNumAtoms())
        raise ValueError("Atom count mismatch after adding hydrogens.")

    # Get conformers for transformation
    conf_orig = original_ligand.GetConformer()
    conf_dock = docked_ligand.GetConformer()

    # Calculate centroids
    def calculate_centroid(conf, mol):
        return np.mean([np.array(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], axis=0)

    centroid_orig = calculate_centroid(conf_orig, original_ligand)
    centroid_dock = calculate_centroid(conf_dock, docked_ligand)

    # Center molecules on origin
    for i in range(original_ligand.GetNumAtoms()):
        pos = conf_orig.GetAtomPosition(i)
        conf_orig.SetAtomPosition(i, pos - centroid_orig)
    for i in range(docked_ligand.GetNumAtoms()):
        pos = conf_dock.GetAtomPosition(i)
        conf_dock.SetAtomPosition(i, pos - centroid_dock)

    # Translate original ligand to docked centroid
    for i in range(original_ligand.GetNumAtoms()):
        pos = conf_orig.GetAtomPosition(i)
        conf_orig.SetAtomPosition(i, pos + centroid_dock)

    # Save the aligned ligand
    Chem.MolToMolFile(original_ligand, aligned_sdf)


def min_max_scaler(values):
    min_value = min(values)
    max_value = max(values)

    # Scale the values using Min-Max scaling
    scaled_values = [(x - min_value) / (max_value - min_value) for x in values]
    
    return scaled_values


def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


def eval_single_datapoint(index, id, args):
    if isinstance(index, dict):
        # reference set
        index = [index]

    ligand_filename = index[0]['ligand_filename']
    num_samples = len(index[:100])
    results = []
    n_eval_success = 0
    all_pair_dist, all_bond_dist = [], []
    for sample_idx, sample_dict in enumerate(tqdm(index[:num_samples], desc='Eval', total=num_samples)):
        if sample_dict['mol'] is None:
            try:
                pred_atom_type = trans.get_atomic_number_from_index(sample_dict['pred_v'], mode='basic')
                mol = reconstruct_from_generated_with_bond(
                    xyz=sample_dict['pred_pos'],
                    atomic_nums=pred_atom_type,
                    bond_index=sample_dict['pred_bond_index'],
                    bond_type=sample_dict['pred_bond_type']
                )
                smiles = Chem.MolToSmiles(mol)
            except:
                logger.warning('Reconstruct failed %s' % f'{sample_idx}')
                mol, smiles = None, None
        else:
            mol = sample_dict['mol']
            smiles = sample_dict['smiles']

        if mol is None or '.' in smiles:
            continue
        # mol_to_sdf(mol, './sdf_files/test.sdf')
        # break

        # chemical and docking check
        try:
            chem_results = scoring_func.get_chem(mol)
            if args.docking_mode == 'qvina':
                vina_task = QVinaDockingTask.from_generated_mol(mol, ligand_filename, protein_root=args.protein_root)
                vina_results = vina_task.run_sync()
            elif args.docking_mode == 'vina':
                vina_task = VinaDockingTask.from_generated_mol(mol, ligand_filename, protein_root=args.protein_root)
                vina_results = vina_task.run(mode='dock')
            elif args.docking_mode in ['vina_full', 'vina_score']:
                vina_task = VinaDockingTask.from_generated_mol(deepcopy(mol),
                                                               ligand_filename, protein_root=args.protein_root)
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                vina_results = {
                    'score_only': score_only_results,
                    'minimize': minimize_results
                }
                if args.docking_mode == 'vina_full':
                    dock_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                    vina_results.update({
                        'dock': dock_results,
                    })

            elif args.docking_mode == 'none':
                vina_results = None
            else:
                raise NotImplementedError
            n_eval_success += 1
        except Exception as e:
            logger.warning('Evaluation failed for %s' % f'{sample_idx}')
            print(str(e))
            continue

        pred_pos, pred_v = sample_dict['pred_pos'], sample_dict['pred_v']
        pred_atom_type = get_atomic_number_from_index(pred_v, mode='add_aromatic')
        pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
        all_pair_dist += pair_dist

        bond_dist = eval_bond_length.bond_distance_from_mol(mol)
        all_bond_dist += bond_dist

        results.append({
            **sample_dict,
            'chem_results': chem_results,
            'vina': vina_results,
            'mols': mol
        })
    logger.info(f'Evaluate No {id} done! {num_samples} samples in total. {n_eval_success} eval success!')
    if args.result_path:
        torch.save(results, os.path.join(args.result_path, f'eval_{id:03d}_{os.path.basename(ligand_filename[:-4])}.pt'))
    return results, all_pair_dist, all_bond_dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('meta_file', type=str)  # 'baselines/results/pocket2mol_pre_dock.pt'
    parser.add_argument('-n', '--eval_num_examples', type=int, default=100)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--protein_root', type=str, default='./data/train_set')
    parser.add_argument('--docking_mode', type=str, default='vina_full',
                        choices=['none', 'qvina', 'vina', 'vina_full', 'vina_score'])
    parser.add_argument('--exhaustiveness', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--aggregate_meta', type=eval, default=False)
    parser.add_argument('--protein_path', type=str)
    parser.add_argument('--ligand_path', type=str)
    parser.add_argument('--weights', type=str)
    args = parser.parse_args()

    # result_path = os.path.join(os.path.dirname(args.meta_file), f'eval_results_docking_{args.docking_mode}')
    if args.result_path:
        os.makedirs(args.result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', args.result_path)
    logger.info(args)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # tmp get ligand filename
    # with open('data/crossdocked_v1.1_rmsd1.0_processed/test_index.pkl', 'rb') as f:
    #     test_index = pickle.load(f)

    if args.aggregate_meta:
        meta_file_list = sorted(glob(os.path.join(args.meta_file, '*/result.pt')))
        print(f'There are {len(meta_file_list)} files to aggregate')
        test_index = []
        for f in tqdm(meta_file_list, desc='Load meta files'):
            test_index.append(torch.load(f))
    else:
        test_index = torch.load(args.meta_file)
        if isinstance(test_index[0], dict):  # single datapoint sampling result
            test_index = [test_index]

    testset_results = []
    testset_pair_dist, testset_bond_dist = [], []
    with Pool(args.num_workers) as p:
        # testset_results = p.starmap(partial(eval_single_datapoint, args=args),
        #                             zip(test_index[:args.eval_num_examples], list(range(args.eval_num_examples))))
        for (r, pd, bd) in tqdm(p.starmap(partial(eval_single_datapoint, args=args),
                                zip(test_index[:args.eval_num_examples], list(range(args.eval_num_examples)))),
                      total=args.eval_num_examples, desc='Overall Eval'):
            testset_results.append(r)
            testset_pair_dist += pd
            testset_bond_dist += bd
        # print('----------------', len([x['chem_results']['ring_size'] for r in testset_results for x in r]))

    # model_name = os.path.basename(args.meta_file).split('_')[0]
    if args.result_path:
        torch.save(testset_results, os.path.join(args.result_path, f'eval_all.pt'))

    qed = [x['chem_results']['qed'] for r in testset_results for x in r]
    sa = [x['chem_results']['sa'] for r in testset_results for x in r]
    vina = [x['vina'][0]['affinity'] for r in testset_results for x in r]
    vina_poses = [x['vina'][0]['pose'] for r in testset_results for x in r]
    vina_scaled = min_max_scaler(vina)
    mols = [x['mols'] for r in testset_results for x in r]

    weights = args.weights.split(',')
    w = [int(item) for item in weights]
    reward = [w[0] * q + w[1] * s - w[2] * v for q, s, v in zip(qed, sa, vina_scaled)]
    max_reward_index = reward.index(max(reward))
    best_mol = mols[max_reward_index]
    best_mol_pose = vina_poses[max_reward_index]
    # print(qed)
    # print(sa)
    # print(vina)
    # print(reward)

    os.makedirs('./new_data_temp', exist_ok=True)
    folder_name = args.protein_path.split('/')[0]
    protein_name = args.protein_path.split('/')[1].split('.')[0]
    os.makedirs('./new_data_temp/'+folder_name)
    shutil.copy(args.protein_root+'/'+args.protein_path, './new_data_temp/'+args.protein_path)
    shutil.copy(args.protein_root+'/'+args.ligand_path, './new_data_temp/'+args.ligand_path)

    not_posed = './new_data_temp/'+folder_name+'/'+protein_name+'_generated.sdf'
    mol_to_sdf(best_mol, not_posed)

    with open("docked_pose.pdb", "w") as pdb_file:
        pdb_file.write(best_mol_pose)

    posed_sdf = './new_data_temp/'+folder_name+'/'+protein_name+'_generated_docked.sdf'
    # Convert the PDB file to SDF using Openbabel
    get_pose_sdf(pdb_file = "docked_pose.pdb", sdf_file = posed_sdf)
    aligned_sdf = './new_data_temp/'+folder_name+'/'+protein_name+'_generated_aligned.sdf'
    transform_to_original(not_posed, posed_sdf, aligned_sdf)