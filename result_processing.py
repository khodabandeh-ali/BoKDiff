import os
import itertools
import shutil
import torch
import pickle
import random
import numpy as np
import pandas as pd
random.seed(42) 

import sys
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.visualize import *
from utils import misc
from utils.evaluation import eval_bond_length
from utils.transforms import get_atomic_number_from_index

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

def results_logger(setting, weights):
    logger = misc.get_logger('evaluate', './')
    file_path = './final_evals_history/' + setting + '/eval_all.pt'
    # file_path = './final_evals/eval_all.pt'
    data = torch.load(file_path)


    c, t, total = 0, 0, 0
    q, s, vm, vs, vd = [], [], [], [], []
    n_atoms = []
    all_pair_dist, all_bond_dist = [], []
    ring_ratios = []
    # print(len(data))
    for r in data:
        num_atoms = [len(x['pred_pos']) for x in r]
        qed = [x['chem_results']['qed'] for x in r]
        sa = [x['chem_results']['sa'] for x in r]
        vina_dock = [x['vina']['dock'][0]['affinity'] for x in r]
        vina_score_only = [x['vina']['score_only'][0]['affinity'] for x in r]
        vina_min = [x['vina']['minimize'][0]['affinity'] for x in r]

        best_reward = -10000
        best_index = None
        
        # No success case
        # if best_index == None:
        #     for i in range(len(qed)):
        #         reward = qed[i] + sa[i] + (vina_dock[i]/-12)
        #         # reward = qed[i] + sa[i] + (vina_dock[i] + vina_score_only[i] + vina_min[i])/-12
        #         if reward > best_reward and vina_min[i] < -6:
        #             best_reward = (qed[i] + sa[i] + (vina_dock[i]/-12))
        #             best_index = i
        
        if best_index == None:
            for i in range(len(qed)):
                reward = weights[0] * qed[i] + weights[1] * sa[i] + weights[2] * (vina_dock[i]/-12)
                # reward = qed[i] + sa[i] + (vina_dock[i] + vina_score_only[i] + vina_min[i])/-12
                if reward > best_reward:
                    best_reward = (qed[i] + sa[i] + (vina_dock[i]/-12))
                    best_index = i

        n_atoms.append(num_atoms[best_index])
        q.append(qed[best_index])
        s.append(sa[best_index])
        vs.append(vina_score_only[best_index])
        vm.append(vina_min[best_index])
        vd.append(vina_dock[best_index])

        pred_pos, pred_v = r[best_index]['pred_pos'], r[best_index]['pred_v']
        pred_atom_type = get_atomic_number_from_index(pred_v, mode='add_aromatic')
        pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
        all_pair_dist += pair_dist

        bond_dist = eval_bond_length.bond_distance_from_mol(r[best_index]['mol'])
        all_bond_dist += bond_dist

        ring_ratios.append(r[best_index]['chem_results']['ring_size'])

    for i in range(len(q)):
        # total += 1
        if q[i] > 0.25 and s[i] > 0.59 and vd[i] < -8.18:
            c += 1
    total = len(q)
    success_rate = c / total * 100

    result_dict = {
        "Setting": setting,
        "Weights": weights,
        "Vina Score-Mean": np.mean(vs),
        "Vina Score-Med": np.median(vs),
        "Vina Min-Mean": np.mean(vm),
        "Vina Min-Med": np.median(vm),
        "Vina Dock-Mean": np.mean(vd),
        "Vina Dock-Med": np.median(vd),
        "QED-Mean": np.mean(q),
        "QED-Med": np.median(q),
        "SA-Mean": np.mean(s),
        "SA-Med": np.median(s),
        "SA-Med": np.median(s),
        "Success Rate": success_rate,
    }
    return result_dict
    # logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(q), np.median(q)))
    # logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(s), np.median(s)))
    # logger.info('Num atoms:   Mean: %.3f Median: %.3f' % (np.mean(num_atoms), np.median(num_atoms)))
    # logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vs), np.median(vs)))
    # logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vm), np.median(vm)))
    # logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vd), np.median(vd)))
    # logger.info('Success Rate :  %.2f' % (success_rate))
    # logger.info('Total :  %.2f' % (total))
    # pair_length_profile = eval_bond_length.get_pair_length_profile(all_pair_dist)
    # js_metrics = eval_bond_length.eval_pair_length_profile(pair_length_profile)
    # logger.info('JS pair distances: ')
    # print_dict(js_metrics, logger)

    # c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    # c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    # logger.info('JS bond distances: ')
    # print_dict(c_bond_length_dict, logger)

    # print_ring_ratio(ring_ratios, logger)


settings = ['ref_BoN_N10', 'ref_BoN_N15', 'ref_BoN_N20', 'ref_BoN_N25', 'ref_BoN_N30',
            'qed_BoN_N20', 'sa_BoN_N20', 'vina_BoN_N20']
# weights = [[1,1,1], [1,0,0], [0,1,0], [0,0,1], [1,1,2], [1,1,3], [1,1,5], [1,1,10]]
weights = [[1,1,1], [1.1,1,0.9]]
results_df = pd.DataFrame()
for w in weights:
    for s in settings:
        result = results_logger(s, w)
        results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
        print(s + ' & ' + str(w) + '  done!')

results_df = results_df.round(2)
results_df.to_csv("results_bests.csv", index=False)

## Grid Search
# # Define the range for weights
# weights_range = [0.8, 0.9, 1.0, 1.1, 1.2]
# # Generate all combinations of weights (for 3 weights)
# weight_combinations = list(itertools.product(weights_range, repeat=3))

# settings = ['ref_BoN_N30']
# results_df = pd.DataFrame()
# for w in weight_combinations:
#     for s in settings:
#         result = results_logger(s, w)
#         results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
#         print(s + ' & ' + str(w) + '  done!')

# results_df = results_df.round(2)
# results_df.to_csv("results_grid_search.csv", index=False)