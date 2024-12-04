# BoKDiff: Best-of-K diffusion Alignment for Enhancing 3D Molecule Generation


## Dependencies
### Install via Conda and Pip
```bash
conda create -n decompdiff python=3.8
conda activate decompdiff
# conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
pip3 install torch
# conda install pyg -c pyg
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
conda install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge

# For decomposition
conda install -c conda-forge mdtraj
pip install alphaspace2

# For Vina Docking
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

# Important Note*
In vina package, modify vina.py and change np.int to np.int32
In alphaspace2 package, modify Snapshot.py and change np.float to np.float64 (line 145)
In alphaspace2 package, modify functions.py and change np.bool to np.bool_ (line 449)
```
# BoKdiff Scripts
## Data indexing
```bash
python utils/training_sample_indexer.py
```
You can download the dataset ("crossdocked_v1.1_rmsd1.0.tar.gz") [here](https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK).

You also need to download data files from [here](https://drive.google.com/drive/folders/1z74dKcDKQbwpo8Uf8EJpGi12T4GCD8_Z), and put them in the ./data folder for evaluation and running the following steps.

## Data Collection, Ranking, and Fine Tuning
```bash
python scripts/alignment.py
```
This script will be using these modified files:
- scripts/sample_modified.py
- scripts/evaluate_mol_modified.py
- scripts/preprocess_new_data.py
- scripts/train_aligned_decomp.py

For any desired adjustment, change the following parameters:
### Weight configurations:
``` python
evaluation_args = ["--docking_mode", "vina", "--aggregate_meta", "True", "--result_path", "./eval_temp",
"--protein_path", protein_src_file, "--ligand_path", ligand_src_file, "--weights", "1,0,0"] #qed,sa,vina
```
### Alignment iteration:
Add the trained model to the 'pretrained_models' folder and modify:
```python
alignment_iter = 1
random.seed(alignment_iter + 94) #94

if alignment_iter == 1:
    ckpt_path = 'pretrained_models/uni_o2_bond.pt'
else:
    ckpt_name = 'al_qed_iter' + str(alignment_iter-1) + '.pt'
    ckpt_path = 'pretrained_models/' + ckpt_name
```
### Batch Size:
``` python
# Data collection
batch_size = 128
```
## Model evaluation
The following script will generate samples based on the test set and evaluate them. You can reduce the 'batch size' from 100 to any smaller number.
```bash
python scripts/final_eval.py
```
## BoKdiff version 2 - Data relocating based on the Reference Ligand center of Mass
```bash
python scripts/alignment_v2.py
```
This script will be using these modified files:
- scripts/sample_modified.py
- scripts/evaluate_mol_modified_v2.py
- scripts/preprocess_new_data_v2.py
- scripts/train_aligned_decomp.py


# Decompdiff Scripts
## Preprocess 
```bash
python scripts/data/preparation/preprocess_subcomplex.py configs/preprocessing/crossdocked.yml
```
We have provided the processed dataset file [here](https://drive.google.com/drive/folders/1z74dKcDKQbwpo8Uf8EJpGi12T4GCD8_Z?usp=share_link).

## Training
To train the model from scratch, you need to download the *.lmdb, *_name2id.pt and split_by_name.pt files and put them in the _data_ directory. Then, you can run the following command:
```bash
python scripts/train_diffusion_decomp.py configs/training.yml
```

## Sampling
To sample molecules given protein pockets in the test set, you need to download test_index.pkl and test_set.zip files, unzip it and put them in the _data_ directory. Then, you can run the following command:
```bash
python scripts/sample_diffusion_decomp.py configs/sampling_drift.yml  \
  --outdir $SAMPLE_OUT_DIR -i $DATA_ID --prior_mode {ref_prior, beta_prior}
```
We have provided the trained model checkpoint [here](https://drive.google.com/drive/folders/1JAB5pp25rEM5Wt-i373_rrAyTsLvAACZ?usp=share_link).

If you want to sample molecules with beta priors, you also need to download files in this [directory](https://drive.google.com/drive/folders/1QOQOuDxdKkipYygZU9OIQUXqV9C28J5O?usp=share_link).

## Evaluation
```bash
python scripts/evaluate_mol_from_meta_full.py $SAMPLE_OUT_DIR \
  --docking_mode {vina, vina_full} \
  --aggregate_meta True --result_path $EVAL_OUT_DIR
```

### Alphaspace2 modifications
Snapshot.py file change two functions:
```python
def calculateContact(self, coords, cutoff):
  """
  Mark alpha/beta/pocket atoms as contact with in cutoff of ref points.

  _Beta atom and pocket atoms are counted as contact if any of their child alpha atoms is in contact.

  Parameters
  ----------
  coords: np.array shape = (n,3)

  Returns
  -------
  """

  self._alpha_contact, min_dist = _markInRange2(self._alpha_xyz, ref_points=coords, cutoff=cutoff)
  self._beta_contact = np.array(
      [np.any(self._alpha_contact[alpha_indices]) for alpha_indices in self._beta_alpha_index_list])

  self._pocket_contact = np.array(
      [np.any(self._alpha_contact[alpha_indices]) for alpha_indices in self._pocket_alpha_index_list])
  return min_dist

def run(self, receptor, binder=None, cutoff=1.6):

  self.genAlphas(receptor)
  
  self.genPockets()

  self.genBetas()
  
  self.genBScore(receptor)
  
  if binder is not None:
      min_dist = self.calculateContact(coords=binder.xyz[0] * 10, cutoff=cutoff)
      return min_dist
```
functions.py file add two new def modules:
```python
def _findInRange2(query_points, ref_points, cutoff):
    min_dist = np.min(cdist(query_points, ref_points))
    indices = np.where(cdist(query_points, ref_points) <= cutoff)[0]
    indices = np.unique(indices)
    return indices, min_dist

def _markInRange2(query_points, ref_points, cutoff):
    indices, min_dist = _findInRange2(query_points, ref_points, cutoff)
    query_bool = np.zeros(len(query_points), dtype=np.bool_)
    query_bool[indices] = 1
    return query_bool, min_dist
```
<!-- ### Continue running when ssh closes
nohup python ./scripts/alignment.py &
nohup python ./scripts/final_eval.py & -->

