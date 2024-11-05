import torch
import pickle

# Load the .pt file
pt_file_path = './data/split_by_name.pt'
loaded_data = torch.load(pt_file_path, map_location=torch.device('cpu'))

# Function to recursively print the structure of the loaded data
def print_structure(obj, level=0):
    indent = '  ' * level
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{indent}{key}:")
            print_structure(value, level + 1)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            print(f"{indent}[{i}]:")
            print_structure(item, level + 1)
            if i == 1:
                break
    elif isinstance(obj, torch.Tensor):
        print(f"{indent}Tensor shape: {obj.shape}, dtype: {obj.dtype}")
    else:
        print(f"{indent}{type(obj).__name__}: {obj}")

new_lst = []
for item in loaded_data['train']:
    new_item = {}
    folder = item[0].split('/')[0]
    pdb_lst = item[0].split('/')[1].split('_')[:3]
    pdb = pdb_lst[0] + '_' + pdb_lst[1] + '_' + pdb_lst[2] + '.pdb'
    new_item = {
        'src_protein_filename': folder + '/' + pdb,
        'src_ligand_filename': item[1],
        'data': {
            'protein_file': './data/crossdocked_v1.1_rmsd1.0_processed/' + item[0],
            'ligand_file': './data/crossdocked_v1.1_rmsd1.0_processed/' + item[1],
        },
    }
    new_lst.append(new_item)

with open("./data/train_index.pkl", "wb") as f:
    pickle.dump(new_lst, f)

with open("./data/train_index.pkl", "rb") as f:
    loaded_list = pickle.load(f)

print(loaded_list[0])
print(len(loaded_list))