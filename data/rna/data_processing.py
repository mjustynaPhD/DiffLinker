# %%
import os
import numpy as np
import torch
from tqdm import tqdm



from Bio.PDB import *
from Bio.SVDSuperimposer import SVDSuperimposer
from rnapolis.annotator import extract_secondary_structure
from rnapolis.parser import read_3d_structure
import sys
sys.path.append('../../')
from src.visualizer import save_xyz_file

pdbs_dir = "/data/3d/input_data/pdbs/"
out_name = 'rna_GC_val.pt'
ONE_HOT = {'P': 3, 'H': 4, 'C': 0, 'O': 2, 'N': 1}
save_vis = True

def main():
    # read 3d structure
    all_transformed = []
    for pdb in tqdm(os.listdir(pdbs_dir)[1000:1001]):
        pdb_path = os.path.join(pdbs_dir, pdb)
        # print(f"Structure extraction from pdb file {pdb}")
        with open(pdb_path) as f:
            structure3d = read_3d_structure(f, 1)
            structure2d = extract_secondary_structure(structure3d, 1)

        interest_res = []
        interest_bps = []
        for bp in structure2d.basePairs:
            bp_type = bp.saenger.is_canonical if bp.saenger is not None else False
            if bp_type and (bp.nt1.name == 'C' and bp.nt2.name == 'G'):
                        # or (bp.nt1.name == 'G' and bp.nt2.name == 'C'):
                # print(f"{bp.nt1.chain}.{bp.nt1.name}{bp.nt1.number}")
                # print(f"{bp.nt2.chain}.{bp.nt2.name}{bp.nt2.number}")
                bp1 = f"{bp.nt1.chain}.{bp.nt1.name}{bp.nt1.number}"
                bp2 = f"{bp.nt2.chain}.{bp.nt2.name}{bp.nt2.number}"
                interest_res.append(bp1)
                interest_res.append(bp2)
                interest_bps.append((bp1, bp2))
            # if not bp_type:
            #     # print(bp_type, bp.nt1, bp.nt2)
            #     # print(f"{bp.nt1.chain}.{bp.nt1.name}{bp.nt1.number}")
            #     # print(f"{bp.nt2.chain}.{bp.nt2.name}{bp.nt2.number}")
            #     bp1 = f"{bp.nt1.chain}.{bp.nt1.name}{bp.nt1.number}"
            #     bp2 = f"{bp.nt2.chain}.{bp.nt2.name}{bp.nt2.number}"
            #     interest_res.append(bp1)
            #     interest_res.append(bp2)
            #     interest_bps.append((bp1, bp2))


        residues = {}
        # Read the pdb file
        parser = PDBParser()
        structure = parser.get_structure('pdb', pdb_path)
        for residue in structure.get_residues():
            res = f"{residue.get_full_id()[2]}.{residue.get_resname()}{residue.get_id()[1]}"
            if res in interest_res:
                # print(res)
                residues[res] = residue
        
        
        for i, bps in enumerate(interest_bps):
            try:
                sel_res = [residues[bps[0]], residues[bps[1]]]
                transformed = pdb_transform(sel_res, name=f"{pdb.split('.')[0]}_{i}")
                all_transformed.append(transformed)
            except TypeError:
                print(f"TypeError: {pdb} {bps[0]} {bps[1]}")
                raise
            except KeyError:
                print(f"KeyError: {pdb} {bps[0]} {bps[1]}")
                raise
            
            if len(all_transformed) >2:
                try:
                    superimposed, rmsd = superimpose(all_transformed[0]['positions'], all_transformed[-1]['positions'])
                    if rmsd > 1.5:
                        print(all_transformed[0]['name'], all_transformed[-1]['name'], rmsd)
                        raise ValueError("RMSD of alignment is too high!")
                    all_transformed[-1]['positions'] = superimposed
                except ValueError as e:
                    print(e, " skipping...")
                    all_transformed.pop()
                except Exception as e:
                    print("Error in superimpose")
                    all_transformed.pop()

    # save xyz samples
    if save_vis:
        for i, sample in enumerate(all_transformed):
                save_xyz_file(
                    f"xyz/",
                    sample["one_hot"].unsqueeze(0),
                    sample["positions"].unsqueeze(0),
                    node_mask=torch.ones(len(sample["one_hot"])).unsqueeze(0),
                    names = [f'{sample["name"]}'],
                    is_geom=None
                )

    # save transformed to *.pt file
    torch.save(all_transformed, out_name)
    print(f"Saved {out_name}")

# %%
def save_as_new_pdb(residues, output_file):
    structure = Structure.Structure('pdb')
    model = Model.Model(0)
    chain = Chain.Chain('A')
    model.add(chain)
    structure.add(model)

    # add the residues to the structure
    for residue in residues:
        chain.add(residue)

    # write the new pdb file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)




def pdb_transform(residues, name):
    transform_dict = {}
    transform_dict["uuid"] = name
    transform_dict["name"] = name
    transform_dict["positions"] = torch.Tensor(get_positions(residues))
    transform_dict["one_hot"] = torch.Tensor(get_one_hot(residues))
    transform_dict["fragment_mask"] = torch.Tensor(get_fragment_mask(residues))
    transform_dict["linker_mask"] = torch.Tensor(get_linker_mask(residues))
    transform_dict["num_atoms"] = transform_dict["one_hot"].shape[0]
    transform_dict['anchors'] = torch.Tensor(get_anchors(residues))
    transform_dict['charges'] = torch.Tensor(np.zeros(transform_dict["num_atoms"]))
    # atoms = get_atoms(residues)
    return transform_dict

def superimpose(ref_coords, model_coords):
    sup = SVDSuperimposer()
    sup.set(np.array(ref_coords, 'f'), np.array(model_coords, 'f'))
    sup.run()
    print('RMSD: {:.2f}'.format(sup.get_rms()))
    return torch.Tensor(sup.get_transformed()), sup.get_rms()

def get_positions(residues):
    positions = []
    for residue in residues:
        for atom in residue.get_atoms():
            positions.append(atom.get_coord())

    # # normalize the positions to 0 mean and 1 std
    positions = np.array(positions)
    positions = (positions - positions.mean(axis=0))
    # normalize positions to 0-1
    # positions = np.array(positions)
    # positions = (positions - positions.min(axis=0)) / (positions.max(axis=0) - positions.min(axis=0))
    return np.array(positions)

def get_one_hot(residues):
    one_hot = []
    for residue in residues:
        atom_names = [atom.name[0] for atom in residue.get_atoms()]
        # make one-hot encoding array using scikit learn
        one_hot_res = _make_one_hot(atom_names)
        one_hot.extend(one_hot_res)
    return np.array(one_hot)

def _make_one_hot(atom_names):
    one_hot = np.zeros((len(atom_names), len(ONE_HOT)))
    for i, atom_name in enumerate(atom_names):
        one_hot[i, ONE_HOT[atom_name]] = 1
    return one_hot

fragment_atoms = ['P', 'C5\'', 'O5\'', 'C4\'', 'O4\'',
                'C3\'', 'O3\'', 'C2\'', 'O2\'', 'C1\'']
def get_fragment_mask(residues):
    fragment_mask = []
    for residue in residues:
        atom_names = [atom.name for atom in residue.get_atoms()]
        fragment_mask.extend([1 if atom_name in fragment_atoms else 0 for atom_name in atom_names])
    return np.array(fragment_mask)

def get_linker_mask(residues):
    linker_mask = []
    for residue in residues:
        atom_names = [atom.name for atom in residue.get_atoms()]
        linker_mask.extend([1 if atom_name not in fragment_atoms else 0 for atom_name in atom_names])
    return np.array(linker_mask)

def get_atoms(residues):
    atom_names = []
    for residue in residues:
        atom_names.extend([atom.name for atom in residue.get_atoms()])
    return atom_names

def get_anchors(residues):
    anchors = []
    for residue in residues:
        atom_names = [atom.name for atom in residue.get_atoms()]
        anchors.extend([1 if atom_name == 'C1\'' else 0 for atom_name in atom_names])
    return np.array(anchors)

if __name__ == "__main__":
    main()
