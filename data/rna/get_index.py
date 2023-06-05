# Read a pdb structure, copy residues and write a new pdb file
# with only the selected residues
# Usage: python get_index.py <pdb_file> <residues> <output_file>

import Bio
from Bio.PDB import *
import sys

# Read the pdb file
parser = PDBParser()
structure = parser.get_structure('pdb', sys.argv[1])

# copy residues
residues = []
for residue in structure.get_residues():
    residues.append(residue)

# select residues
residues = residues[:2]
# create a new structure
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
io.save("output.pdb")

# obabel -ipdb output.pdb -osdf -O output.sdf