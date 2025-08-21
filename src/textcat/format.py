"""
Format converters between ASE, RDKit Mol, xyz
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import io
import ase


def atoms_to_mdl_str(
    atoms, coordinates, bonds, bond_order=False, use_wildcard_token=False
):
    blob = "     RDKit          3D \n"
    blob += "     RDKit          3D \n"
    blob += "     RDKit          3D \n"
    blob += f"{coordinates.shape[0]:3}{len(bonds):3}  0  0  1  0  0  0  0  0999 V2000\n"

    for atom, coord in zip(atoms, coordinates):
        if use_wildcard_token and atom not in ["C", "H", "O", "N", "S"]:
            atom = "R"
        blob += f"{coord[0]:10.4f}{coord[1]:10.4f}{coord[2]:10.4f} {atom:<2s} 0  0  0  0  0  0  0  0  0  0  0  0\n"

    for bond, order in zip(bonds, bond_order):
        blob += f"{bond[0]:3}{bond[1]:3}  {order}  0  0  0  0\n"

    blob += "M  END"
    return blob


def smiles_to_xyz(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    xyz = []
    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        xyz.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")

    xyz_string = f"{mol.GetNumAtoms()}\n\n" + "\n".join(xyz)
    return xyz_string


def vasp_to_ase(vasp_data):
    with io.StringIO(vasp_data) as f:
        atoms = ase.io.read(f, format="vasp")
        atoms.center(vacuum=5)

    return atoms


def smiles_to_ase(smiles):
    xyz = smiles_to_xyz(smiles)
    atoms = xyz_to_ase(xyz)
    return atoms


def xyz_to_ase(xyz_data):
    with io.StringIO(xyz_data) as f:
        atoms = ase.io.read(f, format="xyz")
        atoms.center(vacuum=5)
    return atoms
