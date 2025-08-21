"""
Utility functions to convert xyz to ASE to SMILES to RDKit.
"""

import ast
from itertools import product
import re

from ase import Atoms
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from rdkit import Chem
from scipy.spatial import Voronoi
import torch

from textcat.constants import RADII, RGB_COLORS
from textcat.format import atoms_to_mdl_str


def lmdb_to_atoms(data, relaxed: bool):
    """Convert lmdb entry from OC20 .lmdb datasets
    to ASE Atoms objects"""
    positions = data.pos_relaxed if relaxed else data.pos
    elems = data.atomic_numbers
    cell = data.cell
    return Atoms(symbols=elems, positions=positions, cell=cell.squeeze(), pbc=[1,1,1])

def get_adsorption_ensemble(atoms: Atoms,
                            tags: torch.Tensor, 
                            order: int) -> tuple[set[int], np.ndarray, dict[tuple[int], float]]:
    """
    Extract adsorption ensemble consisting of 
    (i) adsorbate, (ii) surface atoms directly in 
    contact with the adsorbate (order=1), and (iii) order-hop
    slab atoms wrt the adsorbate (order>1).

    Args:
        atoms(ase.Atoms): Atoms object of the adsorption structure.
        tags(torch.Tensor): Atom tags to distinguish adsorbate (2), and surface (1 or 0) atoms.
        order (int): include in the SMILES surface atoms that are order-hop neighbours wrt adsorbate.

    Return:
        ensemble(list[int]): List of the atom indices defining the adsorption ensemble. List needed for smiles.
        nl(np.array): neighbour list.
        distances(dict): distances between neighbours in Angstrom.    
    """
    nl, distances = get_neighbourlist(atoms, tags)
    ensemble = {atom.index for atom in atoms if tags[atom.index].item() == 2}

    for _ in range(order):
        surface_idxs = {pair[1] if pair[0] in ensemble else pair[0] \
            for pair in nl \
            if (pair[0] in ensemble and pair[1] not in ensemble) \
            or (pair[1] in ensemble and pair[0] not in ensemble)}
        ensemble = ensemble.union(surface_idxs)
    
    return list(ensemble), nl, distances


def adsorption2graph(atoms: Atoms,
                     tags: torch.Tensor, 
                     order: int = 1,
                     fmt: str = 'nx'): 
    """
    Convert adsorption structure to graph representation.
    """

    if fmt not in ('nx', 'pyg'):
        raise Exception("Available options are NetworkX (nx) and PyG (pyg)")
    
    ensemble, nl, _ = get_adsorption_ensemble(atoms, tags, order)
    g = nx.Graph()
    g.add_nodes_from(ensemble)
    nx.set_node_attributes(g, {i: atoms[i].symbol for i in g.nodes()}, "elem")
    nx.set_node_attributes(g, {i: RGB_COLORS[atoms[i].symbol] for i in g.nodes()}, "rgb")
    nx.set_node_attributes(g, {i: atoms[i].z for i in g.nodes()}, "Z")
    ensemble_neighbour_list = [pair for pair in nl if pair[0] in g.nodes() and pair[1] in g.nodes()]
    g.add_edges_from(ensemble_neighbour_list)    

    if fmt == 'pyg':
        from torch_geometric.utils import from_networkx
        return from_networkx(g, ["Z"])
    return g
    

def extract_adsorbate_ase(ase: Atoms, tags: torch.Tensor) -> Atoms:
    """
    Valid for OC20 data.
    """
    return Atoms([ase[i] for i in range(len(ase)) if tags[i].item() == 2], 
                 pbc=np.array([1, 1, 1]), 
                 cell=ase.cell)

def get_neighbourlist(atoms: Atoms,
                      tags: torch.Tensor,
                      tol: float = 0.25,
                      scaling_factor: float = 1.5,
                      mic: bool = True, 
                      radii : dict = RADII) -> tuple[np.ndarray, dict[tuple, float]]:
    """
    Get connectivity list from Voronoi analysis, considering periodic boundary conditions.

    Args:
        atoms (Atoms): ASE Atoms object representing the adsorbate-metal system.
        tags(torch.Tensor): atom labels (2: adsorbate, 0,1: material)
        tol (float): tolerance for the distance between two atoms to be considered connected.
        scaling_factor (float): scaling factor for the covalent radii of the metal atoms.
        radii (dict): Dictionary containing atomic radii for all elements. Default to Cordero radii.
        get_distances(bool): if True return distance between each nieghbour pair (in Angstrom).
    Returns:
        np.ndarray: connectivity list of the system (N_edges x 2). Each row represents a pair of connected atoms.
    Notes:
        - Each connection is represented once, i.e. if atom A is connected to atom B, the pair (A, B) will be present in the list,
            but not the pair (B, A).
        - To help in the detection of surface-adsorbate connections, the scaling factor is iteratively
            increased by 0.2 until at least one connection between the adsorbate and the surface is found.
    """

    if len(atoms) < 2:
        return np.array([]), {}

    # First necessary condition for two atoms to be linked: Sharing a Voronoi facet

    coords_arr = np.repeat(np.expand_dims(
        np.copy(atoms.get_scaled_positions()), axis=0), 27, axis=0)
    mirrors = np.repeat(np.expand_dims(np.asarray(
        list(product([-1, 0, 1], repeat=3))), 1), coords_arr.shape[1], axis=1)
    corrected_coords = np.reshape(
        coords_arr + mirrors, (coords_arr.shape[0] * coords_arr.shape[1], coords_arr.shape[2]))
    corrected_coords = np.dot(corrected_coords, atoms.get_cell())
    translator = np.tile(np.arange(coords_arr.shape[1]), coords_arr.shape[0])
    vor_bonds = Voronoi(corrected_coords)
    pairs_corr = translator[vor_bonds.ridge_points]
    pairs_corr = np.unique(np.sort(pairs_corr, axis=1), axis=0)
    pairs_corr = np.delete(pairs_corr, np.argwhere(
        pairs_corr[:, 0] == pairs_corr[:, 1]), axis=0)

    # Second necessary condition for two atoms to be linked: their distance
    # must be smaller than the sum of their covalent radii + tol
    # We enforce surface-adsorbate (S-A) connections by 
    # 1) Increasing the radius of the surface atom
    # 2) Iteratively losing the threshold until at least one S-A is detected

    increment = 0.0
    distances = {}

    while True:
        pairs = []
        for pair in pairs_corr:
            id1, id2 = pair[0], pair[1]
            atom1, atom2 = atoms[id1].symbol, atoms[id2].symbol
            threshold = radii[atom1] + radii[atom2] + tol
            if tags[id1].item() == 2 and tags[id2].item() != 2:
                threshold += max(scaling_factor + increment -
                                 1.0, 0) * radii[atom2]
            if tags[id1].item() != 2 and tags[id2].item() == 2:
                threshold += max(scaling_factor + increment -
                                 1.0, 0) * radii[atom1]

            distance = atoms.get_distance(id1, id2, mic=mic)

            if distance <= threshold:
                pairs.append(pair)
                distances[tuple(pair)] = distance

        c1 = any(
            tags[pair[0]].item() == 2
            and tags[pair[1]].item() != 2
            for pair in pairs
        )
        c2 = any(
            tags[pair[0]].item() != 2
            and tags[pair[1]].item() == 2
            for pair in pairs
        )

        if c1 or c2:
            break
        else:
            increment += 0.2  # hard-coded

    return np.sort(np.array(pairs), axis=1), distances


def plot_graph(g: nx.Graph,
                  node_size: int=320,
                  font_color: str="white",
                  font_weight: str="bold",
                  alpha: float=1.0, 
                  arrowsize: int=10,
                  width: float=1.2,
                  node_index: bool=False, 
                  text: str=None,
                  ax = None):
    """
    Visualize graph with atom labels and colors. 
    Kamada_kawai_layout engine gives the best visualization appearance.
    Args:
        graph(torch_geometric.data.Data): graph object in pyG format.
    """
    node_labels = nx.get_node_attributes(g, 'elem')
    node_colors = list(nx.get_node_attributes(g, 'rgb').values()) 
    nx.draw_networkx(g, 
                  labels=node_labels, 
                  node_size=node_size,
                  font_color=font_color, 
                  font_weight=font_weight,
                  node_color=node_colors, 
                  edge_color='black',
                  alpha=alpha, 
                  arrowsize=arrowsize, 
                  width=width,
                  pos=nx.kamada_kawai_layout(g), 
                  linewidths=0.5, 
                  ax = ax)
    if node_index:
        pos_dict = nx.kamada_kawai_layout(g)
        for node in g.nodes:
            x, y = pos_dict[node]
            plt.text(x+0.05, y+0.05, node, fontsize=7)        
    if text is not None:
        plt.text(0.03, 0.9, text, fontsize=10)
    plt.axis('off')


def adsorption2smiles(atoms: Atoms,
                    tags: torch.Tensor,  
                    order: int = 1, 
                    aa_bonds_explicit: bool = False,
                    as_bonds_explicit: bool = False,
                    ss_bonds_explicit: bool = False,
                    include_bond_lengths: bool = False,
                    verbose : bool = False):
    """
    Convert adsorption structure to SMILES representation.

    Args:
        atoms(ase.Atoms): Atoms object of the adsorption structure.
        tags(torch.Tensor): Atom tags to distinguish adsorbate (2), and surface (1 or 0) atoms.
        order (int): include in the SMILES surface atoms that are order-hop neighbours wrt adsorbate.
        aa_bonds_explicit (bool): Include explicit bonds between adsorbate atoms.
        as_bonds_explicit (bool): Include explicit bonds between adsorbate and surface atoms.
        ss_bonds_explicit (bool): Include explicit bonds between surface atoms.
        include_bond_lengths (bool): If True, bond lengths in Angstrom will replace the
            explicit bonds in the resulting SMILES.

    Notes: 
    - Each atom symbol will be always returned embedded within square brackets to allow RDKit 
        to read HetSMILES (e.g., "[C][H]~[Pd]"). To "clean" the string, you can use ''smiles.replace("[", "")''
        and ''smiles.replace("]", "")''.
    """

    ensemble, nl, distances = get_adsorption_ensemble(atoms, tags, order)
    elems_tot = {atoms[i].symbol for i in range(len(atoms))}
    elems_ensemble = {atoms[i].symbol for i in ensemble}
    elems_diff =  elems_tot - elems_ensemble
    if elems_diff != set():
        fr = False
        if verbose:
            print(f"Warning: {elems_diff} in the material {atoms.get_chemical_formula()} not in the adsorption ensemble!")
    else:
        fr = True
    mask = np.logical_and(np.isin(nl[:, 0], ensemble), np.isin(nl[:, 1], ensemble))
    nl = nl[mask]
    bonds, bonds_order, bonds_distance = [], [], []
    for pair in nl:    
        if tuple(pair) in distances.keys():
            bonds_distance.append(distances[(pair[0], pair[1])])
        tags_pair = (tags[pair[0]].item(), tags[pair[1]].item())
        if sum(tags_pair) == 4:
            if aa_bonds_explicit:
                bonds_order.append(8) 
            else:
                bonds_order.append(1)
        elif sum(tags_pair) < 3 and 2 not in tags_pair: 
            if ss_bonds_explicit:
                bonds_order.append(8) 
            else:
                bonds_order.append(1)
        else:
            if as_bonds_explicit:
                bonds_order.append(8) 
            else:
                bonds_order.append(1)
        bonds.append((ensemble.index(pair[0]) + 1, ensemble.index(pair[1]) + 1))

    mdl_blob = atoms_to_mdl_str([atoms[idx].symbol for idx in ensemble], 
                                atoms.positions[ensemble, :], 
                                bonds, 
                                bonds_order)

    mol = Chem.MolFromMolBlock(
        mdl_blob,
        removeHs=False,
        strictParsing=False,
        sanitize=False,
    )

    # Required to avoid filling metal valence with implicit Hydrogens
    for a in mol.GetAtoms():
        if a.GetNumImplicitHs():
            a.SetNumRadicalElectrons(a.GetNumImplicitHs())
            a.SetNoImplicit(True)
            a.UpdatePropertyCache()

    [a.SetAtomMapNum(0) for i, a in enumerate(mol.GetAtoms())]  # Remove atom mapping
    smiles_with_tildes = Chem.MolToSmiles(mol, 
                        isomericSmiles=False, 
                        kekuleSmiles=False,
                        rootedAtAtom=-1, 
                        canonical=True, 
                        allBondsExplicit=False, 
                        allHsExplicit=True, # important
                        doRandom=False)
    if include_bond_lengths:
        bonds_types = {}
        for bond in mol.GetBonds():
            bonds_types[bond.GetIdx()] = bond.GetBondType()
        ls_bond_order = mol.GetProp("_smilesBondOutputOrder")
        ls_bond_order = ast.literal_eval(ls_bond_order)
        res = []
        for i in ls_bond_order:
            if bonds_types[i] == Chem.BondType.UNSPECIFIED:
                res.append(bonds_distance[i])

        # 3) Create HetSMILES with bond lengths
        new_smiles = ""
        counter = 0 
        for i, char in enumerate(smiles_with_tildes):
            if char == '~':
                new_smiles += "{" + "{:.4f}".format(res[counter]) + "}"
                counter += 1
            else:
                new_smiles += char      
        return new_smiles, fr
    else: 
        return smiles_with_tildes, fr


def numsmiles2smiles(x: str) -> str:
    """
    Given a SMILES string x with encoded bond-lenghts, convert them to 
    an RDKit-accessible SMILES. All explicit bonds will be set as
    "UNDEFINED" type.
    """
    return re.sub(r'\{.*?\}', '~', x)


def gen_smiles_parallel(x):
    """  
    For multiprocessing. Takes both initial and final state 
    and convert them to the corresponding SMILES.
    """
    ais, afs, tags, idx, order, aab, asb, ssb, include_bond_lengths = x
    return idx, adsorption2smiles(ais, tags, order, aab, asb, ssb, include_bond_lengths), \
            adsorption2smiles(afs, tags, order,  aab, asb, ssb, include_bond_lengths)
