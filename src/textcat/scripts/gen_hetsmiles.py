"""
Script for generating adsorption SMILES from OC20 .lmdb datasets.
"""
from textcat.utils import lmdb_to_atoms, gen_smiles_parallel

import argparse

from fairchem.core.datasets import LmdbDataset 
import pandas as pd
from pandarallel import pandarallel
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# from tqdm import tqdm  # for testing


d = 'Process OC20 .lmdb databases to generate correponding SMILES.'

def main():
    parser = argparse.ArgumentParser(description=d)
    parser.add_argument('--lmdb_path', type=str, help='Path to .lmdb database. If 0, all OC20 datasets/splits are processed sequentially.')
    parser.add_argument('--output', type=str, help='Output file name (w/o .txt).')
    parser.add_argument('--cores', type=int, help='number of CPU cores to use.')
    parser.add_argument('--order', type=int, default=1, help='Include order-hop surface neighbour atoms wrt adsorbate in SMILES.')
    parser.add_argument('--aa_bonds', action='store_true', default=False, help='Show connections between adsorbate atoms explicitly in SMILES.')
    parser.add_argument('--as_bonds', action='store_true', default=False, help='Show connections between adsorbate and surface atoms explicitly in SMILES.')
    parser.add_argument('--ss_bonds', action='store_true', default=False, help='Show connections between surface atoms explicitly in SMILES.')
    parser.add_argument('--include_bond_lengths', action='store_true', default=False, help='Replace bonds in SMILES with bond lentghs.')

    args = parser.parse_args()
    print(args)
    
    lmdb = LmdbDataset({'src': args.lmdb_path})
    inputs = [(lmdb_to_atoms(lmdb[i], False), 
            lmdb_to_atoms(lmdb[i], True),
            lmdb[i].tags, 
            i, 
            args.order, 
            args.aa_bonds,
            args.as_bonds,
            args.ss_bonds,
            args.include_bond_lengths) for i in range(len(lmdb))]
    print("Input collected before multiprocessing ...")

    pandarallel.initialize(progress_bar=True, nb_workers=args.cores)
    df = pd.DataFrame()
    df['input'] = inputs
    y = df['input'].parallel_apply(gen_smiles_parallel)

    with open(f'{args.output}.txt', 'w') as f:
        for o in y:
            f.write(f'{o[0]} {o[1][1]} {o[1][0]} {o[2][1]} {o[2][0]}\n')

if __name__ == '__main__':
    main()
