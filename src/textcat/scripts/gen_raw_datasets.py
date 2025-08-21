"""
Generate raw OC20 datasets containing 
metadata and different SMILES representations 
starting from public .lmdb datasets. 
Requires three files:

1) .lmdb files provided by OCP
2) 
"""

from textcat.utils import lmdb_to_atoms, gen_smiles_parallel

import argparse
from tqdm import tqdm 

import pandas as pd
from fairchem.core.datasets import LmdbDataset
from pandarallel import pandarallel
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def get_energy_adsorbate(c=0, h=0, n=0, o=0):
    """
    Same ref. energies from OC20 dataset.
    """
    Ec, Eh, En, Eo = -7.282, -3.477, -8.083, -7.204
    return c * Ec + h * Eh + n * En + o * Eo

def main():
    """
    Generate raw OC20 datasets containing 
    metadata and different SMILES representations 
    starting from public .lmdb datasets. 
    """
    parser = argparse.ArgumentParser(description="Run inference pipeline for OC20 data.")
    parser.add_argument('--lmdb_path', type=str, help="Directory containing model checkpoint, vocab, and scaler.")
    parser.add_argument('--gen_smiles', type=str, nargs='+', help='Generate adsorption SMILES. o1=order1, o2=order2. \
                        For each selection, both initial and final state are generated.')
    parser.add_argument('--num_cores', type=int, help='Number of CPU cores to use for generating SMILES.')

    args = parser.parse_args()

    LMDB = args.lmdb_path
    DF = args.lmdb_path.replace('.lmdb', '.parquet')
    OC20_MAP = None
    ADSORBATE_DICT = None

    
    db = LmdbDataset({'src': LMDB})
    print(f'Loaded {LMDB}')

    # adsorbate features
    ads_symbolss, ads_ids = [], []
    nC, nH, nO, nN = [], [], [], []
    ads_sizes = []
    ads_smiless, ads_inchis, ads_inchikeys = [], [], []
    ads_ase_formulas = []
    ads_energies_eV = []
    # material features
    bulk_symbolss, bulk_mpids, bulk_ids = [], [], []
    # surface features
    hs, ks, ls, hkls = [], [], [], []
    # data features
    sids, anomalies, classes, eads_eV = [], [], [], []
    scaled_eads_eV = []

    for data in tqdm(db):
        key = f'random{data.sid}'
        info = OC20_MAP[key]
        sids.append(data.sid)
        # Adsorbate atom count
        C, H, OX, N = 0, 0, 0, 0 
        for i in range(data.num_nodes):
            if data.tags[i] == 2:
                Z = data.atomic_numbers[i]
                if Z == 6:  # C
                    C += 1
                elif Z == 1:  # H
                    H += 1
                elif Z == 7:  # N
                    N += 1
                elif Z == 8:  # O
                    OX += 1
                else:
                    raise Exception("Something is wronggg.")
        nC.append(C)
        nH.append(H)
        nO.append(OX)
        nN.append(N)
        ads_sizes.append(C + H + OX + N)
        anomalies.append(info['anomaly'])
        classes.append(info['class'])
        h, k, ll = info['miller_index']
        hs.append(h)
        ks.append(k)
        ls.append(ll)
        hkls.append(str(h) + str(k) + str(ll))
        bulk_symbolss.append(info['bulk_symbols'])
        bulk_mpids.append(info['bulk_mpid'])
        bulk_ids.append(info['bulk_id'])
        ads_energies_eV.append(get_energy_adsorbate(C, H, N, OX))
        ads_symbolss.append(info['ads_symbols'])
        ads_ids.append(info['ads_id'])
        if 'test' in LMDB:  # Target unavailable
            eads_eV.append('N/A')
            scaled_eads_eV.append('N/A')
        else:
            eads_eV.append(data.y_relaxed)
            scaled_eads_eV.append(eads_eV[-1] + ads_energies_eV[-1])
        ads_smiless.append(ADSORBATE_DICT[info['ads_id']]["smiles"])
        ads_inchis.append(ADSORBATE_DICT[info['ads_id']]["inchi"])
        ads_inchikeys.append(ADSORBATE_DICT[info['ads_id']]["inchikey"])
        ads_ase_formulas.append(ADSORBATE_DICT[info['ads_id']]["ase_formula"])

    df = pd.DataFrame({'sid': sids,
                  'anomaly': anomalies, 
                  'class': classes, 
                  'C': nC, 
                  'H': nH, 
                  'O': nO, 
                  'N': nN, 
                  'ads_size': ads_sizes, 
                  'ads_symbols': ads_symbolss, 
                  'ads_id': ads_ids, 
                  'ads_smiles': ads_smiless, 
                  'ads_inchi': ads_inchis, 
                  'ads_inchikeys': ads_inchikeys,
                  'ads_ase_formula': ads_ase_formulas,
                  'ads_energy_eV': ads_energies_eV, 
                  'bulk_symbols': bulk_symbolss, 
                  'bulk_mpid': bulk_mpids, 
                  'bulk_id': bulk_ids, 
                  'h': hs, 
                  'k': ks, 
                  'l': ls, 
                  'hkl': hkls, 
                  'eads_eV': eads_eV, 
                  'scaled_eads_eV': scaled_eads_eV})
    
    if args.gen_smiles:
        for option in args.gen_smiles:   
            order = int(option[-1])        
            inputs = [(lmdb_to_atoms(db[i], False), 
            lmdb_to_atoms(db[i], True),
            db[i].tags, 
            i, 
            order, 
            args.aa_bonds,
            args.as_bonds,
            args.ss_bonds,
            False) for i in range(len(db))]
            pandarallel.initialize(progress_bar=True, nb_workers=args.cores)
            dfx = pd.DataFrame()
            dfx['input'] = inputs
            y = dfx['input'].parallel_apply(gen_smiles_parallel)
            df[f'smiles_is_o{order}'] = y[1]
            df[f'fr_is_o{order}'] = y[2]
            df[f'smiles_fs_o{order}'] = y[3]
            df[f'fr_fs_o{order}'] = y[4]
    
    df.to_parquet(DF)
    print(f'Saved {DF}')    


if __name__ == '__main__':
    main()