import numpy as np
import networkx as nx
from pymatgen.core.periodic_table import ElementBase
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import (
    CrystalNN,
    BrunnerNN_reciprocal,
    EconNN,
    MinimumDistanceNN,
)


class SLICES:
    """Invertible Crystal Representation (SLICES and labeled quotient graph)"""

    def __init__(
        self,
        atom_types=None,
        edge_indices=None,
        to_jimages=None,
        graph_method="econnn",
        check_results=False,
        optimizer="BFGS",
        fmax=0.2,
        steps=100,
        relax_model="m3gnet",
    ):
        """__init__

        Args:
            atom_types (np.array, optional): Atom types in a SLICES string. Defaults to None.
            edge_indices (np.array, optional): Edge indices in a SLICES string. Defaults to None.
            to_jimages (np.array, optional): Edge labels in a SLICES string. Defaults to None.
            graph_method (str, optional): The method used for analyzing the local chemical environments
                to generate labeled quotient graphs. Defaults to 'econnn'.
            check_results (bool, optional): Flag to indicate whether to output intermediate results for
                debugging purposes. Defaults to False.
            optimizer (str, optional): Optimizer used in M3GNet_IAP optimization. Defaults to "BFGS".
            fmax (float, optional): Convergence criterion of maximum allowable force on each atom.
                Defaults to 0.2.
            steps (int, optional): Max steps. Defaults to 100.
        """
        self.atom_types = atom_types
        self.edge_indices = edge_indices
        self.to_jimages = to_jimages
        self.graph_method = graph_method
        self.check_results = check_results
        self.atom_symbols = None
        self.SLICES = None
        self.unstable_graph = False  # unstable graph flag
        self.fmax = fmax
        self.steps = steps
        self.relax_model = relax_model

        # copy m3gnet model file?
        # if self.relax_model=="chgnet":
        #     with self.suppress_output():
        #         self.relaxer = StructOptimizer(optimizer_class="BFGS",use_device="cpu")
        # if self.relax_model=="m3gnet":
        #     model_path=m3gnet.models.__path__[0]+'/MP-2021.2.8-EFS/'
        #     if not os.path.isdir(model_path):
        #         data_path=os.path.dirname(__file__)+'/MP-2021.2.8-EFS'
        #         subprocess.call(['mkdir','-p', model_path])
        #         subprocess.call(['cp',data_path+'/checkpoint',data_path+'/m3gnet.data-00000-of-00001',\
        #         data_path+'/m3gnet.index',data_path+'/m3gnet.json',model_path])
        #     self.relaxer = Relaxer(optimizer=optimizer)

    def structure2SLICES(self, structure, strategy=4):
        """Extract edge_indices, to_jimages and atom_types from a pymatgen structure object
         then encode them into a SLICES string.

        Args:
            structure (Structure): A pymatgen Structure.
            strategy (int, optional): Strategy number. Defaults to 3.

        Returns:
            str: A SLICES string.
        """
        structure_graph = self.structure2structure_graph(structure)
        atom_types = np.array(structure.atomic_numbers)
        atom_symbols = [str(ElementBase.from_Z(i)) for i in atom_types]
        G = nx.MultiGraph()
        G.add_nodes_from(structure_graph.graph.nodes)
        G.add_edges_from(
            structure_graph.graph.edges
        )  # convert to MultiGraph (from MultiDiGraph) !MST can only deal with MultiGraph
        edge_indices, to_jimages = [], []
        for i, j, to_jimage in structure_graph.graph.edges(data="to_jimage"):
            edge_indices.append([i, j])
            to_jimages.append(to_jimage)
        return self.get_slices_by_strategy(
            strategy, atom_symbols, edge_indices, to_jimages
        )

    def get_slices_by_strategy(self, strategy, atom_symbols, edge_indices, to_jimages):
        strategy_method_map = {
            1: self.get_slices1,
            2: self.get_slices2,
            3: self.get_slices3,
            4: self.get_slices4,
        }
        method = strategy_method_map.get(strategy)
        if method:
            return method(atom_symbols, edge_indices, to_jimages)
        else:
            raise ValueError(f"Unknown strategy {strategy}")

    def structure2structure_graph(self, structure):
        """Convert a pymatgen structure to a structure_graph.

        Args:
            structure (Structure): A pymatgen Structure.

        Returns:
            StructureGraph: A Pymatgen StructureGraph object.
        """
        if self.graph_method == "brunnernn":
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, BrunnerNN_reciprocal()
            )
        elif self.graph_method == "econnn":
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, EconNN()
            )
        elif self.graph_method == "mininn":
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, MinimumDistanceNN()
            )
        elif self.graph_method == "crystalnn":
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, CrystalNN(porous_adjustment=False, weighted_cn=True)
            )
        else:
            print("ERROR - graph_method not implemented")
        return structure_graph

    @staticmethod
    def get_slices1(atom_symbols, edge_indices, to_jimages):
        SLICES = ""
        for i in range(len(edge_indices)):
            SLICES += (
                atom_symbols[edge_indices[i][0]]
                + " "
                + atom_symbols[edge_indices[i][1]]
                + " "
                + str(edge_indices[i][0])
                + " "
                + str(edge_indices[i][1])
                + " "
            )
            for j in to_jimages[i]:
                if j <= -1:
                    SLICES += "- "
                if j == 0:
                    SLICES += "o "
                if j >= 1:
                    SLICES += "+ "
        return SLICES

    @staticmethod
    def get_slices2(atom_symbols, edge_indices, to_jimages):
        atom_symbols_mod = [(i + "_")[:2] for i in atom_symbols]
        SLICES = ""
        for i in atom_symbols_mod:
            SLICES += i
        for i in range(len(edge_indices)):
            SLICES += ("0" + str(edge_indices[i][0]))[-2:] + (
                "0" + str(edge_indices[i][1])
            )[-2:]
            for j in to_jimages[i]:
                if j <= -1:
                    SLICES += "-"
                if j == 0:
                    SLICES += "o"
                if j >= 1:
                    SLICES += "+"
        return SLICES

    @staticmethod
    def get_slices3(atom_symbols, edge_indices, to_jimages):
        SLICES = ""
        for i in atom_symbols:
            SLICES += i + " "
        for i in range(len(edge_indices)):
            SLICES += str(edge_indices[i][0]) + " " + str(edge_indices[i][1]) + " "
            for j in to_jimages[i]:
                if j <= -1:
                    SLICES += "- "
                if j == 0:
                    SLICES += "o "
                if j >= 1:
                    SLICES += "+ "
        return SLICES

    @staticmethod
    def get_slices4(atom_symbols, edge_indices, to_jimages):
        SLICES = ""
        for i in atom_symbols:
            SLICES += i + " "
        for i in range(len(edge_indices)):
            SLICES += str(edge_indices[i][0]) + " " + str(edge_indices[i][1]) + " "
            for j in to_jimages[i]:
                if j <= -1:  # deal with -2, -3, etc (just in case)
                    SLICES += "-"
                if j == 0:
                    SLICES += "o"
                if j >= 1:  # deal with 2, 3, etc (just in case)
                    SLICES += "+"
            SLICES += " "
        return SLICES
