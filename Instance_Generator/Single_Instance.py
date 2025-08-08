import igraph
import numpy as np
from .utils import get_gabriel_graph, draw_graph_partition, generate_partition


class Single_Instance_Generator:
    """ 
    Class to generate a graph instance for the p-regions problem.

    The created graph is an instance of an igraph graph.
    
    Node attributes:
        "x": list — a vector defining node attributes
        "name": str — the index of the node, as a string

    Graph attributes:
        "pos": dict — a dictionary indicating node spatial coordinates
        "P": dict — a partition created in the graph
        "status": str — indicates whether the created partition P is optimal for the p-regions problem
    """

    def __init__(self, num_points: int, num_regions: int, num_features: int,
                 beta_parameter: float = 1.0, seed: None | int = None):
        """
        Args:
            num_points (int): Number of nodes in the graph
            num_regions (int): Number of regions in the created partition
            num_features (int): Length of feature vectors
            beta_parameter (float, optional): Beta parameter for the feature vector generation. Defaults to 1.0.
            seed (int or None, optional): Random seed for reproducibility. Defaults to None.
        """
        # Set parameters
        self.N: int = num_points
        self.K: int = num_regions
        self.L: int = num_features
        assert self.K <= self.L
        self.beta: float = beta_parameter
        assert self.beta >= 0
        self.seed: None | int = seed

        # Initialize an empty graph with N nodes
        self.graph: igraph.Graph = igraph.Graph(self.N)


    def generate_points(self) -> np.ndarray:
        """ 
        Uniformly sample points from the unit square [0, 1]^2.
        """
        return np.random.rand(self.N, 2)
    

    def draw_partition(self, node_size: int = 100, with_labels = False):
        """  
        Draw the graph with its partition
        """
        # check that it has a partition and 
        if "P" not in self.graph.attributes() or "pos" not in self.graph.attributes():
            print("First create a partition")
            return
        # Draw the graph, with the partition
        draw_graph_partition(self.graph, P = self.graph["P"], pos = self.graph["pos"],
                            node_size=node_size, with_labels=with_labels)


    def generate_instance(self):
        """ 
        Main function.
        Generates a graph instance for the p-regions problem.
        """
        # Control replicability
        if self.seed:
            np.random.seed(self.seed)

        # Generate graph using 2D points
        points: np.ndarray = self.generate_points()
        self.graph = get_gabriel_graph(points)
        self.graph["pos"] = {i: (float(x), float(y)) for i, (x, y) in enumerate(points)}
        self.graph.vs["name"] = [str(i) for i in range(self.N)]

        # Generate a partition in K regions
        P: dict[int, list[int]] = generate_partition(self.graph, self.K)
        self.graph["P"] = P

        # Generate production vectors
