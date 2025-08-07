import igraph
import numpy as np
from .utils import get_grabriel_graph


class Instance_Generator:
    """ 
    Class to generate a graph instance for the p-regions problem.

    The created graph is an instance of an igraph graph.
    Node atributes:
        "x": list =  a vector defining node attributes
        "name": str = the index of the node, in stirng
    Graph attributes:
        "pos": dict = A dictionary indicating node spatial coordinates
        "P": dict = A partition created in the graph
        "status": str = Indicates wheter the created partition P is optimal for p-regions
    """

    def __init__(self, num_points: int, num_regions: int, num_features: int,
                 beta_parameter: float = 1.0, seed: None | int = None):
        """
        Args:
            num_points (int): Number of nodes in the graph
            num_regions (int): Number of regions in the created partition
            num_features (int): Length of production vectors
            beta (int, optional): beta parameter for the production vector creation. Defaults to 1.
        """
        # Set parameters
        self.N: int = num_points
        self.K: int = num_regions
        self.L: int = num_features
        assert self.K <= self.L
        self.beta: float = beta_parameter
        assert self.beta >= 0
        self.seed: None | int = seed

        # Empty graph with N nodes
        self.graph: igraph.Graph = igraph.Graph(self.N)


    def generate_points(self) -> np.ndarray:
        """ 
        Uniformily sample points from the unit square [0, 1]^2
        """
        return np.random.rand(self.N, 2)
    

    def generate_grabirel_graph(self):
        """ 
        Creates a grabriel graph from 2d points
        Add pos dict as a graph attribute
        """
        
        # Sample points  
        points: np.ndarray = self.generate_points()
        # Get grabriel graph
        self.graph = get_grabriel_graph(points)

        # Save pos and anme atributes
        self.graph["pos"] = {i: (x, y) for i, (x, y) in enumerate(points)}
        self.graph.vs["name"] = [str(i) for i in range(self.N)]


    def generate_graph(self):
        """ 
        Main function.
        Generates a graph instance of the p-regions problem
        """
        # Control replicability
        if self.seed:
            np.random.seed(self.seed)

        # Generate graph using points
        self.generate_grabirel_graph()

        # Define a partition, and production vectors
        self.include_graph_partition(graph)
        self.include_node_attributes(graph)

        # Report optimality status
        self.include_optimality_status(graph)


