import igraph
import numpy as np
from .utils import get_gabriel_graph, draw_graph_partition, generate_partition, compute_cluster_labels_from_P
from .k_means_optimality import get_k_means_optimality_status


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


    def get_alpha_parameters(self) -> dict[int, np.ndarray]:
        """  
        Get the parameter alpha_k for each region k
        """
        # Useful variables
        identity = np.eye(self.L)
        ones = np.ones(self.L)

        # Complete for each region: αk = 1 + βek
        parameters = {}
        for k in range(self.K):
            parameters[k+1] = ones + self.beta * identity[k]
        return parameters


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

        # Generate node attribures for each node
        alpha_parameters: dict[int, np.ndarray] = self.get_alpha_parameters()
        for k, P_k in P.items():
            self.graph.vs[P_k]["x"] = np.random.dirichlet(alpha_parameters[k], size = len(P_k))

        # Compute optimality status
        X = np.array(self.graph.vs["x"])
        cluster_labels = compute_cluster_labels_from_P(P, self.N) 
        status: str = get_k_means_optimality_status(X, self.K, cluster_labels)
        self.graph["status"] = status


    def save_instance(self, output_path):
        """ 
        Save the graph instance as a pickle file
        """
        with open(output_path, "wb") as f:
            self.graph.write_pickle(f)
