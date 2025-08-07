import numpy as np
import igraph
from scipy.spatial import Delaunay as Delaunay_scipy


def get_delunay_graph(points: np.ndarray) -> igraph.Graph:
    """ 
    Create an igraph Delaunay graph based on coordinate points.
    """

    # Initialize an empty graph with the correct number of nodes
    graph: igraph.Graph = igraph.Graph(points.shape[0])

    # Use the scipy Delaunay triangulation
    delaunay = Delaunay_scipy(points)
    edges = []
    for tri in delaunay.simplices:
        edges.append((tri[0], tri[1]))
        edges.append((tri[1], tri[2]))
        edges.append((tri[0], tri[2]))
    graph.add_edges(edges)
    graph.simplify()

    return graph


def get_grabriel_graph(points: np.ndarray) -> igraph.Graph:
    """ 
    Create an igraph Gabriel graph based on coordinate points.
    """

    # Initialize graph with the correct number of nodes
    graph: igraph.Graph = igraph.Graph(points.shape[0])

    # Get Delaunay edges
    delunay_graph = get_delunay_graph(points)
    pairs = np.array(delunay_graph.get_edgelist())

    # Filter edges using the beta-skeleton criterion (beta=1 for Gabriel)
    final_edges = __assign_edges_beta(points, pairs, beta=1)
    graph.add_edges(final_edges)
    graph.simplify()
    return graph


def __assign_edges_beta(points, pairs, beta: float):
    """
    Assign edges to the graph based on the beta-skeleton criterion.

    Parameters
    ----------
    points : ndarray
        Array of coordinate points.
    pairs : ndarray
        Array of index pairs representing candidate edges.
    beta : float (beta >= 1)
        Control parameter for the beta-skeleton. 
        beta = 1 corresponds to the Gabriel graph.
    """

    p = points[pairs[:, 0]]
    q = points[pairs[:, 1]]
    radius = np.linalg.norm(p - q, axis=1) * beta / 2
    center_1 = p * 0.5 + q * 0.5
    center_2 = q * 0.5 + p * 0.5  # Same as center_1, included for clarity

    edges = []
    for i in np.arange(pairs.shape[0]):
        dist_1 = np.linalg.norm(points - center_1[i], axis=1)
        dist_2 = np.linalg.norm(points - center_2[i], axis=1)
        empty_test_1 = dist_1 <= radius[i]
        empty_test_2 = dist_2 <= radius[i]
        empty_test = np.delete(empty_test_1 * empty_test_2, pairs[i])
        if not np.any(empty_test):
            edges.append(pairs[i])

    return edges

