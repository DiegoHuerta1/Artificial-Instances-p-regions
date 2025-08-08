import numpy as np
import igraph
from scipy.spatial import Delaunay as Delaunay_scipy
import seaborn as sns
import networkx as nx
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.lines

# ------------------------------------------------------
# Generating the graph

def get_delaunay_graph(points: np.ndarray) -> igraph.Graph:
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


def get_gabriel_graph(points: np.ndarray) -> igraph.Graph:
    """ 
    Create an igraph Gabriel graph based on coordinate points.
    """
    # Initialize the graph with the correct number of nodes
    graph: igraph.Graph = igraph.Graph(points.shape[0])

    # Get Delaunay edges
    delaunay_graph = get_delaunay_graph(points)
    pairs = np.array(delaunay_graph.get_edgelist())

    # Filter edges using the beta-skeleton criterion 
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
    center_1 = (p + q) / 2
    center_2 = (q + p) / 2

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


# ---------------------------------------------------------------
# Generating the partition

def __get_all_assignments(graph: igraph.Graph, P: dict) -> list[tuple[int, int]]:
    """ 
    Get possible assignments (v, k) for an incomplete partition P.
    """
    # Get assigned nodes of P
    assigned_nodes = []
    for P_k in P.values():
        assigned_nodes.extend(P_k)

    # Compute all possible assignments 
    # {(v, k) | (∄h ∈ [K] : v ∈ Ph) ∧ (∃u ∈ N(v) : u ∈ Pk)}
    possible_assignments: list[tuple[int, int]] = []
    
    # Iterate on assigned nodes
    for k, P_k in P.items():
        for u in P_k:
            # Iterate on unassigned neighbors
            for v in graph.neighbors(u):
                if v not in assigned_nodes:
                    # Save the element
                    possible_assignments.append((v, k))

    return possible_assignments
    

def __select_assignment(P: dict[int, list[int]], F: list[tuple[int, int]]) -> tuple[int, int]:
    """  
    Given possible assignments (v, k), select the next node v_star, k_star.
    """
    # Select the region with the fewest nodes
    S = list(set([k for (v, k) in F]))
    k_star = min(S, key=lambda k: len(P[k]))

    # Select v_star randomly
    filtered_assignments = [(v, k) for (v, k) in F if k == k_star]
    selected_index = np.random.choice(len(filtered_assignments))

    # Return assignment
    selected_assignment = filtered_assignments[selected_index]
    return selected_assignment


def __recompute_assignments(graph: igraph.Graph, P: dict[int, list[int]],
                            F_old: list[tuple[int, int]],
                            v_star: int, k_star: int) -> list[tuple[int, int]]:
    """  
    Recompute the possible assignments after selecting (v_star, k_star).
    """
    # Remove assignments
    F_filter = [(v, k) for (v, k) in F_old if v != v_star]

    # Get assigned nodes of P
    assigned_nodes = []
    for P_k in P.values():
        assigned_nodes.extend(P_k)

    # Add new assignments
    # {(v, k∗) : (v ∈ N (v∗)) ∧ (∄h ∈ [K] : v ∈ Ph) ∧ ((v, k∗) /∈ F)}
    F_new: list[tuple[int, int]] = []
    for v in graph.neighbors(v_star):
        if v not in assigned_nodes and (v, k_star) not in F_filter:
            F_new.append((v, k_star))
    
    # Return filtered old and new assignments together
    return F_filter + F_new


def generate_partition(graph: igraph.Graph, num_regions: int) -> dict[int, list[int]]:
    """ 
    Generates a partition on an igraph graph.
    """
    # Start partition with K seeds
    seeds = np.random.choice(graph.vcount(), size=num_regions, replace=False)
    P: dict[int, list[int]] = {(idx+1): [int(seed)] for idx, seed in enumerate(seeds)}

    # Get possible assignments (v, k)
    F: list[tuple[int, int]] = __get_all_assignments(graph, P)

    # Construct partition iteratively
    while len(F) > 0:
        # Select an element
        v_star, k_star = __select_assignment(P, F)
        P[k_star].append(v_star)
        # Compute possible assignments again
        F = __recompute_assignments(graph, P, F, v_star, k_star)
    return P


# --------------------------------------------------------
# Visualization

def get_dict_palette(num_colors: int, palette_name: str) -> dict:
    """ 
    Get a dictionary of a color palette.

    Palette name options:
    deep, muted, bright, pastel, dark, colorblind, husl, hls
    """
    # Start with seaborn palette
    colors = sns.color_palette(palette_name, num_colors)
    # Transform to hex, and dict
    hex_colors = [matplotlib.colors.to_hex(rgb) for rgb in colors]
    dict_palette = {(idx+1): color for idx, color in enumerate(hex_colors)}

    return dict_palette


def draw_graph_partition(graph: igraph.Graph, P: dict,
                         pos=None,
                         palette_name: str = "husl",
                         figsize: tuple = (4, 4),
                         title: str | None = None,
                         node_size=100, with_labels=False):
    """ 
    Draw a partition P (that has names of nodes, not indices).
    """
    # Create an equivalent nx graph
    edges = graph.get_edgelist()
    nx_graph: nx.Graph = nx.Graph()
    nx_graph.add_edges_from(edges)

    # Get the color palette of the partition
    colors_P: dict = get_dict_palette(len(P), palette_name)

    # Get the color of each node
    node_colors: list = []
    for v in nx_graph.nodes:
        for k, P_k in P.items():
            if v in P_k:
                node_colors.append(colors_P[k])

    # Create pos if not available
    if not pos:
        pos = nx.spring_layout(nx_graph, iterations=1000, seed=1)

    # Make the figure
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(nx_graph, pos=pos, node_color=node_colors, ax=ax,
            node_size=node_size, with_labels=with_labels)
    # Legend
    legend_handles = []
    for k, color_k in colors_P.items():
        legend_handles.append(matplotlib.lines.Line2D([0], [0], marker='o', color='w',
                                                      label=f"P_{k}", markerfacecolor=color_k,
                                                      markersize=10))            
    ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.3, 0.9))
    # Final details
    if title is not None:
        ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()
