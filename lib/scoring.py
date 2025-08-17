import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

import lib.simulation as sim
import lib.utils as utils

def complexity(
        original:nx.Graph, 
        simplified:nx.Graph, 
        verbose:bool=False
    ) -> float:
    """
    Calculate and return the complexity part of the scoring function
    """
    # Calculate node count score
    nodes_score = 1 - (simplified.number_of_nodes() / original.number_of_nodes())
    
    # Calculate edge count score
    edges_score = 1 - (simplified.number_of_edges() / original.number_of_edges())
    
    # Calculate cyclomatic number
    pre_cyclo = original.number_of_edges() - original.number_of_nodes() + nx.number_connected_components(original)
    post_cyclo = simplified.number_of_edges() - simplified.number_of_nodes() + nx.number_connected_components(simplified)
    # Ensure score is not negative
    cyclo_score = 1 - (post_cyclo / pre_cyclo) if pre_cyclo > 0 else 0
    cyclo_score = cyclo_score if cyclo_score > 0 else 0

    # Calculating score
    complexity_score = (nodes_score + edges_score + cyclo_score) / 3

    if verbose:
        print(f"nodes_score: {nodes_score}")
        print(f"edges_score: {edges_score}")
        print(f"pre_cyclo: {pre_cyclo}")
        print(f"post_cyclo: {post_cyclo}")
        print(f"cyclomatic_score: {cyclo_score}")
        print(f"complexity_score: {complexity_score}\n")

    # Average the three complexity scores and return result
    return complexity_score

def get_portrait(
        graph: nx.Graph, 
        k: int, 
        weight: str = 'L', 
        seed: int = 42
    ) -> np.ndarray:
    """
    Computes an approximated network portrait matrix B from a sample of k nodes.
    """
    if not graph or graph.number_of_nodes() == 0:
        return np.zeros((1, 1), dtype=int)
    
    # Use a sample of nodes
    nodes = list(graph.nodes())
    if len(nodes) > k:
        # Use a seed for reproducibility
        rng = np.random.default_rng(seed)
        sampled_nodes = rng.choice(nodes, size=k, replace=False)
    else:
        sampled_nodes = nodes

    num_nodes = graph.number_of_nodes()
    max_dist = 0
    portrait_data = [] # Store raw (l, k) pairs first

    # Calculate paths only from sampled nodes
    for node in sampled_nodes:
        try:
            # single_source is much faster than all_pairs
            distances_from_node = nx.single_source_dijkstra_path_length(graph, node, weight=weight)
            
            dist_counts = {}
            for dist in distances_from_node.values():
                if dist > 0 and dist != float('inf'):
                    rounded_dist = int(np.round(dist))
                    max_dist = max(max_dist, rounded_dist)
                    if rounded_dist not in dist_counts:
                        dist_counts[rounded_dist] = 0
                    dist_counts[rounded_dist] += 1
            
            # For each distance l from this source node, we found k neighbors
            for l, k_count in dist_counts.items():
                portrait_data.append((l, k_count))

        except nx.NetworkXError:
            continue

    if not portrait_data:
        return np.zeros((1, 1), dtype=int)

    # Initialize the portrait matrix with the now known max distance
    diameter = max_dist
    portrait = np.zeros((diameter + 1, num_nodes), dtype=int)
    
    # Populate the portrait from the collected data
    for l, k in portrait_data:
        if l <= diameter and k < num_nodes:
            portrait[l, k] += 1
            
    return portrait
    
def structure(
        original:nx.Graph, 
        simplified:nx.Graph, 
        verbose:bool=False
    ) -> float:
    """
    Calculate and return the structure part of the scoring function
    """
    APPROX_ORIGINAL = int(0.2 * len(original.nodes()) if 0.2 * len(original.nodes()) > 300 else len(original.nodes()))
    APPROX_SIMPLIFIED = int(0.2 * len(simplified.nodes()) if 0.2 * len(simplified.nodes()) > 300 else len(simplified.nodes()))

    # Portrait Divergence (PDIV)
    portrait_before = get_portrait(
        original, 
        k=APPROX_ORIGINAL,
        weight='L'
    )
    portrait_after = get_portrait(
        simplified, 
        k=APPROX_ORIGINAL,
        weight='L'
    )

    # Pad the smaller portrait to match the shape of the larger one
    max_diam = max(portrait_before.shape[0], portrait_after.shape[0])
    max_nodes = max(portrait_before.shape[1], portrait_after.shape[1])

    padded_before = np.zeros((max_diam, max_nodes))
    padded_before[:portrait_before.shape[0], :portrait_before.shape[1]] = portrait_before

    padded_after = np.zeros((max_diam, max_nodes))
    padded_after[:portrait_after.shape[0], :portrait_after.shape[1]] = portrait_after

    # Flatten matrices to 1D arrays for distance calculation
    p_dist = padded_before.flatten()
    q_dist = padded_after.flatten()
    
    pdiv_score = 0.0
    # Avoid division by zero
    if np.sum(p_dist) > 0 and np.sum(q_dist) > 0:
        # Jensen-Shannon distance is bounded [0, 1] for base=2.
        js_distance = jensenshannon(p_dist, q_dist, base=2)
        pdiv_score = 1.0 - js_distance
    elif np.array_equal(p_dist, q_dist):
        pdiv_score = 1.0

    # Distribution of Betweenness Centrality: Earth Mover's Distance (EMD)
    centrality_before = nx.betweenness_centrality(
        original, 
        k=APPROX_ORIGINAL, 
        weight='L', 
        seed=42
    )
    centrality_after_raw = nx.betweenness_centrality(
        simplified, 
        k=APPROX_SIMPLIFIED, 
        weight='L', 
        seed=42
    )
    
    node_map = nx.get_node_attributes(simplified, 'original_nodes')
    centrality_after = {}
    for simp_node, orig_nodes_list in node_map.items():
        for orig_node in orig_nodes_list:
             centrality_after[orig_node] = centrality_after_raw.get(simp_node, 0.0)
    
    dist_before = np.array(list(centrality_before.values()))
    dist_after_mapped = [centrality_after.get(n, 0.0) for n in original.nodes()]
    dist_after = np.array(dist_after_mapped)

    max_emd = np.max(dist_before) if len(dist_before) > 0 else 0
    if max_emd > 0:
        emd_score = 1 - (wasserstein_distance(dist_before, dist_after) / max_emd)
    else:
        emd_score = 1.0

    # Spectral Distance of Graph Laplacians
    # Get a fixed ordering of original nodes for consistent matrix construction
    nodelist = list(original.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodelist)}
    num_nodes = len(nodelist)

    # Use the normalized weight for consistency
    laplacian_before = nx.laplacian_matrix(original, nodelist=nodelist, weight='norm_capacity').toarray()
    laplacian_after = np.zeros((num_nodes, num_nodes))

    node_attributes = nx.get_node_attributes(simplified, 'original_nodes')

    if node_attributes:
        # If it's a clustered/virtual graph
        orig_to_simp_map = {}
        for simp_node, orig_nodes_list in node_attributes.items():
            for orig_node in orig_nodes_list:
                orig_to_simp_map[orig_node] = simp_node

        simplified_edges = {tuple(sorted((u, v))): data.get('norm_capacity', 0.0)
                            for u, v, data in simplified.edges(data=True)}

        adj_after_projected = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                u_orig, v_orig = nodelist[i], nodelist[j]
                u_simp, v_simp = orig_to_simp_map.get(u_orig), orig_to_simp_map.get(v_orig)

                if u_simp is None or v_simp is None: continue
                if u_simp == v_simp:
                    adj_after_projected[i, j] = adj_after_projected[j, i] = 1.0
                else:
                    edge_tuple = tuple(sorted((u_simp, v_simp)))
                    if edge_tuple in simplified_edges:
                        weight = simplified_edges[edge_tuple]
                        adj_after_projected[i, j] = adj_after_projected[j, i] = weight

        degree_matrix = np.diag(adj_after_projected.sum(axis=1))
        laplacian_after = degree_matrix - adj_after_projected

    else:
        # If it's not clustered
        nodes_in_simplified = [node for node in nodelist if node in simplified]
        if nodes_in_simplified:
            sub_laplacian = nx.laplacian_matrix(simplified, nodelist=nodes_in_simplified, weight='norm_capacity').toarray()
            after_indices = [node_to_idx[node] for node in nodes_in_simplified]
            laplacian_after[np.ix_(after_indices, after_indices)] = sub_laplacian

    dist_raw = np.linalg.norm(laplacian_before - laplacian_after, 'fro')
    normalization_factor = np.linalg.norm(laplacian_before, 'fro') + np.linalg.norm(laplacian_after, 'fro')

    if normalization_factor > 0:
        # Use exponential decay for a normalized score
        dissimilarity_ratio = dist_raw / normalization_factor
        spectral_score = np.exp(-dissimilarity_ratio)
    else:
        spectral_score = 1.0

    # Calculating score
    structure_score = (pdiv_score + emd_score + spectral_score) / 3

    if verbose:
        print(f"pdiv_score: {pdiv_score}")
        print(f"emd_score: {emd_score}")
        print(f"spectral_dist_score: {spectral_score}")
        print(f"structure_score: {structure_score}\n")

    return structure_score

def count_regions(G:nx.Graph, regions:gpd.GeoDataFrame) -> int:
    df = gpd.GeoDataFrame(utils.graph_to_nodes_df(G))
    df = df.set_geometry(gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:3035").to_crs("EPSG:4326")
    df = df.sjoin(regions, how="left", predicate='within')
    df = df.value_counts("NUTS_ID")
    df = df[df > 0]
    return df.size

def regionality(
        original:nx.Graph, 
        simplified:nx.Graph, 
        regions:gpd.GeoDataFrame, 
        verbose:bool=False
    ) -> float:
    """
    Calculate and return the regionality part of the scoring function
    """
    original_regions = count_regions(original, regions)
    simplified_regions = count_regions(simplified, regions)
    regionality_score = simplified_regions / original_regions if original_regions > 0 else 1.0

    if verbose:
        print(f"original_regions: {original_regions}")
        print(f"simplified_regions: {simplified_regions}")
        print(f"regionality_score: {regionality_score}\n")

    return regionality_score

def get_node_role_score(node_name: str, role_weights: dict) -> float:
    """
    Finds the categorical role score for a single node based on its name.
    """
    for key, value in role_weights.items():
        if key in node_name:
            return value
    return role_weights.get('default', 0.1)

def calculate_total_properties(G: nx.Graph, role_weights: dict) -> float:
    """
    Calculates the total property value of a graph using an edge-centric formula.
    """
    total_graph_score = 0
    for u, v, data in G.edges(data=True):
        capacity = data.get('capacity', 0.0)
        s_role_u = get_node_role_score(str(u), role_weights)
        s_role_v = get_node_role_score(str(v), role_weights)
        
        # The property value of an edge is its capacity multiplied by the importance of the nodes it connects
        edge_property_value = capacity * (s_role_u + s_role_v)
        total_graph_score += edge_property_value
    return total_graph_score

def properties(
        original: nx.Graph,
        simplified: nx.Graph,
        weights: dict,
        verbose: bool = False
    ) -> float:
    """
    Calculates the properties part of the scoring fuction.
    """
    node_map = nx.get_node_attributes(simplified, 'original_nodes')

    if not node_map:
        # If it's not a clustered graph
        original_properties = calculate_total_properties(original, weights)
        simplified_properties = calculate_total_properties(simplified, weights)
    else:
        # If it's a clustered graph
        original_properties = 0
        simplified_properties = 0

        node_to_group_map = {node: simp_node for simp_node, orig_nodes in node_map.items() for node in orig_nodes}

        for u, v, data in original.edges(data=True):
            capacity = data.get('capacity', 0.0)
            s_role_u = get_node_role_score(str(u), weights)
            s_role_v = get_node_role_score(str(v), weights)
            
            edge_property_value = capacity * (s_role_u + s_role_v)
            original_properties += edge_property_value
            
            group_u = node_to_group_map.get(u)
            group_v = node_to_group_map.get(v)
            
            if group_u is not None and group_v is not None and group_u != group_v:
                simplified_properties += edge_property_value

    # Normalization for both cases
    if original_properties > 0:
        properties_score = simplified_properties / original_properties
        properties_score = min(properties_score, 1.0)
    else:
        properties_score = 0
    
    if verbose:
        print(f"original_properties: {original_properties}")
        print(f"simplified_properties: {simplified_properties}")
        print(f"properties_score: {properties_score}\n")
    
    return properties_score


def flow(original: nx.Graph, simplified: nx.Graph, verbose: bool = False) -> float:
    """
    Calculates the flow part of the scoring function.
    """
    max_dev_error = sim.calculate_deliverability_error(original, simplified)
    flow_score = 1 - max_dev_error

    if verbose:
        print(f"max_dev_error: {max_dev_error}")
        print(f"flow_score: {flow_score}\n")

    return flow_score

def score(
    original: nx.DiGraph,
    simplified: nx.DiGraph,
    regions: gpd.GeoDataFrame,
    weights: dict = {},
    property_weights: dict = {},
    verbose: bool = False
) -> float:
    """
    Scoring function assessing the quality of the gas network simplification using a weighted sum of different criteria.
    """
    # Apply weights from the dictionary
    w_complexity = weights.get('complexity', 0.2)
    w_structure = weights.get('structure', 0.2)
    w_regionality = weights.get('regionality', 0.2)
    w_properties = weights.get('properties', 0.2)
    w_flow = weights.get('flow', 0.2)
    
    # Check if weights sum to 1.0
    total_weight = w_complexity + w_structure + w_regionality + w_properties + w_flow
    if not np.isclose(total_weight, 1.0):
        print(f"Warning: Weights do not sum to 1.0 (Sum = {total_weight})")
        return
    
    simplified_undirected = utils.convert_to_graph(simplified)
    original_undirected = utils.convert_to_graph(original)
    
    score_complexity = complexity(original_undirected, simplified_undirected, verbose)
    score_structure = structure(original_undirected, simplified_undirected, verbose)
    score_regionality = regionality(original_undirected, simplified_undirected, regions, verbose)
    score_properties = properties(original_undirected, simplified_undirected, property_weights, verbose)
    score_flow = flow(original, simplified, verbose)
    

    # Calculate the final score as a weighted sum
    final_score = (
        w_complexity * score_complexity +
        w_structure * score_structure +
        w_regionality * score_regionality +
        w_properties * score_properties +
        w_flow * score_flow
    )
    
    if verbose:
        print(f"Overall Weighted Score: {final_score}")
    
    return final_score