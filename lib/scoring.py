import time
import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance

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

def structure(
        original:nx.Graph, 
        simplified:nx.Graph, 
        verbose:bool=False
    ) -> float:
    """
    Calculate and return the structure part of the scoring function
    """
    # Algebraic Connectivity: The second-smallest eigenvalue
    ac_simplified = nx.linalg.algebraic_connectivity(simplified, weight='norm_capacity')
    ac_original = nx.linalg.algebraic_connectivity(original, weight='norm_capacity')
    denominator = ac_simplified + ac_original
    if denominator < 1e-9:
        ac_score = 0.0 if ac_original > 1e-9 else 1.0
    else:
        ac_score = 1 - (abs(ac_simplified - ac_original) / denominator)


    # Distribution of Betweenness Centrality: Earth Mover's Distance (EMD)
    centrality_before = nx.betweenness_centrality(
        original, 
        k=int(0.1 * len(original.nodes())), 
        weight='L', 
        seed=42
    )
    centrality_after_raw = nx.betweenness_centrality(
        simplified, 
        k=int(0.1 * len(simplified.nodes()) if 0.1 * len(simplified.nodes()) > 300 else len(simplified.nodes())), 
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
    if max_emd > 1e-9:
        emd_score = 1 - (wasserstein_distance(dist_before, dist_after) / max_emd)
    else:
        emd_score = 1.0

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
    structure_score = (ac_score + emd_score + spectral_score) / 3

    if verbose:
        print(f"ac_simplified: {ac_simplified}")
        print(f"ac_original: {ac_original}")
        print(f"ac_score: {ac_score}")
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
    
    simplified_undirected = utils.convert_digraph_to_graph(simplified)
    original_undirected = utils.convert_digraph_to_graph(original)
    
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