import networkx as nx
from itertools import product
import networkx.algorithms as nx_algorithms
from networkx.algorithms.community.quality import modularity
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

import lib.gnn as gnn
from lib.utils import estimate_gas_flow, add_norm_capacity, build_clustered_graph, reconnect_nodes, filter_nodes, find_largest_subgraph


def path_contraction(
        G:nx.DiGraph,
        keep_nodes:list=None
    ) -> nx.DiGraph:
    """
    Correctly performs path contraction on a DiGraph with bidirectional edges.
    """
    # Create an undirected version of the graph. This merges the A->B and B->A edges
    G_undirected = nx.Graph()
    G_undirected.add_nodes_from(G.nodes(data=True))

    processed_edges = set()
    for u, v, forward_data in G.edges(data=True):
        # Use a sorted tuple to uniquely identify the undirected edge
        edge_key = tuple(sorted((u, v)))
        if edge_key in processed_edges:
            continue

        # Check if a reverse edge exists to merge properties
        if G.has_edge(v, u):
            reverse_data = G.edges[v, u]
            # Combine data by averaging length and taking the minimum diameter
            new_L = (forward_data.get('L', 0) + reverse_data.get('L', 0)) / 2
            new_DN = min(forward_data.get('DN', 0), reverse_data.get('DN', 0))
            new_Pmax = min(forward_data.get('Pmax', 0), reverse_data.get('Pmax', 0))

            combined_data = {
                'L': new_L,
                'DN': new_DN,
                'Pmax': new_Pmax,
                'capacity' : estimate_gas_flow(new_Pmax, new_DN, new_L)
            }
            G_undirected.add_edge(u, v, **combined_data)
        else:
            # If only a one-way edge exists, just add its data
            G_undirected.add_edge(u, v, **forward_data)
        
        processed_edges.add(edge_key)

    # Performing path contraction on the undirected graph
    while True:
        # Find all nodes with a degree of exactly 2
        nodes_to_contract = [n for n, d in G_undirected.degree() if d == 2]

        if not nodes_to_contract:
            break

        for node in nodes_to_contract:
            # Check if the node is still a valid target as degree might have changed during this while-loop pass
            if node not in G_undirected or G_undirected.degree(node) != 2:
                continue

            # Get the two neighbors of the degree-2 node
            neighbors = list(G_undirected.neighbors(node))
            n1, n2 = neighbors[0], neighbors[1]

            # Avoid creating self-loops
            if n1 == n2:
                continue

            # Get data from the two edges connected to the node
            edge1_data = G_undirected.edges[n1, node]
            edge2_data = G_undirected.edges[node, n2]

            new_L = edge1_data.get('L', 0) + edge2_data.get('L', 0)
            new_Pmax = min(edge1_data.get('Pmax', float('inf')), edge2_data.get('Pmax', float('inf')))
            new_DN = min(edge1_data.get('DN', float('inf')), edge2_data.get('DN', float('inf')))

            # Aggregate properties for the new combined edge
            new_edge_data = {
                'edge_name': f"{n1}^{n2}_contracted",
                'edge_type': 'pipe_contracted',
                'L': new_L,
                'Pmax': new_Pmax,
                'DN': new_DN,
                'capacity' : estimate_gas_flow(new_Pmax, new_DN, new_L)
            }

            G_undirected.remove_node(node)
            G_undirected.add_edge(n1, n2, **new_edge_data)

    # Rebuild the final DiGraph ensuring every edge is bidirectional
    G_simplified = nx.DiGraph()
    G_simplified.add_nodes_from(G_undirected.nodes(data=True))
    for u, v, data in G_undirected.edges(data=True):
        G_simplified.add_edge(u, v, **data)
        G_simplified.add_edge(v, u, **data.copy())

    G_simplified = reconnect_nodes(
        G,
        G_simplified,
        filter_nodes(G, keep_nodes)
    )
    add_norm_capacity(G_simplified)
    return G_simplified

def calculate_absorber_score(n, G):
    w_capacity = 0.8
    w_degree = 0.2
    
    # Sum the capacity of all pipes connected to the neighbor n
    total_neighbor_capacity = sum(d.get('capacity', 0) for _, _, d in G.edges(n, data=True))
    
    return (w_capacity * total_neighbor_capacity) + (w_degree * G.degree(n))


def importance_removal(
    G:nx.DiGraph,
    importance_scores:dict,
    keep_nodes:list=None,
    removal_fraction:float=0.1,
) -> nx.DiGraph:
    """
    Simplifies a graph using Node Absorption. Low-importance nodes are removed,
    and their connections are re-routed to their most important neighbor.
    """
    if not (0 < removal_fraction < 1):
        raise ValueError("removal_fraction must be between 0 and 1.")

    G_simplified = G.copy()
    
    # Rank nodes by importance (lowest score first)
    ranked_nodes = sorted(
        (n, importance_scores.get(n, float('inf'))) for n in G_simplified.nodes()
    )

    # Identify the set of nodes to remove
    num_to_remove = int(len(ranked_nodes) * removal_fraction)
    nodes_to_remove = {node for node, score in ranked_nodes[:num_to_remove]}

    # For each node to be removed plan the re-wiring of its connections
    new_edges_to_add = []
    for node in nodes_to_remove:
        # Ensure the node still exists in the graph
        if not G_simplified.has_node(node):
            continue

        predecessors = list(G_simplified.predecessors(node))
        successors = list(G_simplified.successors(node))
        neighbors = set(predecessors) | set(successors)
        
        if not neighbors:
            continue
            
        # Choose the neighbor with the highest absorber score as the absorber
        valid_neighbors = [n for n in neighbors if n not in nodes_to_remove]
        if not valid_neighbors:
            continue
        
        absorber_node = max(valid_neighbors, key=lambda n: calculate_absorber_score(n, G_simplified))
        
        # Re-wire predecessors to the absorber_node
        for p in predecessors:
            if p != absorber_node:
                # Copy the original edge data for the new re-wired edge
                edge_data = G_simplified.edges[p, node]
                new_edges_to_add.append((p, absorber_node, edge_data))
                
        # Re-wire successors to the absorber_node
        for s in successors:
            if s != absorber_node:
                # Copy the original edge data for the new re-wired edge
                edge_data = G_simplified.edges[node, s]
                new_edges_to_add.append((absorber_node, s, edge_data))

    # Perform all graph modifications at the end
    G_simplified.add_edges_from(new_edges_to_add)
    G_simplified.remove_nodes_from(nodes_to_remove)
    G_simplified = reconnect_nodes(
        G,
        G_simplified,
        filter_nodes(G, keep_nodes)
    )
    G_simplified = find_largest_subgraph(G_simplified)
    add_norm_capacity(G_simplified)
    return G_simplified

def k_core(
        G:nx.DiGraph, 
        keep_nodes:list=None,
        k:int=4,
    ) -> nx.DiGraph:
    G_simplified = nx_algorithms.core.k_core(G, k)
    G_simplified = reconnect_nodes(
        G,
        G_simplified,
        filter_nodes(G, keep_nodes)
    )
    add_norm_capacity(G_simplified)
    return G_simplified

def gnn_clustering(
        G:nx.DiGraph, 
        keep_nodes:list=None,
        n_clusters:int=None, 
        coord_weight:float=None
    ) -> list[frozenset]:
    """
    Main pipeline function to run the entire GNN-based graph simplification process.
    If n_clusters or coord_weight is None, they will be determined automatically.
    """
    # Prepare data with a neutral weight initially for the GNN training
    pyg_data, node_names_map, scaled_coords = gnn.prepare_data(G, coord_weight=1.0)
    trained_model = gnn.train_gnn_model(pyg_data)

    # If hyperparameters are not provided, find the optimal values
    if n_clusters is None or coord_weight is None:
        n_clusters, coord_weight = gnn.find_optimal_hyperparameters(trained_model, pyg_data, scaled_coords)

    # Generate the final communities and graph using the determined hyperparameters
    communities = gnn.get_gnn_communities(trained_model, pyg_data, node_names_map, n_clusters, scaled_coords, coord_weight)
    graph = build_clustered_graph(G, communities)
    
    return communities, graph

def find_optimal_geo_clusters(scaled_coords:np.ndarray, k_range: range = range(20, 251, 10)) -> int:
    best_score = -1
    best_k = -1

    for k in k_range:
        # Run KMeans for the current k value
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_coords)
        
        # Calculate the silhouette score, requires at least 2 to be valid
        if len(set(labels)) > 1:
            score = silhouette_score(scaled_coords, labels)
            
            # Update the best score and k if this one is better
            if score > best_score:
                best_score = score
                best_k = k
        else:
            print(f"  k = {k}, Not enough clusters to calculate a score.")

    if best_k == -1:
        print("Could not determine optimal k. Defaulting to the start of the range.")
        return k_range.start

    print(f"\n---> Optimal k found: {best_k} with score {best_score:.4f} <---")
    return best_k

def k_means(
        G:nx.DiGraph, 
        keep_nodes:list=None, 
        n_clusters:int=None
    ) -> list[frozenset]:
    # Extract node coordinates
    if not G.nodes:
        print("Graph has no nodes. Returning empty list.")
        return []

    try:
        node_list = list(G.nodes())
        coords = np.array([G.nodes[node]['coord'] for node in node_list])
    except KeyError:
        print("Error: Nodes in the graph are missing the 'coord' attribute.")
        return []

    scaler = MinMaxScaler()
    coords_scaled = scaler.fit_transform(coords)

    # Determine cluster count and run KMeans
    if n_clusters is None:
        n_clusters = find_optimal_geo_clusters(coords_scaled)

    print(f"\nClustering {len(node_list)} nodes into {n_clusters} communities based on location...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coords_scaled)
    print("Clustering complete.")

    # Format communities
    communities = [[] for _ in range(n_clusters)]
    for i, label in enumerate(cluster_labels):
        node_name = node_list[i]
        communities[label].append(node_name)

    communities = [frozenset(community) for community in communities if community]
    
    print(f"Geographic Clustering Finished. Found {len(communities)} communities\n")

    G_simplified = build_clustered_graph(G, communities)
    return communities, G_simplified

# Capacity?, norm_capacity
def maximum_spanning_arborescence(G:nx.DiGraph, keep_nodes:list=None, attr:str='capacity') -> nx.DiGraph:
    arborescence = nx.maximum_spanning_arborescence(G, attr=attr)

    # Explicitly copy node data from the original graph to the new one
    for node, data in G.nodes(data=True):
        arborescence.add_node(node, **data)
    
    return arborescence

def greedy_modularity_communities(G:nx.DiGraph, keep_nodes:list=None) -> list[frozenset]:
    communities = nx.algorithms.community.greedy_modularity_communities(G, weight='capacity')
    G_simplified = build_clustered_graph(G, communities)
    print("Gefundene Communities: " + str(len(communities)))
    print(f"Modularität Q = {modularity(G, communities)}")
    return communities, G_simplified

def louvain_communities(G:nx.DiGraph, keep_nodes:list=None, seed:int=42) -> list[frozenset]:
    communities = nx.algorithms.community.louvain_communities(G, weight='capacity', seed=seed)
    G_simplified = build_clustered_graph(G, communities)
    print("Gefundene Communities: " + str(len(communities)))
    print(f"Modularität Q = {modularity(G, communities)}")
    return communities, G_simplified