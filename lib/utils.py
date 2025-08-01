from collections import defaultdict
import geopandas as gpd
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx
from networkx.algorithms.community.quality import modularity

def graph_node_names(G: nx.DiGraph) -> list[str]:
    """
    Returns a flat list of node names from a graph.
    """
    if not G.nodes:
        return []
    
    node_data_iterator = iter(G.nodes(data=True))
    try:
        _, first_node_data = next(node_data_iterator)
        
        if 'original_nodes' in first_node_data:
            all_original_nodes = []
            all_original_nodes.extend(first_node_data['original_nodes'])
            for _, data in node_data_iterator:
                all_original_nodes.extend(data.get('original_nodes', []))
            return all_original_nodes
        else:
            return list(G.nodes())
            
    except StopIteration:
        return []

def graph_to_nodes_df(G:nx.DiGraph) -> pd.DataFrame:
    """
    Converts graph's nodes into a dataframe while preserving all attributes.
    """
    df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index').reset_index().rename(columns={"index": "nodes"})
    df["x"] = df["coord"].apply(lambda xy: xy[0])
    df["y"] = df["coord"].apply(lambda xy: xy[1])
    return df

def run_algo(original:nx.DiGraph, func) -> list[frozenset]:
    """
    Run a NetworkX algorithm and return the results.
    """
    results = func(original)
    print("Gefundene Communities: " + str(len(results)))
    mod_value = modularity(original, results)
    print(f"Modularität Q = {mod_value}")
    return results

def build_clustered_graph(
    original: nx.DiGraph,
    results: list[frozenset]
) -> nx.DiGraph:
    """
    Builds a robust simplified nx.DiGraph from community detection results.
    """
    # Map each original node to its new group (cluster) ID
    node_to_group_map = {node: i for i, community in enumerate(results) for node in community}

    # Aggregate node properties for each group
    nodes_df = pd.DataFrame.from_dict(dict(original.nodes(data=True)), orient='index')
    nodes_df['node_id'] = nodes_df.index
    nodes_df['group'] = nodes_df['node_id'].map(node_to_group_map)
    
    # Handle missing coordinate data
    nodes_df['x'] = nodes_df.get("coord", pd.Series(dtype=object)).map(lambda x: x[0] if isinstance(x, (list, tuple)) else None)
    nodes_df['y'] = nodes_df.get("coord", pd.Series(dtype=object)).map(lambda x: x[1] if isinstance(x, (list, tuple)) else None)

    simplified_nodes_agg = nodes_df.groupby("group").agg(
        x=('x', 'mean'),
        y=('y', 'mean'),
        original_nodes=('node_id', list),
        original_node_data=('node_id', lambda ids: {nid: original.nodes[nid] for nid in ids})
    )

    # Aggregate edge properties for connections between clusters
    inter_cluster_edges = defaultdict(list)
    for u, v, data in original.edges(data=True):
        group_u = node_to_group_map.get(u)
        group_v = node_to_group_map.get(v)

        if group_u is not None and group_v is not None and group_u != group_v:
            edge_tuple = (group_u, group_v)
            inter_cluster_edges[edge_tuple].append(data)

    # Create a single new edge for each directed connection between clusters
    simplified_edges = []
    for (group_u, group_v), edges_data_list in inter_cluster_edges.items():
        total_capacity = sum(d.get('capacity', 0) for d in edges_data_list)
        if total_capacity < 1e-9: continue

        # Use capacity-weighted average for length
        avg_len = sum(d.get('L', 0) * d.get('capacity', 0) for d in edges_data_list) / total_capacity
        
        # Calculate the physically equivalent diameter
        sum_of_dn_powered = sum(d.get('DN', 0) ** 2.5 for d in edges_data_list)
        equivalent_dn = sum_of_dn_powered ** (1/2.5)
        avg_pmax = sum(d.get('Pmax', 0) * d.get('capacity', 0) for d in edges_data_list) / total_capacity

        new_edge_data = {
            'capacity': total_capacity,
            'L': avg_len,
            'DN': equivalent_dn,
            'Pmax': avg_pmax,
            'edge_type': 'aggregated_pipe_equivalent'
        }
        simplified_edges.append((group_u, group_v, new_edge_data))

    # Build the final simplified graph
    simplified_graph = nx.DiGraph()
    for group_id, data in simplified_nodes_agg.iterrows():
        simplified_graph.add_node(
            f"C_{group_id}",
            coord=(data['x'], data['y']),
            x=data['x'],
            y=data['y'],
            original_nodes=data['original_nodes'],
            original_node_data=data['original_node_data']
        )
    
    simplified_graph.add_edges_from(
        (f"C_{u}", f"C_{v}", data) for u, v, data in simplified_edges
    )

    for _, simp_data in simplified_graph.nodes(data=True):
        total_cluster_supply = 0
        # Look inside the cluster to find the original nodes it contains
        for orig_node_id in simp_data['original_nodes']:
            # Get the supply value from the original graph's node
            total_cluster_supply += original.nodes[orig_node_id].get('supply', 0)
        
        # Assign the summed value as the 'supply' attribute for the cluster node
        simp_data['supply'] = total_cluster_supply

    add_norm_capacity(simplified_graph)
    return simplified_graph


def plot_network(
        graph:nx.DiGraph, 
        gdf:gpd.GeoDataFrame=None, 
        clusters:list=None, 
        node_size:int=20, 
        title:str="Title", 
        padding_ratio:float=0.15, 
        nodes:bool=True, 
        edges:bool=True
    ) -> None:
    """
    Plot any NetworkX graph with optional NUTS regions background from a GeoDataFrame.
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    if gdf is not None:
        gdf.plot(ax=ax, color='lightgrey', edgecolor='white', alpha=0.5, linewidth=2, zorder=0)

    try:
        pos = {node: (data["coord"][0], data["coord"][1]) for node, data in graph.nodes(data=True)}
    except KeyError:
        pos = {node: (data["x"], data["y"]) for node, data in graph.nodes(data=True)}

    if nodes:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'lime', 'pink', 'teal', 'gold', 'navy', 'brown', 'olive']
        if clusters is None:
            clusters = [list(graph.nodes())]
        
        for i, res in enumerate(clusters):
            nx.draw_networkx_nodes(graph, pos, nodelist=res, node_size=node_size, node_color=colors[i % len(colors)], ax=ax)

    if edges:
        nx.draw_networkx_edges(graph, pos, alpha=1, arrows=False, ax=ax)
    
    x_vals = [coord[0] for coord in pos.values()]
    y_vals = [coord[1] for coord in pos.values()]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_pad = (x_max - x_min) * padding_ratio
    y_pad = (y_max - y_min) * padding_ratio
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_title(title)
    ax.set_axis_on()  
    plt.tight_layout()
    plt.show()

def filter_nodes(graph: nx.DiGraph, prefixes: list[str]) -> list:
    """
    Filters nodes in a networkx graph based on whether their label attribute starts with "CS".
    """
    prefix_tuple = tuple(prefixes)
    return [(node, data) for node, data in graph.nodes(data=True) if str(node).startswith(prefix_tuple)]

def find_closest_node(
    original_graph: nx.DiGraph,
    start_node: str,
    target_nodes: set
) -> str:
    """Helper function to find the closest node in a target set."""
    try:
        all_paths = nx.single_source_dijkstra_path_length(original_graph, source=start_node)
        closest_target = None
        min_len = float('inf')
        for target, path_len in all_paths.items():
            if target in target_nodes:
                if path_len < min_len:
                    min_len = path_len
                    closest_target = target
        return closest_target
    except nx.NodeNotFound:
        return None

def reconnect_nodes(
    graph: nx.DiGraph,
    original_graph: nx.DiGraph,
    nodes: list[tuple]
) -> nx.DiGraph:
    """
    Adds filtered nodes and their edges to a regular (non-clustered) graph.
    """
    nodes_added_count = 0
    edges_added_count = 0
    nodes_in_current_graph = set(graph.nodes())

    for node_to_add, node_data in nodes:
        if node_to_add in nodes_in_current_graph:
            continue

        potential_edges = []
        
        for _, original_neighbor, edge_data in original_graph.edges(node_to_add, data=True):
            target_node = None
            
            # The neighbor exists directly in the graph
            if original_neighbor in nodes_in_current_graph:
                target_node = original_neighbor
            
            # Reroute to the closest available node
            else:
                closest_node = find_closest_node(
                    original_graph,
                    original_neighbor,
                    nodes_in_current_graph
                )
                if closest_node:
                    target_node = closest_node
            
            if target_node is not None:
                potential_edges.append((target_node, edge_data))

        if potential_edges:
            graph.add_node(node_to_add, **node_data)
            nodes_in_current_graph.add(node_to_add)
            nodes_added_count += 1

            for target, data in potential_edges:
                if not graph.has_edge(node_to_add, target):
                    graph.add_edge(node_to_add, target, **data)
                    edges_added_count += 1
    
    print(f"{nodes_added_count} nodes inserted.")
    print(f"{edges_added_count} edges inserted.")
    return graph

def find_disconnected_nodes(graph: nx.DiGraph) -> list:
    """
    Finds all nodes in a NetworkX graph that have a degree of 0.
    """
    return [node for node in graph.nodes() if graph.degree(node) == 0]

def find_largest_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Finds the largest connected component in a DiGraph.
    """
    connected_components = list(nx.weakly_connected_components(G))

    if not connected_components:
        print("Graph appears to be empty.")
        return nx.DiGraph()

    # Find the component with the most nodes
    largest_component_nodes = max(connected_components, key=len)

    # Create and return the subgraph of the largest component
    largest_component_subgraph = G.subgraph(largest_component_nodes).copy()
    
    return largest_component_subgraph

def convert_to_graph(G_di: nx.DiGraph) -> nx.Graph:
    """
    Converts a nx.DiGraph to nx.Graph, safely merging attributes from bi-directional edges to prevent data loss.
    """
    # Create a new undirected graph from the directed one
    G_un = G_di.to_undirected(as_view=False)
    
    # Find bi-directional edges in the original DiGraph
    for u, v in G_un.edges():
        if G_di.has_edge(u, v) and G_di.has_edge(v, u):
            # Both directions exist, so we merge attributes
            data_uv = G_di.edges[u, v]
            data_vu = G_di.edges[v, u]

            # For capacity take the maximum of the two
            merged_capacity = max(data_uv.get('capacity', 0.0), data_vu.get('capacity', 0.0))
            merged_L = data_uv.get('L', 0) + data_vu.get('L', 0)
            merged_DN = min(data_uv.get('DN', float('inf')), data_vu.get('DN', float('inf')))

            G_un.edges[u, v]['capacity'] = merged_capacity
            G_un.edges[u, v]['L'] = merged_L
            G_un.edges[u, v]['DN'] = merged_DN

    return G_un

def convert_to_digraph(G:nx.Graph) -> nx.DiGraph:
    """
    Convert nx.Graph to nx.DiGraph.
    """
    G_di = nx.DiGraph()
    G_di.add_nodes_from(G.nodes(data=True))

    # For every edge in the original data add forward and reverse edge
    for u, v, data in G.edges(data=True):
        G_di.add_edge(u, v, **data)
        # Add the reverse edge
        G_di.add_edge(v, u, **data.copy())

    return G_di

def estimate_gas_flow(p_bar: float, dn_mm: float, length_km: float, G:float=0.6, T:float=288) -> float:
    """
    Estimate gas flow in kg/s using a Panhandle A-like approximation.
    
    Parameters:
    - p_bar: max pressure (bar)
    - dn_mm: nominal diameter (mm)
    - length_km: pipeline length (km)
    - G: specific gravity (default 0.6 for natural gas)
    - T: temperature in Kelvin (default 288 K)
    
    Returns:
    - Estimated flow in a consistent unit (e.g., kg(/s)).
    """
    if length_km <= 0:
        length_km = 0.001 
    try:
        flow = 0.0035 * (p_bar ** 1.054) * (dn_mm ** 2.53) / (length_km ** 0.527 * G ** 0.473 * T ** 0.5)
        density_kg_nm3 = 0.8
        return (flow * density_kg_nm3) / 3600
    except (ValueError, ZeroDivisionError):
        return 0.0

def add_capacity(G:nx.DiGraph) -> None:
    """
    Add capacity values in kg/s to edges.
    """
    for u, v, data in G.edges(data=True):
            dn = data.get('DN', 0)
            pmax = data.get('Pmax', 0)
            length = data.get('L', 0)
            
            if dn == float('inf') or pmax == float('inf') or length == float('inf'):
                data['capacity'] = 0.0
                continue

            # Calculate capacity
            capacity = estimate_gas_flow(p_bar=pmax, dn_mm=dn, length_km=length)
            data['capacity'] = capacity

def add_norm_capacity(G:nx.DiGraph) -> None:
    """
    Add normalized capacity values to edges.
    """
    # Get all raw capacity values into a list
    capacities = [
        data['capacity'] for u, v, data in G.edges(data=True) if 'capacity' in data
    ]

    if capacities:
        # Find the global min and max capacity
        min_cap = min(capacities)
        max_cap = max(capacities)

        # Loop again to calculate and store the normalized capacity
        for u, v, data in G.edges(data=True):
            if 'capacity' in data:
                if max_cap > min_cap:
                    norm_val = 0.01 + 0.99 * (data['capacity'] - min_cap) / (max_cap - min_cap)
                else:
                    norm_val = 1.0
                data['norm_capacity'] = norm_val

def add_supply(G:nx.DiGraph, throughput:float=5000.0) -> None:
    """
    Add dummy supply/demand data to nodes. Define a realistic total system throughput (e.g., in kg/s).
    """
    TOTAL_SYSTEM_THROUGHPUT = throughput # kg/s

    # Calculate potential demand for all consumer nodes
    demand_nodes = {}
    total_demand_potential = 0
    for node, data in G.nodes(data=True):
        if str(node).startswith(("IND", "DSO")):
            
            all_neighbors = set(G.successors(node)) | set(G.predecessors(node))
            potential = 0
            for neighbor in all_neighbors:
                # Check for the edge in both directions safely
                edge_data = G.get_edge_data(node, neighbor) or G.get_edge_data(neighbor, node)
                
                if edge_data:
                    potential += edge_data.get('capacity', 0)
                    
            demand_nodes[node] = potential
            total_demand_potential += potential

    # Distribute the total throughput as demand
    if total_demand_potential > 0:
        for node, potential in demand_nodes.items():
            # Assign demand proportional to the node's potential
            demand = (potential / total_demand_potential) * TOTAL_SYSTEM_THROUGHPUT
            G.nodes[node]['supply'] = -demand

    # Calculate potential supply for all source nodes
    supply_nodes = {}
    total_supply_potential = 0
    for node, data in G.nodes(data=True):
        if str(node).startswith(("GPR", "LNG", "BIO", "IC", "GS")):
            
            # Get all neighbors by combining successors and predecessors
            all_neighbors = set(G.successors(node)) | set(G.predecessors(node))
            potential = 0
            for neighbor in all_neighbors:
                # Check for the edge in both directions safely
                edge_data = G.get_edge_data(node, neighbor) or G.get_edge_data(neighbor, node)
                
                if edge_data:
                    potential += edge_data.get('capacity', 0)

            supply_nodes[node] = potential
            total_supply_potential += potential

    # Distribute the total demand as supply to balance the system
    if total_supply_potential > 0:
        # The total supply must exactly match the total demand
        total_supply_needed = TOTAL_SYSTEM_THROUGHPUT
        for node, potential in supply_nodes.items():
            supply = (potential / total_supply_potential) * total_supply_needed
            G.nodes[node]['supply'] = supply

    # Set supply for all other (passive) nodes to zero
    for node, data in G.nodes(data=True):
        if 'supply' not in data:
            data['supply'] = 0.0