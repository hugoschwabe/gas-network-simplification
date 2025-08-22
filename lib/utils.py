from collections import defaultdict
import inspect
import geopandas as gpd
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx
import copy
from pyvis.network import Network
import numpy as np
from matplotlib.lines import Line2D
import json

def nuts3(path:str="./data//NUTS3/NUTS_RG_20M_2024_4326.shp"):
    nuts3 = gpd.read_file(path).set_index("NUTS_ID")
    nuts3 = nuts3[nuts3["LEVL_CODE"] == 3]
    nuts3 = nuts3.to_crs("EPSG:4326")
    nuts3 = nuts3.reset_index()
    return nuts3

def importance_weights(path:str="data/detailed_property_weights.csv"):
    importance = pd.read_csv(path)
    importance = importance.set_index("node_name")
    importance["impact_pct"] = importance["impact_pct"] / importance["impact_pct"].max()
    importance = importance["impact_pct"].sort_values().to_dict()
    return importance

def _sanitize_for_gml(data):
    """
    Sanitizes an attribute value for GML compatibility.
    """
    if isinstance(data, dict):
        # Heuristic: Check if the values are also dictionaries (the problematic structure)
        if data and isinstance(next(iter(data.values())), dict):
            # This is the dict-of-dicts we need to restructure
            restructured_list = []
            for key, value_dict in data.items():
                new_entry = {'original_name': str(key)} # The invalid key becomes a value
                # Make sure the inner dictionary is also sanitized
                new_entry.update(_sanitize_for_gml(value_dict))
                restructured_list.append(new_entry)
            return restructured_list
        else:
            # This is a normal dictionary, just sanitize its values
            return {key: _sanitize_for_gml(value) for key, value in data.items()}
    
    # If it's a list, we need to sanitize each of its items
    elif isinstance(data, list):
        return [_sanitize_for_gml(item) for item in data]
    
    # Otherwise (string, number, etc.), it's safe
    else:
        return data


def write_gml(G: nx.Graph, path: str) -> None:
    """
    Writes a graph to GML, sanitizing incompatible attribute structures first.
    """
    G_copy = copy.deepcopy(G)

    # Sanitize node attributes
    for node, data in G_copy.nodes(data=True):
        data.pop("supply", None)
        for attr_key, attr_value in data.items():
            data[attr_key] = _sanitize_for_gml(attr_value)

    # Sanitize edge attributes
    for u, v, data in G_copy.edges(data=True):
        data.pop("capacity", None)
        data.pop("norm_capacity", None)
        for attr_key, attr_value in data.items():
            data[attr_key] = _sanitize_for_gml(attr_value)

    # Write to GML
    nx.write_gml(G_copy, path)
    print(f"Graph successfully saved to {path} after restructuring attributes.")

def graph_node_names(G:nx.Graph) -> list[str]:
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

def graph_to_nodes_df(G:nx.Graph) -> pd.DataFrame:
    """
    Converts graph's nodes into a dataframe while preserving all attributes.
    """
    df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index').reset_index().rename(columns={"index": "nodes"})
    df["x"] = df["coord"].apply(lambda xy: xy[0])
    df["y"] = df["coord"].apply(lambda xy: xy[1])
    return df

def graph_to_edges_df(G:nx.Graph) -> pd.DataFrame:
    """
    Converts a NetworkX graph's edge data into a pandas DataFrame, sorted by the capacity attribute in descending order.
    """
    # Create a list of dictionaries, where each dictionary represents an edge
    edge_list = []
    for u, v, data in G.edges(data=True):
        edge_info = {'source': u, 'target': v}
        edge_info.update(data)  # Add all other edge attributes
        edge_list.append(edge_info)

    # Convert the list to a DataFrame
    df = pd.DataFrame(edge_list)

    # Sort the DataFrame by the capacity column if it exists
    if 'capacity' in df.columns:
        df = df.sort_values(by='capacity', ascending=False).reset_index(drop=True)

    return df#[df["L"] > 1]

def build_clustered_graph(
    G:nx.Graph,
    communities: list[frozenset],
    keep_nodes:list[str]
) -> nx.Graph:
    """
    Builds a robust simplified nx.Graph from community detection results.
    """
    simplified_graph = nx.Graph()
    
    # Map each non-critical node to its new cluster ID
    node_to_cluster_map = {
        node: f"C_{i}" for i, community in enumerate(communities) for node in community
    }

    # Create the cluster nodes ("super-nodes")
    for i, community in enumerate(communities):
        cluster_name = f"C_{i}"
        # Aggregate properties for the nodes within this cluster
        community_nodes_data = {node: G.nodes[node] for node in community}
        coords = [d.get('coord', (None, None)) for d in community_nodes_data.values()]
        valid_coords = [c for c in coords if c[0] is not None]
        
        # Calculate mean coordinates for the cluster
        avg_x = sum(c[0] for c in valid_coords) / len(valid_coords) if valid_coords else None
        avg_y = sum(c[1] for c in valid_coords) / len(valid_coords) if valid_coords else None
        
        simplified_graph.add_node(
            cluster_name,
            coord=(avg_x, avg_y),
            original_nodes=list(community),
            supply=sum(d.get('supply', 0) for d in community_nodes_data.values())
        )
    
    # Add the protected critical nodes back into the graph
    critical_nodes = {
        node for node in G.nodes() 
        if any(str(node).lower().startswith(p.lower()) for p in keep_nodes)
    }
    for node in critical_nodes:
        simplified_graph.add_node(node, **G.nodes[node])

    # Create edges between clusters and connect critical nodes
    edges_to_aggregate = defaultdict(list)
    
    for u, v, data in G.edges(data=True):
        u_is_crit = u in critical_nodes
        v_is_crit = v in critical_nodes
        
        # Determine the endpoint names in the new graph
        u_new = u if u_is_crit else node_to_cluster_map.get(u)
        v_new = v if v_is_crit else node_to_cluster_map.get(v)

        if u_new is None or v_new is None or u_new == v_new:
            continue
            
        # Group edges between the same two new nodes for aggregation
        edge_tuple = tuple(sorted((u_new, v_new)))
        edges_to_aggregate[edge_tuple].append(data)

    # Aggregate the edge properties and add to the graph
    for (u_new, v_new), edge_data_list in edges_to_aggregate.items():
        total_capacity = sum(d.get('capacity', 0) for d in edge_data_list)
        if total_capacity <= 0: continue
            
        avg_len = sum(d.get('L', 0) * d.get('capacity', 0) for d in edge_data_list) / total_capacity
        sum_dn_powered = sum(d.get('DN', 0) ** 2.5 for d in edge_data_list)
        equiv_dn = (sum_dn_powered ** (1/2.5)) if sum_dn_powered > 0 else 0
        avg_pmax = sum(d.get('Pmax', 0) * d.get('capacity', 0) for d in edge_data_list) / total_capacity

        simplified_graph.add_edge(
            u_new, v_new,
            capacity=total_capacity, L=avg_len, DN=equiv_dn, Pmax=avg_pmax,
            edge_type='aggregated_pipe'
        )
    return simplified_graph

def run_algo(func, original:nx.Graph, **kwargs) -> nx.Graph:
    """
    Run a NetworkX algorithm and return the results.
    """
    try:
        sig = inspect.signature(func)
        valid_arg_names = {p.name for p in sig.parameters.values()}

        filtered_kwargs = {
            key: value for key, value in kwargs.items() 
            if key in valid_arg_names
        }

        _, graph = func(G=original, **filtered_kwargs)
        return graph
    except Exception as e:
        print(e)
        return

def plot_network(
        G:nx.Graph, 
        gdf:gpd.GeoDataFrame=None, 
        clusters:list=None, 
        title:str="Title", 
        padding_ratio:float=0.15, 
        nodes:bool=True, 
        edges:bool=True
    ) -> None:
    """
    Plot any NetworkX graph with optional NUTS regions background from a GeoDataFrame.
    """
    fig, ax = plt.subplots(figsize=(16, 20))
    if gdf is not None:
        gdf.plot(ax=ax, color='lightgrey', edgecolor='white', alpha=0.5, linewidth=2, zorder=0)

    try:
        pos = {node: (data["coord"][0], data["coord"][1]) for node, data in G.nodes(data=True)}
    except KeyError:
        pos = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)}
    """
    if nodes:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'lime', 'pink', 'teal', 'gold', 'navy', 'brown', 'olive']
        if clusters is None:
            clusters = [list(G.nodes())]
        
        for i, res in enumerate(clusters):
            nx.draw_networkx_nodes(G, pos, nodelist=res, node_size=node_size, node_color=colors[i % len(colors)], ax=ax)
    """
    cs_nodes = [node for node in G.nodes() if str(node).lower().startswith('cs')]
    cv_nodes = [node for node in G.nodes() if str(node).lower().startswith('cv')]
    other_nodes = [
        node for node in G.nodes() 
        if not str(node).lower().startswith(('cs', 'cv'))
    ]

    # Draw the edges and labels for all nodes
    edge_capacities = [G[u][v]['capacity'] for u, v in G.edges()]
    edge_capacities = [1+(x/200) if 1+(x/200)<5 else 5 for x in edge_capacities]
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, edge_color='gray', width=edge_capacities)
    #nx.draw_networkx_labels(G, pos, ax=ax)

    # Draw each category of nodes with a different color
    nx.draw_networkx_nodes(
        G, pos, nodelist=other_nodes, node_color='black', node_size=2*max(edge_capacities), ax=ax, label='Junctions/Other'
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=cs_nodes, node_color='red', node_size=6*max(edge_capacities), ax=ax, label='Compressor Stations'
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=cv_nodes, node_color='blue', node_size=6*max(edge_capacities), ax=ax, label='Control Valves'
    )

    
    x_vals = [coord[0] for coord in pos.values()]
    y_vals = [coord[1] for coord in pos.values()]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_pad = (x_max - x_min) * padding_ratio
    y_pad = (y_max - y_min) * padding_ratio
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_title(title, fontdict={'fontsize':20})
    ax.set_axis_on()  
    ax.set
    plt.tight_layout()
    plt.show()

def plot_networks_side_by_side(
    original_graph: nx.Graph,
    simplified_graph: nx.Graph,
    gdf: gpd.GeoDataFrame = None,
    original_title: str = "Original Network",
    simplified_title: str = "Simplified Network",
    padding_ratio: float = 0.15
) -> None:
    """
    Plots two NetworkX graphs side-by-side with a single shared legend.
    
    Args:
        original_graph: The original NetworkX graph.
        simplified_graph: The simplified NetworkX graph.
        gdf: Optional GeoDataFrame for plotting a background map.
        original_title: The title for the original graph's subplot.
        simplified_title: The title for the simplified graph's subplot.
        padding_ratio: The ratio of padding to add around the network plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(24, 16))

    graphs = {
        original_title: original_graph,
        simplified_title: simplified_graph
    }

    # Determine common bounds for both plots for consistent scale and position
    x_vals_all, y_vals_all = [], []
    for g in graphs.values():
        try:
            pos = {node: (data["coord"][0], data["coord"][1]) for node, data in g.nodes(data=True)}
        except KeyError:
            pos = {node: (data["x"], data["y"]) for node, data in g.nodes(data=True)}
        
        if pos:
            x_vals_all.extend([coord[0] for coord in pos.values()])
            y_vals_all.extend([coord[1] for coord in pos.values()])

    if not x_vals_all or not y_vals_all:
        print("Cannot plot graphs with no node coordinates.")
        return
        
    x_min, x_max = min(x_vals_all), max(x_vals_all)
    y_min, y_max = min(y_vals_all), max(y_vals_all)
    x_pad = (x_max - x_min) * padding_ratio
    y_pad = (y_max - y_min) * padding_ratio
    xlims = (x_min - x_pad, x_max + x_pad)
    ylims = (y_min - y_pad, y_max + y_pad)

    # Plot each graph on its respective subplot
    for ax, (title, G) in zip(axes, graphs.items()):
        if gdf is not None:
            gdf.plot(ax=ax, color='lightgrey', edgecolor='white', alpha=0.5, linewidth=2, zorder=0)

        try:
            pos = {node: (data["coord"][0], data["coord"][1]) for node, data in G.nodes(data=True)}
        except KeyError:
            pos = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)}
        
        if not pos:
            ax.set_title(f"{title}", fontdict={'fontsize': 16})
            continue

        # Categorize nodes
        cs_nodes = [node for node in G.nodes() if str(node).lower().startswith('cs')]
        cv_nodes = [node for node in G.nodes() if str(node).lower().startswith('cv')]
        other_nodes = [
            node for node in G.nodes()
            if not str(node).lower().startswith(('cs', 'cv'))
        ]
        
        # Draw edges
        edge_widths = [1]
        if G.edges():
            edge_capacities = [G[u][v].get('capacity', 1) for u, v in G.edges()]
            edge_widths = [1 + (x / 200) if 1 + (x / 200) < 5 else 5 for x in edge_capacities]
            nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, edge_color='gray', width=edge_widths)
        
        # Draw nodes
        node_size_base = 1.5*max(edge_widths)
        nx.draw_networkx_nodes(
            G, pos, nodelist=other_nodes, node_color='black', node_size=node_size_base, ax=ax
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=cs_nodes, node_color='red', node_size=3 * node_size_base, ax=ax
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=cv_nodes, node_color='blue', node_size=3 * node_size_base, ax=ax
        )

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(title, fontdict={'fontsize': 20})
        ax.set_axis_on()
        # Remove the x and y tick labels
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                         labelbottom=False, labelleft=False)

    # Create a single, shared legend for the entire figure
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Compressor Station', markerfacecolor='red', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Control Valve', markerfacecolor='blue', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Other Junction', markerfacecolor='black', markersize=10)
    ]
    # Place legend at the bottom center of the figure
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize='x-large', frameon=False)
    
    # Add text explaining edge width below the legend
    fig.text(0.5, 0.05, 'Edge width is proportional to pipeline capacity', ha='center', va='center', fontsize=14)

    # Adjust layout to make space for legend and text
    fig.subplots_adjust(bottom=0.001)
    plt.show()


def plot_network_pyvis(
    G: nx.Graph, 
    title: str = "Interactive Gas Network",
    filename: str = "interactive_network.html"
) -> None:
    try:
        pos = {node: data["coord"] for node, data in G.nodes(data=True)}
    except KeyError:
        pos = {node: (data.get("x", 0), data.get("y", 0)) for node, data in G.nodes(data=True)}
    valid_pos = {
        node: (x, y) for node, (x, y) in pos.items() 
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)
    }
    scaled_pos = {}
    if valid_pos:
        pos_df = pd.DataFrame.from_dict(valid_pos, orient='index', columns=['x', 'y'])
        x_min, x_max, y_min, y_max = pos_df['x'].min(), pos_df['x'].max(), pos_df['y'].min(), pos_df['y'].max()
        canvas_width, canvas_height = 1500, 1000
        for node, (x, y) in valid_pos.items():
            scaled_x = canvas_width * (x - x_min) / (x_max - x_min) if x_max > x_min else canvas_width / 2
            scaled_y = canvas_height * (y - y_min) / (y_max - y_min) if y_max > y_min else canvas_height / 2
            scaled_pos[node] = (scaled_x, -scaled_y)
            
    net = Network(height="900px", width="100%", heading=title, directed=False, notebook=True, cdn_resources='remote')
    options = {
        "groups": {
            "compressors": {
                "color": "red",
                "shape": "square"
            },
            "valves": {
                "color": "blue",
                "shape": "triangle"
            },
            "junctions": {
                "color": "black",
                "shape": "dot"
            }
        },
        "nodes": {
            "font": {"size": 0},
        },
        "edges": {
            "smooth": {"enabled": False},
        },
        "physics": {
            "enabled": False
        },
        "interaction": {
            "dragNodes": False,
            "dragView": True,
            "zoomView": True
        }
    }
    net.set_options(json.dumps(options))

    # Add Nodes by assigning them to a group
    for node, data in G.nodes(data=True):
        node_name = str(node)
        node_position = scaled_pos.get(node)
        
        # Determine the group for the node
        if node_name.lower().startswith('cs'):
            group = 'compressors'
        elif node_name.lower().startswith('cv'):
            group = 'valves'
        else:
            group = 'junctions'
        
        hover_title = f"Node: {node_name}<br>Data: {data}"
        
        if node_position:
            net.add_node(
                node, 
                x=node_position[0], y=node_position[1],
                group=group,
                physics=False,
                title=hover_title,
                size=1
            )
        else:
            net.add_node(
                node, 
                group=group,
                physics=True,
                title=hover_title,
                size=1
            )

    # Add Edges
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, color='gray', title=f"Data: {data}")
        
    net.show(filename)
    print(f"Interactive graph saved to '{filename}'")

def filter_nodes(G:nx.Graph, prefixes:list[str]) -> list:
    """
    Filters nodes in a networkx graph based on whether their label attribute starts with "CS".
    """
    prefix_tuple = tuple(prefixes)
    return [(node, data) for node, data in G.nodes(data=True) if str(node).startswith(prefix_tuple)]

def find_closest_node(
    G:nx.Graph,
    start_node:str,
    target_nodes:set
) -> str:
    """Helper function to find the closest node in a target set."""
    try:
        all_paths = nx.single_source_dijkstra_path_length(G, source=start_node)
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
    G:nx.Graph,
    simplified:nx.Graph,
    nodes:list[tuple]
) -> nx.Graph:
    """
    Adds filtered nodes and their edges to a regular (non-clustered) graph.
    """
    nodes_added_count = 0
    edges_added_count = 0
    nodes_in_current_graph = set(simplified.nodes())

    for node_to_add, node_data in nodes:
        if node_to_add in nodes_in_current_graph:
            continue

        potential_edges = []
        
        for _, original_neighbor, edge_data in G.edges(node_to_add, data=True):
            target_node = None
            
            # The neighbor exists directly in the graph
            if original_neighbor in nodes_in_current_graph:
                target_node = original_neighbor
            
            # Reroute to the closest available node
            else:
                closest_node = find_closest_node(
                    G,
                    original_neighbor,
                    nodes_in_current_graph
                )
                if closest_node:
                    target_node = closest_node
            
            if target_node is not None:
                potential_edges.append((target_node, edge_data))

        if potential_edges:
            simplified.add_node(node_to_add, **node_data)
            nodes_in_current_graph.add(node_to_add)
            nodes_added_count += 1

            for target, data in potential_edges:
                if not simplified.has_edge(node_to_add, target):
                    simplified.add_edge(node_to_add, target, **data)
                    edges_added_count += 1
    
    print(f"{nodes_added_count} nodes inserted.")
    print(f"{edges_added_count} edges inserted.")

    print("Reconnecting edges for critical components (CS/CV)...")
    component_edges_added = 0
    
    # Identify all component nodes in the original graph for quick lookup
    component_nodes = {
        node for node, data in G.nodes(data=True)
        if data.get("node_type") in ["compressor station", "control valve"]
    }

    # Iterate through all edges in the original graph
    for u, v, original_edge_data in G.edges(data=True):
        # Check if this edge is connected to a component
        if u in component_nodes or v in component_nodes:
            # Check if both endpoints of this critical edge exist in the current simplified graph
            if u in nodes_in_current_graph and v in nodes_in_current_graph:
                # If the edge does not already exist, add it back
                if not simplified.has_edge(u, v):
                    simplified.add_edge(u, v, **original_edge_data)
                    component_edges_added += 1

    print(f"{component_edges_added} critical component edges re-inserted.\n")
    return simplified

def find_largest_subgraph(G:nx.Graph) -> nx.Graph:
    """
    Finds the largest connected component in a Graph.
    """
    connected_components = list(nx.connected_components(G))

    if not connected_components:
        print("Graph appears to be empty.")
        return nx.Graph()

    # Find the component with the most nodes
    largest_component_nodes = max(connected_components, key=len)

    # Create and return the subgraph of the largest component
    largest_component_subgraph = G.subgraph(largest_component_nodes).copy()
    
    return largest_component_subgraph

def estimate_gas_flow(
        p_bar:float, 
        dn_mm:float, 
        length_km:float, 
        G:float=0.6, 
        T_k:float=288.15, 
        k_p:float=0.2
    ) -> float:
    """
    Estimate gas flow in kg/s using a Panhandle B-like approximation.
    
    Parameters:
    - p_bar: max pressure (bar)
    - dn_mm: nominal diameter (mm)
    - length_km: pipeline length (km)
    - G: specific gravity (default 0.6 for natural gas)
    - T_k: temperature in Kelvin (default 288.15 K)
    - k_p: pressure coefficient (default drop of 0.2 bar/km)
    
    Returns:
    - Estimated flow in a consistent unit (e.g. kg/s).
    """
    if length_km <= 0:
        length_km = 0.001 
    try:

        p2_bar = max(p_bar - (k_p * length_km), 0.01)
        #p2_bar = 0.01
        
        delta_p_pa = ((p_bar * 100000) ** 2) - ((p2_bar * 100000) ** 2)
        dn_m = dn_mm / 1000
        length_m = length_km * 1000

        flow = 0.3755 * (delta_p_pa ** 0.51) * (dn_m ** 2.53) / (length_m ** 0.51 * G ** 0.49 * T_k ** 0.51)

        density_sc_kgm3 = 0.77 # Approx. density of natural gas (G=0.6) at standard conditions
        return flow * density_sc_kgm3
    except (ValueError, ZeroDivisionError):
        return 0.0

def add_capacity(G:nx.Graph, k_p:float=0.2) -> None:
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
            capacity = estimate_gas_flow(p_bar=pmax, dn_mm=dn, length_km=length, k_p=k_p)
            data['capacity'] = capacity

def add_norm_capacity(G:nx.Graph) -> None:
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

def add_dummy_supply(G:nx.Graph, throughput:float=5000.0) -> None:
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

def add_supply_from_csv(
        G: nx.Graph, 
        csv_path:str, 
        verbose:bool = False
    ) -> None:
    """
    Reads real supply/demand data from a semicolon-delimited CSV file, calculates the supply in kg/s and adds it as a supply attribute to the graph's nodes.
    """
    try:
        df = pd.read_csv(csv_path, sep=';')
    except FileNotFoundError:
        print(f"ERROR: The file was not found at {csv_path}")
        return

    # Check for required columns using their exact names
    required_cols = ['node', 'pressure[barg]', 'flow[MW]', 'CV[kWh/m3]', 'R[1]']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: CSV must contain the exact columns: {required_cols}")
        return

    # Initialize supply attribute for all nodes to zero
    nx.set_node_attributes(G, 0.0, name='supply')

    # Iterate through CSV and update graph nodes
    nodes_updated = 0
    nodes_not_found = []
    
    DENSITY_AIR_SC = 1.225  # kg/m³

    for index, row in df.iterrows():
        node_name = row['node']
        
        if G.has_node(node_name):
            energy_flow_mw = float(row['flow[MW]'].replace(',', '.'))
            calorific_value_kwh_m3 = float(row['CV[kWh/m3]'].replace(',', '.'))
            relative_density = float(row['R[1]'].replace(',', '.'))

            if float(calorific_value_kwh_m3) > 0:
                vol_flow_m3_hr = (energy_flow_mw * 1000) / calorific_value_kwh_m3
                vol_flow_m3_s = vol_flow_m3_hr / 3600
                gas_density_sc = relative_density * DENSITY_AIR_SC
                mass_flow_kgs = vol_flow_m3_s * gas_density_sc
            else:
                mass_flow_kgs = 0.0
            
            # Assuming a positive flow in the data represents demand (a negative supply)
            G.nodes[node_name]['supply'] = -mass_flow_kgs
            nodes_updated += 1
        else:
            nodes_not_found.append(node_name)
            
    if verbose:
        print(f"Updated 'supply' attribute for {nodes_updated} nodes based on CSV data.")
        if nodes_not_found:
            print(f"WARNING: The following {len(nodes_not_found)} nodes from the CSV were not found in the graph and were skipped:")
            print(nodes_not_found[:10])