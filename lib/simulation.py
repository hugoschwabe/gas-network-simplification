import pandapipes as pp
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.flow import edmonds_karp
from typing import Callable

def simulate_network(G: nx.Graph) -> pp.pandapipesNet:
    """
    Builds and runs a pandapipes simulation from a Graph with active components (compressor stations, control valves).
    """

    print("Building pandapipes network...")
    net = pp.create_empty_network(fluid="methane")
    nx_to_pp_junctions = {}

    print("Creating network junctions...")
    for node, data in G.nodes(data=True):
        junction_idx = pp.create_junction(net, pn_bar=1.0, tfluid_k=283.15, name=str(node))
        nx_to_pp_junctions[node] = junction_idx

    # Handle Sources, Sinks, and the External Grid (Pressure Reference)
    print("Creating sources, sinks, and pressure reference...")
    reference_node_found = False
    for node, data in G.nodes(data=True):
        # The first Interconnector found is the pressure reference
        if str(node).startswith("IC") and not reference_node_found:
            pp.create_ext_grid(net, junction=nx_to_pp_junctions[node], p_bar=60.0, t_k=283.15)
            print(f"Selected Interconnector '{node}' as the main pressure reference.")
            reference_node_found = True
            # Skip creating a source/sink for the reference node
            continue

        # Create sources and sinks based on the supply attribute
        supply_value = data.get('supply', 0)
        if supply_value > 0:
            pp.create_source(net, junction=nx_to_pp_junctions[node], mdot_kg_per_s=supply_value)
        elif supply_value < 0:
            pp.create_sink(net, junction=nx_to_pp_junctions[node], mdot_kg_per_s=abs(supply_value))
    
    if not reference_node_found:
        print("ERROR: No Interconnector (IC) node found. Cannot set a pressure reference.")
        return None

    print("Creating network components (pipes, compressors, valves)...")
    vector = calculate_flow_vector(G)
    for u, v, data in G.edges(data=True):
        edge_type = data.get("edge_type", "pipe")

        from_node, to_node = get_compressor_direction(G, (u, v), vector)
        junc_from = nx_to_pp_junctions[from_node]
        junc_to = nx_to_pp_junctions[to_node]

        if edge_type == "compressor station":
            ratio = data.get('pressure_ratio', 1.2)
            pp.create_compressor(net, from_junction=junc_from, to_junction=junc_to,
                                pressure_ratio=ratio, name=f"CS_{u}-{v}")
        elif edge_type == "control valve":
            diam_m = data.get('DN', 100) / 1000
            pp.create_valve(net, from_junction=junc_from, to_junction=junc_to,
                            diameter_m=diam_m, opened=True, name=f"CV_{u}-{v}")
        else: # This is a standard pipe
            pp.create_pipe_from_parameters(net, from_junction=junc_from,
                                           to_junction=junc_to,
                                           length_km=data.get('L', 0.0),
                                           diameter_m=data.get('DN', 0.0) / 1000,
                                           name=f"Pipe_{u}-{v}")

    # Final checks and simulation run
    zero_diameter_pipes = net.pipe[net.pipe.diameter_m <= 0]
    if not zero_diameter_pipes.empty:
        print(f"WARNING: Found {len(zero_diameter_pipes)} pipes with zero or negative diameter. Skipping.")
        return None

    print("\nRunning simulation...")
    try:
        pp.pipeflow(net, stop_condition="tol", iter=30, tol_p=0.0, tol_v=0.0)
        print("Simulation successful!")
        return net
    except Exception as e:
        print(f"Simulation failed: {e}")
        return None
    

def simulate_clustered_network(simplified_graph: nx.Graph) -> pp.pandapipesNet:
    """
    Builds and runs a pandapipes simulation for a potentially clustered graph.
    """
    print("Building pandapipes network with aggregation...")
    net = pp.create_empty_network(fluid="methane")
    nx_to_pp_junctions = {}
    
    # Create a simple junction for each super-node
    for simp_node in simplified_graph.nodes():
        junction_idx = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15, name=f"Junction {simp_node}")
        nx_to_pp_junctions[simp_node] = junction_idx

    # Get the mapping from virtual nodes to original nodes and their data
    virtual_node_map = nx.get_node_attributes(simplified_graph, 'original_nodes')
    original_node_data = nx.get_node_attributes(simplified_graph, 'original_node_data')

    # Aggregation logic
    for simp_node, junction_idx in nx_to_pp_junctions.items():
        original_nodes_in_cluster = virtual_node_map.get(simp_node, [simp_node])
        
        net_supply_kg_per_s = 0
        for orig_node in original_nodes_in_cluster:
            data = original_node_data.get(orig_node, {})
            net_supply_kg_per_s += data.get('supply', 0)

        # Create a single source or sink with the NET aggregated flow
        if net_supply_kg_per_s > 0:
            pp.create_source(net, junction=junction_idx, mdot_kg_per_s=net_supply_kg_per_s)
        elif net_supply_kg_per_s < 0:
            pp.create_sink(net, junction=junction_idx, mdot_kg_per_s=abs(net_supply_kg_per_s))

    # Add pipes between the super-node junctions
    for u, v, data in simplified_graph.edges(data=True):
        pp.create_pipe_from_parameters(net, from_junction=nx_to_pp_junctions[u], to_junction=nx_to_pp_junctions[v], 
                                       length_km=data.get('L', 0) / 1000, diameter_m=data.get('DN', 0) / 100)

    return net

def prepare_graph_for_max_flow(g: nx.Graph) -> nx.Graph:
    """
    Takes any network graph with 'supply' attributes on its nodes and adds
    super_source/super_sink nodes for max_flow analysis.
    """
    flow_network = g.copy()
    super_source, super_sink = "super_source", "super_sink"
    flow_network.add_nodes_from([super_source, super_sink])

    for node, data in g.nodes(data=True):
        supply = data.get('supply', 0.0)
        if supply > 0:
            flow_network.add_edge(super_source, node, capacity=supply)
        elif supply < 0:
            flow_network.add_edge(node, super_sink, capacity=abs(supply))
            
    return flow_network

def calculate_max_deliverability(
        G:nx.Graph, 
        sources:str, 
        sinks:str, 
        capacity:str='capacity', 
        flow_func:Callable=edmonds_karp
    ) -> float:
    """
    Calculates the deliverability of a network from sources to sinks.
    """
    # Prepare graph by adding super-source/sinks
    G_flow = prepare_graph_for_max_flow(G)
    flow_value, _ = nx.maximum_flow(G_flow, sources, sinks, capacity=capacity, flow_func=flow_func)
    return flow_value

def calculate_deliverability_error(G_original: nx.Graph, G_simplified: nx.Graph) -> float:
    """
    Calculates the physical error score between an original and a simplified networkx graph.
    """
    f_orig = calculate_max_deliverability(
        G_original, 
        sources="super_source", 
        sinks="super_sink", 
        capacity="capacity",
        flow_func=edmonds_karp
    )
    f_simp = calculate_max_deliverability(
        G_simplified, 
        sources="super_source", 
        sinks="super_sink", 
        capacity="capacity",
        flow_func=edmonds_karp
    )

    print(f"Original Deliverability: {f_orig} kg/s")
    print(f"Simplified Deliverability: {f_simp} kg/s")

    if f_orig == 0:
        return 0.0 if f_simp == 0 else float('inf')

    error = abs(f_orig - f_simp) / f_orig
    if error > 1: return 1

    return error

def calculate_flow_vector(G: nx.Graph) -> np.ndarray:
    """
    Calculates the dominant flow vector for a network based on the geographic centers of supply and demand.
    """
    node_data = [
        {'node': node, 'x': data['coord'][0], 'y': data['coord'][1], 'supply': data['supply']}
        for node, data in G.nodes(data=True)
        if 'coord' in data and 'supply' in data
    ]
    
    if not node_data:
        print("WARNING: Graph has no nodes with 'x', 'y', and 'supply' attributes.")
        return None

    df = pd.DataFrame(node_data)
    
    supply_points = df[df['supply'] > 0]
    demand_points = df[df['supply'] < 0]

    if supply_points.empty or demand_points.empty:
        print("WARNING: Cannot determine flow vector (no supplies or demands).")
        return None
        
    demand_points = demand_points.copy()
    demand_points['supply'] = demand_points['supply'].abs()

    supply_center_x = (supply_points['x'] * supply_points['supply']).sum() / supply_points['supply'].sum()
    supply_center_y = (supply_points['y'] * supply_points['supply']).sum() / supply_points['supply'].sum()

    demand_center_x = (demand_points['x'] * demand_points['supply']).sum() / demand_points['supply'].sum()
    demand_center_y = (demand_points['y'] * demand_points['supply']).sum() / demand_points['supply'].sum()

    return np.array([demand_center_x - supply_center_x, demand_center_y - supply_center_y])

def get_compressor_direction(
    G: nx.Graph, 
    compressor_edge: tuple[str, str], 
    flow_vector: np.ndarray
) -> tuple[str, str]:
    """
    Orients a single compressor edge to align with a pre-calculated dominant flow vector.
    """
    # Get the two connection points directly from the edge tuple
    pt1, pt2 = compressor_edge
    
    # Check if both nodes have the required coord attribute
    if not all('coord' in G.nodes[p] for p in [pt1, pt2]):
        print(f"WARNING: Nodes for edge {compressor_edge} are missing the coord attribute. Returning default direction.")
        return pt1, pt2 # Return default if coordinates are missing

    # Correctly extract coordinates into 1D numpy arrays
    coords1 = np.array(G.nodes[pt1]['coord'])
    coords2 = np.array(G.nodes[pt2]['coord'])
    
    # Calculate the compressor's own vector and the dot product for alignment
    compressor_vector = coords2 - coords1
    dot_product = np.dot(flow_vector, compressor_vector)
    
    return (pt1, pt2) if dot_product > 0 else (pt2, pt1)

def prepare_cs_for_simulation(G: nx.Graph) -> nx.Graph:
    """
    Prepares an undirected graph by replacing compressor edges with a detailed inlet/outlet node structure.
    """
    G_mod = G.copy()
    
    # Find all compressor edges first, as the graph will be modified
    compressor_edges = [
        (u, v, data) for u, v, data in G.edges(data=True)
        if data.get("edge_type") == "compressor station"
    ]

    print(f"Found {len(compressor_edges)} compressor edges to remodel.")

    for u, v, cs_data in compressor_edges:
        # Define names for the new inlet and outlet nodes
        u_sorted, v_sorted = sorted((u, v))
        inlet_node_name = f"{u_sorted}-{v_sorted}_virtual_1"
        outlet_node_name = f"{u_sorted}-{v_sorted}vitrual_2"
        
        # New nodes can inherit data (like coordinates)
        inlet_node_data = G_mod.nodes.get(u, {})
        outlet_node_data = G_mod.nodes.get(v, {})

        # Define properties for the new short connector pipes
        connector_pipe_data = {
            "edge_type": "pipe",
            "L": 0.00001,  # Minimal length in km
            "DN": cs_data.get("DN"),
            "Pmax": cs_data.get("Pmax")
        }

        # Remove the original compressor edge
        if G_mod.has_edge(u, v):
            G_mod.remove_edge(u, v)
        
        # Add the new nodes
        G_mod.add_node(inlet_node_name, **inlet_node_data)
        G_mod.add_node(outlet_node_name, **outlet_node_data)
        
        # Add the new 3-edge undirected chain
        G_mod.add_edge(u, inlet_node_name, **connector_pipe_data)
        G_mod.add_edge(inlet_node_name, outlet_node_name, **cs_data)
        G_mod.add_edge(outlet_node_name, v, **connector_pipe_data)

    return G_mod

def calculate_total_flow(net: pp.pandapipesNet) -> float:
    """
    Calculates the total mass flow within a network from pandapipes simulation results by summing the absolute mass flow rates in all pipes.
    """
    # Check if the simulation results exist
    if 'res_pipe' not in net or net.res_pipe.empty:
        print("WARNING: Simulation results (net.res_pipe) not found. Returning 0.0")
        return 0.0

    # The 'mdot_from_kg_per_s' column contains the mass flow for each pipe.
    total_flow = net.res_pipe['mdot_from_kg_per_s'].abs().sum()
    
    return total_flow

def check_supply_balance(G: nx.Graph) -> dict[str, float]:
    """
    Calculates the total supply (sources) and total demand (sinks) from the supply attribute of the nodes in a graph.
    """
    total_supply = 0.0
    total_demand = 0.0

    # Iterate through all nodes and sum up supply/demand
    for node, data in G.nodes(data=True):
        supply_value = data.get('supply', 0.0)
        if supply_value > 0:
            total_supply += supply_value
        elif supply_value < 0:
            total_demand += abs(supply_value)

    balance = total_supply - total_demand

    print("--- Supply/Demand Balance Check ---")
    print(f"Total Supply (Sources): {total_supply:,.2f} kg/s")
    print(f"Total Demand (Sinks):   {total_demand:,.2f} kg/s")
    print(f"Balance (Supply - Demand): {balance:,.2f} kg/s\n")
    
    return {
        "total_supply": total_supply,
        "total_demand": total_demand,
        "balance": balance
    }