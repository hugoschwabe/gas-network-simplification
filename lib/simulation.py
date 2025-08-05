import pandapipes as pp
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
from typing import Callable

def simulate_network(G: nx.DiGraph) -> pp.pandapipesNet:
    """
    Builds and runs a pandapipes simulation on a DiGraph.
    """
    if not nx.is_weakly_connected(G):
        print("Network has disconnected parts. Skipping simulation.")
        return None

    print("Building pandapipes network...")
    net = pp.create_empty_network(fluid="methane")
    nx_to_pp_junctions = {}

    # Create junctions only for non-CS/CV nodes
    print("Creating network junctions...")
    for node, data in G.nodes(data=True):
        if str(node).startswith(("CS", "CV")):
            continue
        junction_idx = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15, name=str(node))
        nx_to_pp_junctions[node] = junction_idx

    # Identify and select the pressure reference node (IC)
    reference_candidates = {n for n in G.nodes if str(n).startswith("IC")}
    if not reference_candidates:
        print("ERROR: No Interconnector (IC) nodes found. Cannot set a pressure reference.")
        return None
    reference_node = list(reference_candidates)[0]
    print(f"Selected Interconnector '{reference_node}' as the main pressure reference.")

    # Create the single external grid
    pp.create_ext_grid(net, junction=nx_to_pp_junctions[reference_node], p_bar=60.0, t_k=293.15)

    # Create all other network components
    processed_edges = set()

    # Handle node-based components
    print("Creating node-based components (sources, sinks, compressors, valves)...")
    for node, data in G.nodes(data=True):
        node_name = str(node)
        if node == reference_node:
            continue

        if node_name.startswith("CS") or node_name.startswith("CV"):
            # A component should have exactly one inlet and one outlet
            if G.in_degree(node) != 1 or G.out_degree(node) != 1:
                print(f"WARNING: Component '{node_name}' requires 1 inlet/1 outlet, but has "
                      f"{G.in_degree(node)}/{G.out_degree(node)}. Skipping.")
                continue

            # Get the defined inlet and outlet from the directed graph
            inlet = list(G.predecessors(node))[0]
            outlet = list(G.successors(node))[0]

            if str(inlet).startswith(("CS", "CV")) or str(outlet).startswith(("CS", "CV")):
                print(f"WARNING: Component '{node_name}' is adjacent to another active component. Skipping.")
                continue

            # Get junctions for the inlet and outlet nodes
            junc_in = nx_to_pp_junctions[inlet]
            junc_out = nx_to_pp_junctions[outlet]
            
            # Use the defined direction to orient the component
            if node_name.startswith("CS"):
                ratio = data.get('pressure_ratio', 1.2)
                pp.create_compressor(net, from_junction=junc_in, to_junction=junc_out,
                                    pressure_ratio=ratio, name=node_name)
            elif node_name.startswith("CV"):
                diam_m = data.get('DN', 100) / 100
                pp.create_valve(net, from_junction=junc_in, to_junction=junc_out,
                                diameter_m=diam_m, opened=True, name=node_name)

            # Mark the two directed edges as processed
            processed_edges.add((inlet, node))
            processed_edges.add((node, outlet))

        # Handle sources and sinks
        else:
            supply_value = data.get('supply', 0)
            if supply_value > 0:
                pp.create_source(net, junction=nx_to_pp_junctions[node], mdot_kg_per_s=supply_value)
            elif supply_value < 0:
                pp.create_sink(net, junction=nx_to_pp_junctions[node], mdot_kg_per_s=abs(supply_value))

    # Create pipes for all remaining edges
    print("Creating passive components (pipes)...")
    for u, v, data in G.edges(data=True):
        # Skip any edge connected to a CS or CV node. This acts as a safeguard for components that had invalid configurations
        if str(u).startswith(("CS", "CV")) or str(v).startswith(("CS", "CV")):
            continue
        
        if (u, v) in processed_edges:
            continue
            
        pp.create_pipe_from_parameters(net, from_junction=nx_to_pp_junctions[u],
                                       to_junction=nx_to_pp_junctions[v],
                                       length_km=data.get('L', 0) / 1000,
                                       diameter_m=data.get('DN', 0) / 100,
                                       name=f"Pipe {u}-{v}")

    # Final checks and simulation run
    zero_diameter_pipes = net.pipe[net.pipe.diameter_m == 0]
    if not zero_diameter_pipes.empty:
        print("WARNING: Found pipes with zero diameter, which act as blockages. Skipping.")
        return None

    print("\nRunning simulation...")
    try:
        pp.pipeflow(net, stop_condition="tol", iter=30, tol_p=1e-4, tol_v=1e-4)
        print("Simulation successful!")
        return net
    except Exception as e:
        print(f"Simulation failed: {e}")
        return None
    

def simulate_clustered_network(simplified_graph: nx.DiGraph) -> pp.pandapipesNet:
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
    original_node_data = nx.get_node_attributes(simplified_graph, 'original_node_data') # Assuming you store this

    # --- AGGREGATION LOGIC ---
    for simp_node, junction_idx in nx_to_pp_junctions.items():
        original_nodes_in_cluster = virtual_node_map.get(simp_node, [simp_node])
        
        net_supply_kg_per_s = 0
        for orig_node in original_nodes_in_cluster:
            # You must have access to the original node's data here
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

def prepare_graph_for_max_flow(g: nx.DiGraph) -> nx.DiGraph:
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
        G:nx.DiGraph, 
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

def calculate_deliverability_error(G_original: nx.DiGraph, G_simplified: nx.DiGraph) -> float:
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

    print(f"Original Deliverability: {f_orig:.2f} kg/s")
    print(f"Simplified Deliverability: {f_simp:.2f} kg/s")

    if f_orig == 0:
        return 0.0 if f_simp == 0 else float('inf')

    error = abs(f_orig - f_simp) / f_orig
    if error > 1: return 1

    return error