import networkx as nx
from networkx.algorithms.flow import edmonds_karp
import pandas as pd
from tqdm import tqdm
import geopandas as gpd

from lib.utils import estimate_gas_flow, graph_to_nodes_df
from lib.simulation import calculate_max_deliverability, prepare_graph_for_max_flow

def calculate_node_type_importance(graph: nx.Graph) -> pd.DataFrame:
    """
    Performs a more efficient Network Resilience & Contingency Analysis (N-1 Criterion).
    Avoids redundant graph copying by temporarily removing and restoring nodes.
    """
    ALL_NODE_TYPES = ['BIO', 'CS', 'CV', 'DSO', 'GPR', 'IC', 'IND', 'LNG', 'ST', 'TPP', 'X']
    sorted_node_types = sorted(ALL_NODE_TYPES, key=len, reverse=True)
    all_nodes = list(graph.nodes())

    # Prepare the flow network once
    flow_network = prepare_graph_for_max_flow(graph)
    
    # Run baseline simulation.
    baseline_deliverability, _ = nx.maximum_flow(flow_network, "super_source", "super_sink", capacity="capacity")

    if baseline_deliverability <= 0:
        print("Warning: Baseline deliverability is 0. Cannot calculate importance scores.")
        return pd.DataFrame(columns=['node_name', 'node_type', 'impact_pct'])

    print(f"Baseline Network Deliverability: {baseline_deliverability:,.2f} kg/s\n")
    
    results_list = []
    progress_bar = tqdm(all_nodes, desc="N-1 Contingency Analysis", unit="node")

    for node_to_remove in progress_bar:
        if not flow_network.has_node(node_to_remove):
            continue

        # Store original node data and its incident edges
        original_node_data = flow_network.nodes[node_to_remove].copy()
        edges_to_restore = list(flow_network.edges(node_to_remove, data=True))
        
        flow_network.remove_node(node_to_remove)

        # Run the max flow calculation
        try:
            contingency_deliverability, _ = nx.maximum_flow(flow_network, "super_source", "super_sink", capacity="capacity")
        except nx.NetworkXError:
            contingency_deliverability = 0.0

        # Restore the graph for the next iteration
        flow_network.add_node(node_to_remove, **original_node_data)
        flow_network.add_edges_from(edges_to_restore)
        
        drop_percentage = (baseline_deliverability - contingency_deliverability) / baseline_deliverability
        node_type = next((t for t in sorted_node_types if str(node_to_remove).lower().startswith(t.lower())), 'UNKNOWN')
        results_list.append({
            'node_name': node_to_remove, 'node_type': node_type, 'impact_pct': drop_percentage
        })
        progress_bar.set_postfix({
            "Node": f"{str(node_to_remove)[:15]:<15}", "Type": node_type, "Drop": f"{drop_percentage:.2%}"
        })

    print("\n\nAnalysis Complete. Returning detailed results DataFrame.")
    print(f"Graphs are isomorphic: {nx.is_isomorphic(graph, flow_network)}")
    return pd.DataFrame(results_list)

def aggregate_results(results:pd.DataFrame) -> None:
    # Perform the aggregation using pandas groupby and print the results
    if not results.empty:
        aggregated_scores = results.groupby('node_type')['impact_pct'].mean().sort_values(ascending=False)
        
        # Convert to DataFrame for pretty printing
        final_df = aggregated_scores.to_frame(name='avg_importance_score')
        #final_df["norm_avg_importance_score"] = final_df["avg_importance_score"] / final_df["avg_importance_score"].max()
        final_df["norm_avg_importance_score"] = (final_df["avg_importance_score"]-final_df["avg_importance_score"].min())/(final_df["avg_importance_score"].max()-final_df["avg_importance_score"].min())
        final_df.to_csv("data/property_weights.csv")

def run_analysis(G:nx.Graph) -> None:
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Run the analysis function to get detailed results
    detailed_results_df = calculate_node_type_importance(G)
    detailed_results_df.to_csv("data/detailed_property_weights.csv", index=False)

    aggregate_results(detailed_results_df)

#run_analysis(nx.read_gml("./data/de2025_simp.gml"))