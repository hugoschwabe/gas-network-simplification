import networkx as nx
import pandas as pd
from tqdm import tqdm
import geopandas as gpd

from lib.utils import estimate_gas_flow, graph_to_nodes_df
from lib.simulation import calculate_max_deliverability

def calculate_node_type_importance(graph: nx.DiGraph) -> pd.DataFrame:
    """
    Performs a Network Resilience & Contingency Analysis (N-1 Criterion).
    """
    # Define Node Roles and All Types
    SOURCE_TYPES = ['BIO', 'GPR', 'IC', 'LNG']
    SINK_TYPES = ['DSO', 'IND', 'TPP']
    ALL_NODE_TYPES = ['BIO', 'CS', 'CV', 'DSO', 'GPR', 'IC', 'IND', 'LNG', 'ST', 'TPP', 'X']
    
    # Make matching more robust by sorting
    sorted_node_types = sorted(ALL_NODE_TYPES, key=len, reverse=True)
    
    all_nodes = list(graph.nodes())
    
    # Identify source and sink nodes
    source_nodes = list(set([n for n in all_nodes for t in SOURCE_TYPES if n.lower().startswith(t.lower())]))
    sink_nodes = list(set([n for n in all_nodes for t in SINK_TYPES if n.lower().startswith(t.lower())]))

    print(f"Identified {len(source_nodes)} source nodes and {len(sink_nodes)} sink nodes.")

    # Run baseline simulation
    baseline_deliverability = calculate_max_deliverability(graph, source_nodes, sink_nodes, capacity_attr='capacity')

    if baseline_deliverability <= 0:
        print("Warning: Baseline deliverability is 0. Cannot calculate importance scores.")
        return pd.DataFrame(columns=['node_name', 'node_type', 'impact_pct'])

    print(f"Baseline Network Deliverability: {baseline_deliverability:,.2f}\n")

    # Systematically remove nodes and measure impact
    results_list = []

    progress_bar = tqdm(all_nodes, desc="N-1 Contingency Analysis", unit="node")
    for node_to_remove in progress_bar:
        # Determine node type using case-insensitive matching against the sorted list.
        node_type = 'UNKNOWN'
        for t in sorted_node_types:
            if node_to_remove.lower().startswith(t.lower()):
                node_type = t # Assign the canonical type name (e.g., 'CS')
                break # Found the longest possible match, so we can stop.

        temp_graph = graph.copy()
        temp_graph.remove_node(node_to_remove)

        contingency_deliverability = calculate_max_deliverability(
            temp_graph, source_nodes, sink_nodes, capacity_attr='capacity'
        )

        drop_percentage = (baseline_deliverability - contingency_deliverability) / baseline_deliverability
        
        results_list.append({
            'node_name': node_to_remove,
            'node_type': node_type,
            'impact_pct': drop_percentage
        })
        
        progress_bar.set_postfix({
            "Node": f"{node_to_remove[:15]:<15}",
            "Type": node_type,
            "Drop": f"{drop_percentage:.2%}"
        })

    print("\n\nAnalysis Complete. Returning detailed results DataFrame.")
    return pd.DataFrame(results_list)


def run_analysis() -> None:
    G = nx.read_gml("./data/de2025_simp.gml")

    nodes = graph_to_nodes_df(G)
    nodes["x"] = nodes["coord"].map(lambda x: x[0])
    nodes["y"] = nodes["coord"].map(lambda x: x[1])
    nodes = gpd.GeoDataFrame(nodes)
    nodes = nodes.set_geometry(gpd.points_from_xy(nodes["x"], nodes["y"]), crs="EPSG:3035").to_crs("EPSG:4326")

    original = nx.DiGraph()
    original.add_nodes_from(nodes.set_index("nodes").T.to_dict().items())
    original.add_edges_from(G.edges(data=True))

    for u, v, data in original.edges(data=True):
            dn = data.get('DN', 0)
            pmax = data.get('Pmax', 0)
            length = data.get('L', 0)
            
            # Calculate capacity
            capacity = estimate_gas_flow(p_bar=pmax, dn_mm=dn, length_km=length)
            data['capacity'] = capacity

    print(f"Network has {original.number_of_nodes()} nodes and {original.number_of_edges()} edges")

    # Run the analysis function to get detailed results
    detailed_results_df = calculate_node_type_importance(original)
    detailed_results_df.to_csv("data/detailed_property_weights.csv", index=False)
    
def aggregate_results(results) -> None:
    # Perform the aggregation using pandas groupby and print the results
    if not results.empty:
        aggregated_scores = results.groupby('node_type')['impact_pct'].mean().sort_values(ascending=False)
        
        # Convert to DataFrame for pretty printing
        final_df = aggregated_scores.to_frame(name='avg_importance_score')
        final_df["norm_avg_importance_score"] = final_df["avg_importance_score"] / final_df["avg_importance_score"].max()
        final_df.to_csv("data/property_weights.csv")