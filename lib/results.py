import networkx as nx
import pandas as pd

from lib import utils
from lib import scoring
from lib import property_weights as weights
from lib import algorithms

def scoring_results():
	nuts3 = utils.nuts3()
	
	original = nx.read_gml("./data/de_cs_virtual_edge.gml")
	utils.add_capacity(original)
	utils.add_norm_capacity(original)
	utils.add_supply_from_csv(original, "./data/de_cs_virtual_edge_flow.csv")
	
	try:
		property_weights = pd.read_csv("data/property_weights.csv").set_index("node_type")
	except:
		weights.run_analysis(original)
	property_weights = property_weights["norm_avg_importance_score"].to_dict()
	
	graphs = {}
	algos = {
		"greedy_modularity": algorithms.greedy_modularity_communities,
		"louvain": algorithms.louvain_communities,
		"k_means": algorithms.k_means,
		"gnn": algorithms.gnn_clustering,
		"k2_cores": algorithms.k_core,
		"path_contraction": algorithms.path_contraction,
		"path_contraction_2": algorithms.path_contraction,
		"importance_removal": algorithms.importance_removal, 
	}
	for key, algo in algos.items():
		if key == "path_contraction_2":
			graph = utils.run_algo(
			func=algo,
			original=graphs["k2_cores"].copy(),
			keep_nodes=["CS", "CV", "IC"],
			property_weights=property_weights,
			importance_weights=utils.importance_weights()
		)
		else:
			graph = utils.run_algo(
				func=algo,
				original=original.copy(),
				keep_nodes=["CS", "CV", "IC"],
				property_weights=property_weights,
				importance_weights=utils.importance_weights()
			)
		graphs[key] = graph

	results = {}
	for key, graph in graphs.items():
		utils.write_gml(graph, f"graphs/{key}.gml")
		print(f"scoring {key}")
		score, dict = scoring.score(
			original, 
			graph, 
			nuts3, 
			property_weights=property_weights,
			verbose=False
		)
		results[key] = dict
	
	df = pd.DataFrame(results)
	df.to_csv("data/scoring_results.csv")

	return df

def snapshot(G):
	supply = 0
	demand = 0

	for value in dict(G.nodes(data="supply")).values():
		if value <= 0: demand = demand + value
		else: supply = + supply + value

	print(f"supply: {supply}")
	print(f"demand: {demand}")