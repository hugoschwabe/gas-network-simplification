import pandas as pd
import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F
from torch_geometric.nn import GAE, GATv2Conv
from torch_geometric.data import Data

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import silhouette_score

# GNN model definition
class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        return self.conv2(x, edge_index, edge_attr=edge_attr)

# Data preparation function
def prepare_data(G: nx.DiGraph, coord_weight: float = 1.0) -> tuple[Data, list, np.ndarray]:
    node_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    original_node_names = node_df.index.tolist()

    node_df['type'] = node_df.index.to_series().apply(lambda x: x.split('_')[0])
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    node_type_encoded = encoder.fit_transform(node_df[['type']])
    
    coords = np.array([data['coord'] for _, data in G.nodes(data=True)])
    scaler_coords = MinMaxScaler()
    coords_scaled = scaler_coords.fit_transform(coords)

    node_features = np.hstack([node_type_encoded, coords_scaled * coord_weight])

    edge_list = list(G.edges(data=True))
    node_to_idx = {name: i for i, name in enumerate(original_node_names)}

    if not edge_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_features_scaled = torch.empty((0, 3))
    else:
        edge_index_list = [(node_to_idx[u], node_to_idx[v]) for u, v, _ in edge_list]
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_features_df = pd.DataFrame([{'length': d.get('L', 0), 'diameter': d.get('DN', 0), 'pressure': d.get('PN', 0)} for _, _, d in edge_list])
        scaler_edges = MinMaxScaler()
        edge_features_scaled = scaler_edges.fit_transform(edge_features_df)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_features_scaled, dtype=torch.float)
    )
    
    return data, original_node_names, coords_scaled

# GNN training function
def train_gnn_model(data: Data, epochs: int = 200, hidden_channels: int = 64, out_channels: int = 32) -> GAE:
    encoder = GNNEncoder(data.num_node_features, data.num_edge_features, hidden_channels, out_channels)
    model = GAE(encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("\n--- Starting GNN Model Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index, data.edge_attr)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
    return model

# GNN clustering function
def get_gnn_communities(model: GAE, data: Data, original_node_names: list, n_clusters: int, scaled_coords: np.ndarray, coord_weight: float) -> list[frozenset]:
    """
    Generates node embeddings, combines them with coordinates, clusters them, and formats the output.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index, data.edge_attr).cpu().numpy()

    # Combine embeddings with weighted coordinates for clustering
    combined_features_for_clustering = np.hstack([embeddings, scaled_coords * coord_weight])
    
    print(f"Clustering {combined_features_for_clustering.shape[0]} nodes using combined features (Embeddings + Coords)...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(combined_features_for_clustering)
    print("Clustering complete.")

    communities = [[] for _ in range(n_clusters)]
    for i, label in enumerate(cluster_labels):
        communities[label].append(original_node_names[i])

    return [frozenset(community) for community in communities if community]

# Helper for hyperparameter tuning 
def find_optimal_hyperparameters(model: GAE, data: Data, scaled_coords: np.ndarray, 
                                 k_range: range = range(50, 251, 25), 
                                 weight_range: list = [0.5, 1.0, 2.0, 5.0, 10.0]):
    """
    Finds the best 'k' and 'coord_weight' by searching for the combination
    that maximizes the Silhouette Score.
    """
    print("\n--- Finding Optimal Hyperparameters (k and coord_weight) ---")
    model.eval()
    with torch.no_grad():
        # Get the base embeddings from the GNN
        embeddings = model.encode(data.x, data.edge_index, data.edge_attr).cpu().numpy()
    
    best_score = -1
    best_k = -1
    best_weight = -1

    for weight in weight_range:
        print(f"\nTesting Coordinate Weight: {weight}")
        # Create the combined feature set for this weight
        combined_features = np.hstack([embeddings, scaled_coords * weight])
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(combined_features)
            score = silhouette_score(combined_features, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
                best_weight = weight
                
    print(f"\n---> Optimal Hyperparameters Found <---")
    print(f"  - Best Coordinate Weight: {best_weight}")
    print(f"  - Best Number of Clusters (k): {best_k}")
    print(f"  - With Silhouette Score: {best_score:.4f}")
    
    return best_k, best_weight