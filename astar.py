import os
import argparse
import imageio.v2 as imageio
import pickle
import numpy as np
import cv2
import time
from roadtracer.lib.discoverlib import graph
from viz import get_connections
from viz import insert_connections
from viz import write_graph_to_file_bidirectional
from viz import save
from viz import restruct_graph

import sys
sys.path.append('/content/sam_road')
from dataset import spacenet_data_partition, cityscale_data_partition, read_rgb_img
import triage

def adj_dict_to_nodes_edges(adj_dict):
    node_list = list(adj_dict.keys())
    node_index_map = {node: i for i, node in enumerate(node_list)}
    
    edges = set()
    for node, neighbors in adj_dict.items():
        for neighbor in neighbors:
            edge = tuple(sorted((node_index_map[node], node_index_map[neighbor])))
            edges.add(edge)
    
    nodes = np.array(node_list, dtype=np.float32)
    edges = np.array(list(edges), dtype=np.int32)
    
    return nodes, edges

def main(base_folder, dataset, min_graph_distance, max_straight_distance):
    base_folder = base_folder
    graph_path = f'{base_folder}/graph'
    mask_path = f'{base_folder}/mask'
    processed_path = f'{base_folder}/processed'
    viz_path = f'{base_folder}/viz_astar'
    
    if dataset == 'cityscale':
        _, _, test_img_indices = cityscale_data_partition()
        rgb_pattern = '/content/sam_road/cityscale/20cities/region_{}_sat.png'
    elif dataset == 'spacenet':
        _, _, test_img_indices = spacenet_data_partition()
        rgb_pattern = '/content/sam_road/spacenet/RGB_1.0_meter/{}__rgb.png'

    os.makedirs(processed_path, exist_ok=True)
    for filename in os.listdir(graph_path):
        file_path = os.path.join(graph_path, filename)
        
        if os.path.isfile(file_path):
            filename = os.path.splitext(filename)[0]
            
            print(f"processing {filename}")
            
            # A* Implementation
            outim = imageio.imread(os.path.join(mask_path, f"{filename}_road.png")).astype('float32') / 255.0
            outim = outim.swapaxes(0, 1)
            
            with open(os.path.join(graph_path, f"{filename}.p"), 'rb') as file:
                graph_data = pickle.load(file)
            
            write_graph_to_file_bidirectional(graph_data, os.path.join(processed_path, f"{filename}.txt"))
            
            g = graph.read_graph(os.path.join(processed_path, f"{filename}.txt"))
            connections = get_connections(g, outim, min_graph_distance, max_straight_distance)
            g = insert_connections(g, connections)
            save(g, os.path.join(processed_path, f"{filename}.txt"))
            
            restruct_graph(os.path.join(processed_path, f"{filename}.txt"), os.path.join(processed_path, f"{filename}.p"))
            
            # Visualize
            with open(os.path.join(processed_path, f"{filename}.p"), 'rb') as f:
                adj_dict = pickle.load(f)
            
            nodes, edges = adj_dict_to_nodes_edges(adj_dict)

            img = read_rgb_img(rgb_pattern.format(filename))
            viz_img = np.copy(img)
            img_size = viz_img.shape[0]

            if not os.path.exists(viz_path):
                os.makedirs(viz_path)
                
            if nodes is not None and len(nodes) > 0:
                if dataset == 'spacenet':
                    nodes = np.stack([400 - nodes[:, 0], nodes[:, 1]], axis=1)
                viz_img = triage.visualize_image_and_graph(viz_img, nodes / img_size, edges, viz_img.shape[0])
                
            cv2.imwrite(os.path.join(viz_path, f'{filename}.png'), viz_img)
            
    for file in os.listdir(processed_path):
        if file.endswith('.txt'):
            os.remove(os.path.join(processed_path, file))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph reconstruction parameters")
    parser.add_argument('--base_folder', type=str, help='Base folder')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--min_graph_distance', type=int, default=16, help='Minimum graph distance (default: 16)')
    parser.add_argument('--max_straight_distance', type=int, default=32, help='Maximum straight-line distance (default: 32)')
    args = parser.parse_args()

    start_time = time.time()
    
    main(args.base_folder, args.dataset, args.min_graph_distance, args.max_straight_distance)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
