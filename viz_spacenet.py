import os
import argparse
import imageio.v2 as imageio
import pickle
from roadtracer.lib.discoverlib import graph
from viz import get_connections
from viz import insert_connections
from viz import write_graph_to_file_bidirectional
from viz import save
from viz import restruct_graph

def main(base_folder, min_graph_distance, max_straight_distance):
    base_folder = base_folder
    graph_path = f'{base_folder}/graph'
    mask_path = f'{base_folder}/mask'
    processed_path = f'{base_folder}/processed'

    os.makedirs(processed_path, exist_ok=True)
    for filename in os.listdir(graph_path):
        file_path = os.path.join(graph_path, filename)
        
        if os.path.isfile(file_path):
            filename = os.path.splitext(filename)[0]
            
            print(f"processing {filename}")
            
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
            
    for file in os.listdir(processed_path):
        if file.endswith('.txt'):
            os.remove(os.path.join(processed_path, file))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph reconstruction parameters")
    parser.add_argument('--base_folder', type=str, help='Base folder')
    parser.add_argument('--min_graph_distance', type=int, default=16, help='Minimum graph distance (default: 16)')
    parser.add_argument('--max_straight_distance', type=int, default=32, help='Maximum straight-line distance (default: 32)')
    args = parser.parse_args()

    main(args.base_folder, args.min_graph_distance, args.max_straight_distance)
