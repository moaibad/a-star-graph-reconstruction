import os
import imageio.v2 as imageio
import pickle
from roadtracer.lib.discoverlib import graph
from viz import get_connections
from viz import insert_connections
from viz import write_graph_to_file_bidirectional
from viz import save
from viz import restruct_graph

base_folder = 'test'
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
        connections = get_connections(g, outim)
        g = insert_connections(g, connections)
        save(g, os.path.join(processed_path, f"{filename}.txt"))
        
        restruct_graph(os.path.join(processed_path, f"{filename}.txt"), os.path.join(processed_path, f"{filename}.p"))
        
for file in os.listdir(processed_path):
    if file.endswith('.txt'):
        os.remove(os.path.join(processed_path, file))
