import torch
import os
import numpy as np
from sklearn.manifold import Isomap
import argparse
import matplotlib.pyplot as plt

def load_and_combine_feature_maps(directory_path):
    maps = {}

    # Load all feature maps
    for file_name in os.listdir(directory_path):
        if 'features_' in file_name and file_name.endswith('.npy'):
            file_path = os.path.join(directory_path, file_name)
            print(f'Loading feature map from {file_path}...')
            feature_map = np.load(file_path)
            
            # Remove '.npy' extension
            feature_map_no_ext = file_name[:-4]

            # Determine base name (ignoring 'noisy')
            base_name = feature_map_no_ext.replace('_noisy', '')
            
            # Combine maps with the same base name
            if base_name in maps:
                maps[base_name] = (maps[base_name][0] + feature_map, maps[base_name][1] + "_combined")
            else:
                maps[base_name] = (feature_map, feature_map_no_ext)
            
            print(f'Finished loading feature map from {file_path}')

    if not maps:
        print(f"No feature maps found in {directory_path}.")
    
    return maps

def load_and_combine_predictions(directory_path):
    predictions = {}

    # Load all prediction files
    for file_name in os.listdir(directory_path):
        if 'y_pred_' in file_name and file_name.endswith('.npy'):
            file_path = os.path.join(directory_path, file_name)
            print(f'Loading prediction file from {file_path}...')
            prediction = np.load(file_path)
            
            # Remove '.npy' extension
            prediction_name_no_ext = file_name[:-4]

            # Determine base name (ignoring 'noisy')
            base_name = prediction_name_no_ext.replace('_noisy', '')
            
            # Combine predictions with the same base name
            if base_name in predictions:
                predictions[base_name] = (predictions[base_name][0] + prediction, predictions[base_name][1] + "_combined")
            else:
                predictions[base_name] = (prediction, prediction_name_no_ext)
            
            print(f'Finished loading prediction file from {file_path}')

    if not predictions:
        print(f"No prediction files found in {directory_path}.")
    
    return predictions

def perform_isomap_and_plot(features, predictions, base_name):
    isomap = Isomap(n_components=2, n_neighbors=5)
    isomap_embedding = isomap.fit_transform(features[base_name][0])  # Combined feature map
    plt.figure(figsize=(8, 6))
    plt.scatter(isomap_embedding[:, 0], isomap_embedding[:, 1], c=predictions[base_name][0], cmap='viridis')  # Combined predictions
    plt.colorbar()
    plt.title(f'Isomap for {base_name}')
    plt.savefig(os.path.join(isomap_dir, f"{base_name}_isomap.png"), bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Galaxy10 models')
    parser.add_argument('--path', type=str, default='data/feature_maps', help='Path to the directory containing the training output files')
    args = parser.parse_args()
    
    
    combined_feature_maps = load_and_combine_feature_maps(f'{args.path}/features')
    combined_predictions = load_and_combine_predictions(f'{args.path}/y_pred')
    
    isomap_dir = os.path.join(args.path, 'isomap')
    if not os.path.exists(isomap_dir):
        os.makedirs(isomap_dir)
    
   # Perform Isomap and plot for each base name
    for base_name in combined_feature_maps.keys():
        perform_isomap_and_plot(combined_feature_maps, combined_predictions, base_name)