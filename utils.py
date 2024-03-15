
import os
import warnings
import glob
import random
import torch
import numpy as np
import tifffile
import pickle
import matplotlib.pyplot as plt

def create_result_dir(project_dir, name='new_results'):

    results_dir = os.path.join(project_dir, name, 'results')
    os.makedirs(results_dir, exist_ok=True)
    checkpoints_dir = os.path.join(project_dir, name, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    return results_dir, checkpoints_dir

def normalize_dataset(dataset):
    all_means = []
    all_stds = []
    all_sizes = []

    # Compute mean, std, and size for each stack
    for stack in dataset:
        all_means.append(np.mean(stack))
        all_stds.append(np.std(stack))
        all_sizes.append(stack.size)

    # Convert lists to numpy arrays for easier computation
    array_means = np.array(all_means)
    array_stds = np.array(all_stds)
    array_sizes = np.array(all_sizes)

    # Compute weighted average of mean and std based on array sizes
    total_size = np.sum(array_sizes)
    weighted_mean = np.sum(array_means * array_sizes) / total_size
    weighted_std = np.sqrt(np.sum(array_stds**2 * array_sizes) / total_size)

    # Set global mean and std
    mean = weighted_mean
    std = weighted_std

    # Compute global minimum and maximum over the entire dataset
    global_min = np.min([np.min(stack) for stack in dataset])
    global_max = np.max([np.max(stack) for stack in dataset])

    # Apply global normalization to the entire dataset using the global min and max
    normalized_dataset = []
    for stack in dataset:
        # Normalize each slice in the stack using the global mean and std
        stack_normalized = (stack - mean) / std

        # Normalize each slice in the stack using the global min and max
        stack_normalized = (stack - global_min) / (global_max - global_min)

        # Clip and normalize to [0, 1] for each slice in the stack using the global min and max
        stack_normalized = np.clip(stack_normalized, 0, 1)

        normalized_dataset.append(stack_normalized.astype(np.float32))

    return normalized_dataset


def compute_global_mean_and_std(dataset_path, checkpoints_path):
    """
    Computes and saves the global mean and standard deviation across all TIFF stacks
    in the given directory and its subdirectories, saving the results in the same directory.

    Parameters:
    - dataset_path: Path to the directory containing the TIFF files.
    """
    all_means = []
    all_stds = []
    for subdir, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(('.tif', '.tiff')):
                filepath = os.path.join(subdir, filename)
                stack = tifffile.imread(filepath)
                all_means.append(np.mean(stack))
                all_stds.append(np.std(stack))
                
    global_mean = np.mean(all_means)
    global_std = np.mean(all_stds)
    
    # Define the save_path in the same directory as the dataset
    save_path = os.path.join(checkpoints_path, 'normalization_params.pkl')

    # Save the computed global mean and standard deviation to a file
    with open(save_path, 'wb') as f:
        pickle.dump({'mean': global_mean, 'std': global_std}, f)
    
    print(f"Global mean and std parameters saved to {save_path}")
    return global_mean, global_std



from concurrent.futures import ProcessPoolExecutor

def process_image(file_path):
    """
    Function to read an image file and compute its mean and standard deviation.
    """
    if file_path.lower().endswith('.tiff'):
        stack = tifffile.imread(file_path)
        return np.mean(stack), np.std(stack)
    return None




def denormalize_image(normalized_img, mean, std):
    """
    Denormalizes an image back to its original range using the provided mean and standard deviation.

    Parameters:
    - normalized_img: The image to be denormalized.
    - mean: The mean used for the initial normalization.
    - std: The standard deviation used for the initial normalization.

    Returns:
    - The denormalized image.
    """
    original_img = (normalized_img * std) + mean
    return original_img.astype(np.float32)


import os
import pickle

def load_normalization_params(data_dir):
    """
    Loads the mean and standard deviation values from a pickle file located in the specified data directory.

    Parameters:
    - data_dir: Path to the directory containing the 'normalization_params.pkl' file.

    Returns:
    - A tuple containing the mean and standard deviation values.
    """
    # Construct the path to the pickle file
    load_path = os.path.join(data_dir, 'normalization_params.pkl')
    
    # Load the parameters from the pickle file
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    
    mean = params['mean']
    std = params['std']
    
    return mean, std




def load_min_max_params(dir):
    """
    Loads the global minimum and maximum values from a pickle file located in the specified data directory.

    Parameters:
    - data_dir: Path to the directory containing the 'min_max_params.pkl' file.

    Returns:
    - A tuple containing the global minimum and maximum values.
    """
    # Construct the path to the pickle file
    load_path = os.path.join(dir, 'min_max_params.pkl')
    
    # Load the parameters from the pickle file
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    
    global_min = params['global_min']
    global_max = params['global_max']
    
    return global_min, global_max


def plot_intensity_distribution(image_array, title='1', block_execution=True):
    """
    Plots the intensity distribution and controls execution flow based on 'block_execution'.
    """
    # Check if the input is a PyTorch tensor and convert it to a NumPy array if so
    if isinstance(image_array, torch.Tensor):
        # Ensure it's on the CPU and convert to NumPy
        image_array = image_array.detach().numpy()

    print(np.max(image_array))

    # Create a new figure for each plot to avoid conflicts
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(image_array.flatten(), bins=50, color='blue', alpha=0.7)
    ax.set_title('Intensity Distribution '+ title)
    ax.set_xlabel('Intensity Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    if block_execution:
        plt.show()
        plt.pause(1)
        plt.close(fig)
    else:
        plt.draw()
        plt.pause(1)  # Allows GUI to update
        plt.close(fig)  # Close the new figure explicitly



def get_file_path(local_path, remote_path):

    path = ''

    # Detect the operating system
    if os.name == 'nt':  # Windows
        path = local_path
    else:  # Linux and others
        path = remote_path
    
    if not os.path.exists(path):
        warnings.warn(f"Project directory '{path}' not found. Please verify the path.")
        return
    print(f"Using file path: {path}")

    return path


def clip_extremes(data, lower_percentile=0, upper_percentile=100):
    """
    Clip pixel values to the specified lower and upper percentiles.
    """
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return np.clip(data, lower_bound, upper_bound)



def compute_global_min_max_and_save(dataset_path):
    """
    Computes and saves the global minimum and maximum values across all TIFF stacks
    in the given directory and its subdirectories, saving the results in the same directory.

    Parameters:
    - dataset_path: Path to the directory containing the TIFF files.
    """
    global_min = float('inf')
    global_max = float('-inf')
    for subdir, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(('.tif', '.tiff')):
                filepath = os.path.join(subdir, filename)
                stack = tifffile.imread(filepath)
                stack_min = np.min(stack)
                stack_max = np.max(stack)
                global_min = min(global_min, stack_min)
                global_max = max(global_max, stack_max)
    
    # Define the save_path in the same directory as the dataset
    save_path = os.path.join(dataset_path, 'min_max_params.pkl')

    # Save the computed global minimum and maximum to a file
    with open(save_path, 'wb') as f:
        pickle.dump({'global_min': global_min, 'global_max': global_max}, f)
    
    print(f"Global min and max parameters saved to {save_path}")
    return global_min, global_max




def tensor_to_image(tensor, image_path):
    """
    Converts a PyTorch tensor to a NumPy array, normalizes it to the 0-1 range, and saves the image.

    Args:
    - tensor (torch.Tensor): A tensor representing the image.
    - image_path (str): Path to save the image.
    """
    if tensor.dim() == 4:  # Batch of images
        tensor = tensor[0]  # Take the first image from the batch
    tensor = tensor.detach().cpu().numpy()

    # Handle 3D tensor (D, H, W) by selecting the middle slice
    if tensor.shape[0] > 3:  # Assuming the first dimension is depth
        tensor = tensor[tensor.shape[0] // 2]  # Select the middle slice

    if tensor.ndim == 3 and tensor.shape[0] == 3:  # RGB image
        img = np.transpose(tensor, (1, 2, 0))  # Convert to (H, W, C)
    else:  # Grayscale image or single slice from 3D volume
        img = np.squeeze(tensor)  # Remove channel dim if it exists

    img = img - np.min(img)
    img = img / np.max(img)

    plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
    plt.axis('off')
    plt.savefig(image_path)
    plt.close()  # Close the figure to avoid memory leak





