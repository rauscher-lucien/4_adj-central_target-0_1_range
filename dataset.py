import os
import numpy as np
import torch
import glob
import tifffile

from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image

from utils import *



class N2NSliceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.file_list_by_folder = self.get_file_list_by_folder(root_folder_path)
        self.pairs, self.cumulative_slices = self.make_pairs()

    def get_file_list_by_folder(self, root_folder_path):
        file_list_by_folder = {}
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            file_paths = [os.path.join(subdir, f) for f in sorted_files]
            if file_paths:
                file_list_by_folder[subdir] = file_paths
        return file_list_by_folder

    def make_pairs(self):
        pairs = []
        cumulative_slices = [0]  # Initialize with 0 to start cumulative count
        for folder, files in self.file_list_by_folder.items():
            for f in files:
                volume = tifffile.imread(f)
                num_slices = volume.shape[0] - 1  # Subtract 1 because we form pairs within the same volume
                for i in range(num_slices):
                    # Each pair is (current_slice, next_slice) in the same volume
                    pairs.append((f, i, i + 1))  # Store filename and slice indices
                # Update cumulative slices count
                cumulative_slices.append(cumulative_slices[-1] + num_slices)
        return pairs, cumulative_slices

    def __len__(self):
        # The total number of pairs is the last element in the cumulative_slices list
        return self.cumulative_slices[-1]

    def __getitem__(self, index):
        # Find which pair the index falls into
        pair_index = next(i for i, total in enumerate(self.cumulative_slices) if total > index) - 1
        # Get the corresponding file and slice indices
        file_path, slice_index_1, slice_index_2 = self.pairs[pair_index]
        
        volume = tifffile.imread(file_path)
        input_slice = volume[slice_index_1, :, :]
        target_slice = volume[slice_index_2, :, :]

        input_slice_final = input_slice[..., None]
        target_slice_final = target_slice[..., None]

        if self.transform:
            data = self.transform((input_slice_final, target_slice_final))

        return data
    



class N2N4InputSliceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.file_list_by_folder = self.get_file_list_by_folder(root_folder_path)
        self.pairs, self.cumulative_slices = self.make_pairs()

    def get_file_list_by_folder(self, root_folder_path):
        file_list_by_folder = {}
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            file_paths = [os.path.join(subdir, f) for f in sorted_files]
            if file_paths:
                file_list_by_folder[subdir] = file_paths
        return file_list_by_folder

    def make_pairs(self):
        pairs = []
        cumulative_slices = [0]
        for folder, files in self.file_list_by_folder.items():
            for f in files:
                volume = tifffile.imread(f)
                num_slices = volume.shape[0]
                if num_slices >= 5:  # Ensure at least 5 slices for forming pairs
                    for i in range(num_slices - 4):
                        input_slices_indices = [i, i+1, i+3, i+4]
                        target_slice_index = i + 2
                        pairs.append((f, input_slices_indices, target_slice_index))
                        cumulative_slices.append(cumulative_slices[-1] + 1)
        return pairs, cumulative_slices

    def __len__(self):
        return self.cumulative_slices[-1]

    def __getitem__(self, index):
        pair_index = next(i for i, total in enumerate(self.cumulative_slices) if total > index) - 1
        file_path, input_slice_indices, target_slice_index = self.pairs[pair_index]
        
        volume = tifffile.imread(file_path)
        input_slices = np.stack([volume[i] for i in input_slice_indices], axis=-1)  # Stack slices along the last dimension
        target_slice = volume[target_slice_index][..., np.newaxis]  # Add a new axis to match input shape

        if self.transform:
            input_slices, target_slice = self.transform((input_slices, target_slice))

        return input_slices, target_slice
    


class N2N4InputSliceDataset2(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.preloaded_data = {}  # To store preloaded data
        self.pairs, self.cumulative_slices = self.preload_and_make_pairs(root_folder_path)

    def preload_and_make_pairs(self, root_folder_path):
        pairs = []
        cumulative_slices = [0]
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                self.preloaded_data[full_path] = volume  # Preload data here
                num_slices = volume.shape[0]
                if num_slices >= 5:  # Ensure at least 5 slices for forming pairs
                    for i in range(num_slices - 4):
                        input_slices_indices = [i, i+1, i+3, i+4]
                        target_slice_index = i + 2
                        pairs.append((full_path, input_slices_indices, target_slice_index))
                        cumulative_slices.append(cumulative_slices[-1] + 1)
        return pairs, cumulative_slices

    def __len__(self):
        return self.cumulative_slices[-1]

    def __getitem__(self, index):
        pair_index = next(i for i, total in enumerate(self.cumulative_slices) if total > index) - 1
        file_path, input_slice_indices, target_slice_index = self.pairs[pair_index]
        
        # Access preloaded data instead of reading from file
        volume = self.preloaded_data[file_path]
        input_slices = np.stack([volume[i] for i in input_slice_indices], axis=-1)
        target_slice = volume[target_slice_index][..., np.newaxis]

        if self.transform:
            input_slices, target_slice = self.transform((input_slices, target_slice))

        return input_slices, target_slice



class N2N4InputSliceInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.file_list_by_folder = self.get_file_list_by_folder(root_folder_path)
        self.input_stacks, self.cumulative_slices = self.make_input_stacks()

    def get_file_list_by_folder(self, root_folder_path):
        file_list_by_folder = {}
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            file_paths = [os.path.join(subdir, f) for f in sorted_files]
            if file_paths:
                file_list_by_folder[subdir] = file_paths
        return file_list_by_folder

    def make_input_stacks(self):
        input_stacks = []
        cumulative_slices = [0]
        for folder, files in self.file_list_by_folder.items():
            for f in files:
                volume = tifffile.imread(f)
                num_slices = volume.shape[0]
                if num_slices >= 5:  # Ensure at least 5 slices for forming pairs
                    for i in range(num_slices - 4):
                        input_slices_indices = [i, i+1, i+3, i+4]
                        input_stacks.append((f, input_slices_indices))
                        cumulative_slices.append(cumulative_slices[-1] + 1)
        return input_stacks, cumulative_slices

    def __len__(self):
        return self.cumulative_slices[-1]

    def __getitem__(self, index):
        input_stacks_index = next(i for i, total in enumerate(self.cumulative_slices) if total > index) - 1
        file_path, input_slice_indices = self.input_stacks[input_stacks_index]
        
        volume = tifffile.imread(file_path)
        input_slices = np.stack([volume[i] for i in input_slice_indices], axis=-1)

        if self.transform:
            input_slices = self.transform(input_slices)

        return input_slices


class N2NSliceDataset2(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, remove_percentage=50, transform=None):
        """
        Initializes the dataset with the path to a folder containing TIFF stacks, 
        the percentage of slices to randomly remove from each substack in every TIFF stack, 
        and an optional transform to be applied to each pair.

        Parameters:
        - root_folder_path: Path to the folder containing TIFF stack files.
        - remove_percentage: Percentage of slices to randomly remove from each substack.
        - transform: Optional transform to be applied to each pair of slices.
        """
        self.root_folder_path = root_folder_path
        self.remove_percentage = remove_percentage
        self.transform = transform
        self.pairs = self.load_and_process_stacks()

    def load_and_process_stacks(self):
        pairs = []
        for filename in os.listdir(self.root_folder_path):
            if filename.lower().endswith(('.tif', '.tiff')):
                filepath = os.path.join(self.root_folder_path, filename)
                stack = tifffile.imread(filepath)
                pairs.extend(self.process_stack(stack))
        return pairs

    def process_stack(self, stack):
        total_slices = stack.shape[0]
        mid_point = total_slices // 2
        num_slices_to_remove = int(mid_point * self.remove_percentage / 100)

        substack_1 = stack[:mid_point]
        substack_2 = stack[mid_point:2 * mid_point]

        if num_slices_to_remove > 0:
            slices_to_remove = np.random.choice(mid_point, num_slices_to_remove, replace=False)
            substack_1 = np.delete(substack_1, slices_to_remove, axis=0)
            substack_2 = np.delete(substack_2, slices_to_remove, axis=0)

        return [(substack_1[i], substack_2[i]) for i in range(substack_1.shape[0])]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_slice, target_slice = self.pairs[index]
        input_slice_final = input_slice[..., None]
        target_slice_final = target_slice[..., None]

        if self.transform:
            data = self.transform((input_slice_final, target_slice_final))
        else:
            data = (input_slice_final, target_slice_final)

        return data



class N2NInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None, mean=None, std=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.mean = mean
        self.std = std
        self.file_list = self.get_file_list(root_folder_path)
        self.cumulative_slices = self.get_cumulative_slices()

    def get_file_list(self, root_folder_path):
        file_list = []
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            file_paths = [os.path.join(subdir, f) for f in sorted_files]
            file_list.extend(file_paths)
        return file_list

    def get_cumulative_slices(self):
        cumulative_slices = [0]
        for img_path in self.file_list:
            volume = tifffile.imread(img_path)
            cumulative_slices.append(cumulative_slices[-1] + volume.shape[0])
        return cumulative_slices

    def __len__(self):
        # The total number of slices is the last element in the cumulative_slices list
        return self.cumulative_slices[-1]

    def __getitem__(self, index):
        # Find which image stack the index falls into
        stack_index = next(i for i, total in enumerate(self.cumulative_slices) if total > index) - 1
        # Adjust index based on the start of the current stack
        adjusted_index = index - self.cumulative_slices[stack_index]
        
        img_path = self.file_list[stack_index]
        volume = tifffile.imread(img_path)

        # Extract the specific slice from the volume
        slice_ = volume[adjusted_index, :, :]

        # Add channel dimension if necessary
        slice_final = slice_[..., None]

        if self.transform:
            slice_final = self.transform(slice_final)

        return slice_final

