import socket

def get_ip_address():
    """Get the current IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

import os
def has_write_permission(directory_path):
    """Check if the current user has write permission for the specified directory."""
    return os.access(directory_path, os.W_OK)


import pandas as pd
import numpy as np
import logging

def save_preprocessed_data(gt_array, variant_info_df, output_file_prefix):
    numpy_save_file_name = f"{output_file_prefix}_matrix.npy"
    pandas_save_file_name = f"{output_file_prefix}_variant.csv"

    np.save(numpy_save_file_name, gt_array)
    variant_info_df.to_csv(pandas_save_file_name, sep=",", index = False)

    print(f"genotype matrix shape (#samples, #features) : {gt_array.shape} -> saved to {numpy_save_file_name}")
    print(f"variant info dataframe (#features, ): {variant_info_df.shape} -> saved to {pandas_save_file_name}")

def read_preprocessed_data(target_file_prefix):
    mat_file_name = f"{target_file_prefix}_matrix.npy"
    variant_info_file_name = f"{target_file_prefix}_variant.csv"
    print(f"Reading data from files {mat_file_name} and {variant_info_file_name}")

    if (not os.path.isfile(mat_file_name)) or (not os.path.isfile(variant_info_file_name)):
        logging.warning(f"can not find preprocessed files starting with {target_file_prefix}")
        return None, None;

    gt_array = np.load(mat_file_name)
    variant_info_df = pd.read_csv(variant_info_file_name, dtype = str)

    assert gt_array.shape[1] == variant_info_df.shape[0]

    print(f"Read genotype array of shape {gt_array.shape} and variant info dataframe of shape {variant_info_df.shape}")
    return gt_array, variant_info_df

class SNPDataSet:
    def __init__(self, genotype_array, variant_info_df, sample_annotation_df):
        """
        Initialize the SNPDataSet with genotype data, variant information, and sample annotations.

        :param genotype_array: NumPy array with shape (# samples, # snps)
        :param variant_info_df: Pandas DataFrame with shape (# snps, variant information)
        :param sample_annotation_df: Pandas DataFrame with shape (# samples, sample annotations)
        """

        assert genotype_array.shape[0] == sample_annotation_df.shape[0]
        assert genotype_array.shape[1] == variant_info_df.shape[0]

        self.genotype_array = genotype_array
        self.variant_info_df = variant_info_df
        self.sample_annotation_df = sample_annotation_df

    @classmethod
    def from_file(cls, target_file_prefix, sample_annotation_df):
        """
        Initialize the SNPDataSet from files.

        :param target_file_prefix: Prefix for file saving genotype array ("{target_file_prefix}_matrix.npy") and variant information ("{target_file_prefix}__variant.csv")
        :param sample_annotation_df: sample annotations dataframe
        """

        genotype_array, variant_info_df = read_preprocessed_data(target_file_prefix)

        # Create an instance of SNPDataSet with the loaded data
        return cls(genotype_array, variant_info_df, sample_annotation_df)
    
    def filter_variant(self, filter_array, inplace = True):
        """Fileter variants by the given boolean array"""

        assert filter_array.shape[0] == self.genotype_array.shape[1], f"Filter size is not matched. Passed filter shape: {filter_array.shape}, Genotype array: {self.genotype_array.shape}"

        if inplace:
            num_variant_before = self.genotype_array.shape[1]

            self.genotype_array = self.genotype_array[:, filter_array]
            self.variant_info_df = self.variant_info_df.iloc[filter_array]

            assert self.genotype_array.shape[1] == self.variant_info_df.shape[0] 
            num_variant_after = self.genotype_array.shape[1]

            print(f"Filter variant retained {num_variant_after} / {num_variant_before}")
        else:
            raise NotImplementedError


    def save_data(self, output_file_prefix):
        save_preprocessed_data(self.genotype_array, self.variant_info_df, output_file_prefix)

### test initialization
# a = np.array([[1,2,3,4], [5,6,7,8]])
# v = pd.DataFrame([["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]])
# data = SNPDataSet(a,v,sample_annotation_df[:2])
