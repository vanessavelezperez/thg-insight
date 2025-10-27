import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from utils import path_formatter

def get_matrix(ms):
    #ms = ms.to_numpy()
    RT = ms[:,0]
    im = ms[:,2:]
    TIC = ms[:,1]
    return RT, im, TIC

def collate(CSV_FILES, OUTPUT_H5):
    # Main conversion
    with h5py.File(OUTPUT_H5, "w") as h5:
        # Create mandatory datasets with YOUR predefined values
        for csv_file in CSV_FILES:
            df = pd.read_csv(csv_file)
            RT, intensity_matrix, TIC = get_matrix(df.to_numpy())

            sample_group = h5.create_group(f"{csv_file.name}")
            sample_group.create_dataset("RT", data=RT)
            sample_group.create_dataset("TIC", data=TIC)
            sample_group.create_dataset("intensity_matrix", data=intensity_matrix)

    print(f"Created MSHub-ready file: {OUTPUT_H5}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Path to home directory where all csvs are")
    parser.add_argument("--output", default=None, help="Path+Name of h5 output file")
    args = parser.parse_args()


    output = args.output if args.output else (args.input / 'collated_files.h5')
    files = path_formatter(args.input)
    collate(files, output)
