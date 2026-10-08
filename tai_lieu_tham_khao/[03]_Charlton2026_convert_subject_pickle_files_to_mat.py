#!/usr/bin/env python3
"""
convert_subject_pickle_files_to_mat.py

Converts individual subject pickle files (S1.pkl to S15.pkl) of the PPG-DaLiA dataset
into MATLAB .mat files for collation and signal processing.

Author: Dr. Peter H. Charlton / Adapted for AIoT Capstone
Citation: [3] P. H. Charlton (2026)
Student: Nguyen Bach Tung (MSSV: 23110166)
"""

import os
import sys
import pickle
import scipy.io as sio

def convert_pickle_to_mat(root_folder):
    print(f"Scanning for PPG-DaLiA pickle files in: {root_folder}")
    if not os.path.exists(root_folder):
        print(f"Directory not found: {root_folder}")
        return

    subjects = [f"S{i}" for i in range(1, 16)]
    converted_count = 0

    for subj in subjects:
        subj_dir = os.path.join(root_folder, subj)
        pkl_path = os.path.join(subj_dir, f"{subj}.pkl")
        mat_path = os.path.join(subj_dir, f"{subj}.mat")

        if os.path.exists(pkl_path):
            print(f"Converting {subj} ({pkl_path}) -> {mat_path}...")
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            sio.savemat(mat_path, data)
            converted_count += 1
            print(f"  Successfully saved {mat_path}")
        else:
            print(f"  File not found: {pkl_path} (skipped)")

    print(f"Conversion complete: {converted_count} files converted.")

if __name__ == "__main__":
    default_dir = os.path.join(os.getcwd(), "data", "raw", "PPG_DaLiA")
    target_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    convert_pickle_to_mat(target_dir)
