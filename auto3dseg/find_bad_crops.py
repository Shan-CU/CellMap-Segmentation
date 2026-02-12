#!/usr/bin/env python3
"""Find NIfTI crops that would cause CropForegroundd to collapse to empty.

A crop is "bad" if the EM image is constant (all same value), because
CropForegroundd computes the bounding box of non-zero voxels and gets
an empty result, producing a [0,0,0] spatial tensor that crashes Spacingd.
"""
import json
import os
import sys

import nibabel as nib
import numpy as np


def main():
    datalist_path = "auto3dseg/nifti_data/datalist.json"
    basedir = os.getcwd()

    with open(datalist_path) as f:
        dl = json.load(f)

    bad_train = []
    bad_val = []

    for split_key, bad_list in [("training", bad_train), ("validation", bad_val)]:
        entries = dl.get(split_key, [])
        for i, entry in enumerate(entries):
            img_path = os.path.join(basedir, entry["image"])
            if not os.path.exists(img_path):
                print(f"MISSING: {entry['image']}")
                bad_list.append(entry["image"])
                continue
            img = nib.load(img_path)
            data = np.asarray(img.dataobj)
            mn, mx = float(data.min()), float(data.max())
            nz = np.count_nonzero(data)
            if mn == mx:
                print(f"CONSTANT ({split_key}[{i}]): {entry['image']}  shape={data.shape} val={mx}")
                bad_list.append(entry["image"])
            elif nz < 100:
                print(f"NEAR-EMPTY ({split_key}[{i}]): {entry['image']}  shape={data.shape} nonzero={nz}")
                bad_list.append(entry["image"])

    print(f"\n=== Summary ===")
    print(f"Training:   {len(dl['training'])} total, {len(bad_train)} bad")
    print(f"Validation: {len(dl['validation'])} total, {len(bad_val)} bad")

    if bad_train or bad_val:
        print(f"\nBad image paths:")
        for p in bad_train + bad_val:
            print(f"  {p}")


if __name__ == "__main__":
    main()
