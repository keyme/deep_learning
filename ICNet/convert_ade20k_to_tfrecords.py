"""
This module converts the ADE20k dataset into semantic segmentation tfrecords.

This module expects the extracted contents of the dataset downloaded from:
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
"""

import os
import argparse

from utils import load_data_paths, build_semseg_record


SEMSEG_SHAPE = (480, 480)


def main():
    """Convert Pascal VOC2012 into tfrecords."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Root path of extracted Pascal VOC2012 dataset")
    parser.add_argument("output_dir", help="Directory to write tfrecord to")

    args = parser.parse_args()

    train_paths = list(load_data_paths(os.path.join(args.dataset_root, "images", "training")))
    val_paths = list(load_data_paths(os.path.join(args.dataset_root, "images", "validation")))

    train_img_to_mask, val_img_to_mask = {}, {}
    for p in train_paths:
        if p.endswith(".jpg"):
            train_img_to_mask[p] = p.replace("images", "annotations").replace(".jpg", ".png")

    for p in val_paths:
        if p.endswith(".jpg"):
            val_img_to_mask[p] = p.replace("images", "annotations").replace(".jpg", ".png")

    for dataset in [train_img_to_mask, val_img_to_mask]:
        for img_path, mask_path in dataset.items():
            assert os.path.exists(img_path) and os.path.exists(mask_path)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    build_semseg_record(train_img_to_mask, os.path.join(args.output_dir, "ade20k_train.tfrecords"), SEMSEG_SHAPE)
    build_semseg_record(val_img_to_mask, os.path.join(args.output_dir, "ade20k_val.tfrecords"), SEMSEG_SHAPE)

if __name__ == "__main__":
    main()
