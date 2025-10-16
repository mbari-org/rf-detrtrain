import os
import shutil
from rfdetr import RFDETRMedium

# Import dataset preparation utilities
from prepare_dataset import split, voc_to_coco


def main():
    original_coco = "output_coco.json"
    images_dir1 = "/mnt/DeepSea-AI/scratch/i2mapbulk/Baseline_mbari-i2map-vits-b8-20251008-vss/transformed/images/"
    images_dir2 = "/mnt/DeepSea-AI/scratch/i2map/Baseline_mbari-i2map-vits-b8-20251008-vss/transformed/images/"
    voc_dir1 = "/mnt/DeepSea-AI/scratch/i2mapbulk/Baseline_mbari-i2map-vits-b8-20251008-vss/transformed/voc/"
    voc_dir2 = "/mnt/DeepSea-AI/scratch/i2map/Baseline_mbari-i2map-vits-b8-20251008-vss/transformed/voc/"
    dataset_location = "/mnt/DeepSea-AI/scratch/test_dataset/"  # Base directory for train/valid/test
    # images_dir1 = "/tmp/delme/Baseline_mbari-i2map-vits-b8-20251008-vss/images/"
    # images_dir2 = "/tmp/delme2/images/"
    # voc_dir1 = "/tmp/delme/Baseline_mbari-i2map-vits-b8-20251008-vss/voc/"
    # voc_dir2 = "/tmp/delme2/voc/"
    # dataset_location = "/tmp/test_dataset/"  # Base directory for train/valid/test

    # Combine two VOC datasets with their respective image directories
    voc_to_coco([(voc_dir1, images_dir1), (voc_dir2, images_dir2)], original_coco)

    # Clean the final dataset location if it exists
    if os.path.exists(dataset_location):
        shutil.rmtree(dataset_location)

    # Split the dataset, providing all image directories
    split(original_coco, [images_dir1, images_dir2], dataset_location)

    # model = RFDETRLarge()
    # model.train(dataset_dir=dataset_location, epochs=50, batch_size=8, grad_accum_steps=2)
    model = RFDETRMedium()
    model.train(dataset_dir=dataset_location, epochs=10, batch_size=16, grad_accum_steps=2)


if __name__ == "__main__":
    main()
