# Test Data for AWS SageMaker Training

## Sample Dataset

A minimal sample dataset is provided in `sample/` directory for quick testing:
- **Size**: ~83MB
- **Images**: 3 per split (train/valid/test)
- **Format**: COCO format with `_annotations.coco.json`

### Structure
```
sample/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.png
│   ├── image2.png
│   └── image3.png
├── valid/
│   ├── _annotations.coco.json
│   └── ...
└── test/
    ├── _annotations.coco.json
    └── ...
```

## Full Dataset

 As required for the RF-DETR RoboFlow model, create a COCO format dataset with:
   ```
   dataset/
   ├── train/
   │   ├── _annotations.coco.json
   │   └── *.png/jpg (images)
   ├── valid/
   │   ├── _annotations.coco.json
   │   └── *.png/jpg (images)
   └── test/ (optional)
       ├── _annotations.coco.json
       └── *.png/jpg (images)
   ```

### COCO Annotation Format

The `_annotations.coco.json` file should have this structure:
```json
{
  "images": [
    {
      "id": 0,
      "file_name": "image1.png",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "bbox": [x, y, width, height],
      "area": width * height,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "class_name",
      "supercategory": "none"
    }
  ]
}
```

## Using Test Data

### With Sample Dataset (Quick Test)
```bash
# Use the included sample
./scripts/test_docker_local.sh rfdetr-sagemaker-training latest test_data/sample
```

### For SageMaker Deployment
```bash
# The scripts will upload your dataset to S3
./scripts/example_usage.sh
# Edit LOCAL_DATA_PATH in the script to point to your dataset
```
