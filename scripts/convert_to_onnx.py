#!/usr/bin/env python3
"""
Command line utility to convert RF-DETR models to ONNX format.
"""

import argparse
import warnings

# Suppress the deprecation warning
warnings.filterwarnings('ignore', category=UserWarning)

from rfdetr import RFDETRNano, RFDETRBase, RFDETRMedium, RFDETRLarge

model_classes = {
    'nano': RFDETRNano,
    'base': RFDETRBase,
    'medium': RFDETRMedium,
    'large': RFDETRLarge,
}

def main():
    parser = argparse.ArgumentParser(description='Convert RF-DETR model to ONNX format')
    parser.add_argument('--model_size', choices=['nano', 'base', 'medium', 'large'], default='large',
                        help='Model size to use (default: large)')
    parser.add_argument('--weights', required=True,
                        help='Path to the pre-trained weights file (.pth)')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of classes (default: 1)')

    args = parser.parse_args()

    model_class = model_classes[args.model_size]
    model = model_class(num_classes=args.num_classes, weights=args.weights)
    output_path = model.export()
    print(output_path)

if __name__ == '__main__':
    main()