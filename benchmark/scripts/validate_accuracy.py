#!/usr/bin/env python3
"""Validate model accuracy across backends (CPU, Rocket, RKNN)."""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
import ai_edge_litert.interpreter as tflite


def preprocess_image(image_path, input_shape, input_dtype):
    """Resize and convert image to model input format."""
    img = Image.open(image_path).convert('RGB')
    h, w = input_shape[1], input_shape[2]
    img = img.resize((w, h))
    data = np.array(img, dtype=input_dtype)
    if len(input_shape) == 4:
        data = np.expand_dims(data, axis=0)
    return data


def run_inference(model_path, input_data, delegate_path=None):
    """Run single inference, return output tensor."""
    delegates = None
    if delegate_path:
        delegates = [tflite.load_delegate(delegate_path)]
    interp = tflite.Interpreter(
        model_path=model_path, experimental_delegates=delegates)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    interp.set_tensor(input_details[0]['index'], input_data)
    interp.invoke()
    return interp.get_tensor(interp.get_output_details()[0]['index']).copy()


def validate_classification(model_path, image_dir, labels_file,
                            delegate_path=None, top_k=5):
    """Compute Top-1 and Top-5 accuracy on labeled images."""
    interp = tflite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    del interp

    # Load labels
    labels = {}
    with open(labels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] != '-1':
                labels[parts[0]] = int(parts[1])

    if not labels:
        print("No labeled images found, skipping accuracy validation")
        return None

    top1_correct = 0
    top5_correct = 0
    total = 0

    for img_name, true_label in labels.items():
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue

        input_data = preprocess_image(img_path, input_shape, input_dtype)
        output = run_inference(model_path, input_data, delegate_path)

        output_flat = output.flatten()
        top5_indices = np.argsort(output_flat)[-top_k:][::-1]

        if top5_indices[0] == true_label:
            top1_correct += 1
        if true_label in top5_indices:
            top5_correct += 1
        total += 1

    if total == 0:
        return None

    return {
        'top1_accuracy': top1_correct / total,
        'top5_accuracy': top5_correct / total,
        'total_images': total,
    }


def validate_correctness(model_path, image_dir, delegate_path, num_images=10):
    """Compare delegate output vs CPU reference for correctness."""
    interp = tflite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    del interp

    images = [f for f in os.listdir(image_dir)
              if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))][:num_images]

    if not images:
        # Use random data
        images = [None] * num_images

    max_diffs = []
    for img_name in images:
        if img_name:
            input_data = preprocess_image(
                os.path.join(image_dir, img_name), input_shape, input_dtype)
        else:
            input_data = np.random.randint(
                0, 255, size=input_shape).astype(input_dtype)

        cpu_output = run_inference(model_path, input_data)
        npu_output = run_inference(model_path, input_data, delegate_path)

        diff = np.max(np.abs(cpu_output.astype(int) - npu_output.astype(int)))
        max_diffs.append(int(diff))

    worst_diff = max(max_diffs)
    return {
        'max_quant_diff': worst_diff,
        'mean_max_diff': float(np.mean(max_diffs)),
        'all_diffs': max_diffs,
        'correctness': 'PASS' if worst_diff <= 1 else 'FAIL',
        'num_images': len(images),
    }


def main():
    parser = argparse.ArgumentParser(description='Validate model accuracy')
    parser.add_argument('model', help='Path to .tflite model')
    parser.add_argument('--delegate', '-d', default=None,
                        help='Path to Teflon delegate .so')
    parser.add_argument('--image-dir', '-i', required=True,
                        help='Directory with validation images')
    parser.add_argument('--labels', '-l', default=None,
                        help='Labels file for accuracy (image_name label_idx)')
    parser.add_argument('--task', '-t', default='classification',
                        choices=['classification', 'detection', 'segmentation',
                                 'correctness'],
                        help='Validation task type')
    parser.add_argument('--output-json', '-o', default=None)
    parser.add_argument('--baseline', '-b', default=None,
                        help='Baseline JSON to compare against for regression')
    args = parser.parse_args()

    model_name = os.path.basename(args.model).replace('.tflite', '')
    backend = 'rocket' if args.delegate else 'cpu'
    print(f"Model:   {model_name}")
    print(f"Backend: {backend}")
    print(f"Task:    {args.task}")
    print()

    result = {}

    if args.task == 'correctness' and args.delegate:
        result = validate_correctness(
            args.model, args.image_dir, args.delegate)
        print(f"  Correctness: {result['correctness']}")
        print(f"  Max quant diff: {result['max_quant_diff']}")
        print(f"  Mean max diff:  {result['mean_max_diff']:.2f}")
        print(f"  Images tested:  {result['num_images']}")

    elif args.task == 'classification' and args.labels:
        result = validate_classification(
            args.model, args.image_dir, args.labels, args.delegate)
        if result:
            print(f"  Top-1 accuracy: {result['top1_accuracy']:.1%}")
            print(f"  Top-5 accuracy: {result['top5_accuracy']:.1%}")
            print(f"  Images: {result['total_images']}")
        else:
            print("  No labeled images available")

    else:
        print(f"  Task '{args.task}' with current args not fully supported yet")
        print("  Use --task correctness --delegate <path> for correctness check")

    # Regression check
    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline) as f:
            baseline = json.load(f)
        if isinstance(baseline, list):
            baseline = baseline[-1]
        if 'top1_accuracy' in result and 'top1_accuracy' in baseline:
            drop = baseline['top1_accuracy'] - result['top1_accuracy']
            if drop > 0.01:
                print(f"\n  REGRESSION: Top-1 dropped {drop:.1%} from baseline!")
                sys.exit(1)

    if args.output_json and result:
        record = {'model': model_name, 'backend': backend, 'task': args.task,
                  **result}
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        existing = []
        if os.path.exists(args.output_json):
            with open(args.output_json) as f:
                existing = json.load(f)
        existing.append(record)
        with open(args.output_json, 'w') as f:
            json.dump(existing, f, indent=2)


if __name__ == '__main__':
    main()
