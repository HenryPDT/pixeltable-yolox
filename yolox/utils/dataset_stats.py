# Copyright (c) Megvii, Inc. and its affiliates.

import numpy as np
from loguru import logger


def analyze_dataset_stats(dataset, dataset_name="Dataset"):
    """
    Analyze dataset statistics including total images, annotated images, and background images.
    
    Args:
        dataset: Dataset object (CocoDataset, VocDetection, or similar)
        dataset_name (str): Name of the dataset for logging
        
    Returns:
        dict: Dictionary containing dataset statistics
    """
    total_images = len(dataset)
    annotated_images = 0
    background_images = 0
    total_annotations = 0
    
    if total_images == 0:
        logger.warning(f"{dataset_name} is empty!")
        return {
            'total_images': 0,
            'annotated_images': 0,
            'background_images': 0,
            'total_annotations': 0,
            'avg_annotations_per_image': 0
        }
    
    logger.info(f"Analyzing {dataset_name} statistics...")
    
    # Get the underlying dataset if it's wrapped in MosaicDetection
    base_dataset = getattr(dataset, '_dataset', dataset)
    
    # Sample a subset for large datasets to avoid long analysis times
    sample_size = min(1000, total_images) if total_images > 1000 else total_images
    indices = np.linspace(0, total_images - 1, sample_size, dtype=int)
    
    for i in indices:
        try:
            # Get annotations for this image
            annotations = None
            
            if hasattr(base_dataset, 'load_anno'):
                annotations = base_dataset.load_anno(i)
            elif hasattr(base_dataset, 'annotations') and i < len(base_dataset.annotations):
                # Handle both COCO format (tuple) and VOC format 
                anno_data = base_dataset.annotations[i]
                if isinstance(anno_data, tuple) and len(anno_data) > 0:
                    annotations = anno_data[0]  # Get the first element (bbox annotations)
                else:
                    annotations = anno_data
            
            # Fallback: try to get item and extract target
            if annotations is None:
                try:
                    if hasattr(base_dataset, 'pull_item'):
                        _, target, _, _ = base_dataset.pull_item(i)
                    else:
                        item = base_dataset[i]
                        if isinstance(item, tuple) and len(item) >= 2:
                            target = item[1]
                        else:
                            target = None
                    annotations = target
                except:
                    annotations = None
            
            # Check if image has annotations
            if annotations is not None and len(annotations) > 0:
                # Filter out zero/empty annotations
                if isinstance(annotations, np.ndarray):
                    # Check if it's a standard format with class labels in the last column
                    if annotations.ndim == 2 and annotations.shape[1] >= 5:
                        # Filter annotations where class >= 0 (valid classes)
                        valid_annotations = annotations[annotations[:, 4] >= 0]
                    else:
                        valid_annotations = annotations
                    
                    if len(valid_annotations) > 0:
                        annotated_images += 1
                        total_annotations += len(valid_annotations)
                    else:
                        background_images += 1
                else:
                    # Non-numpy format, assume it has annotations
                    annotated_images += 1
                    total_annotations += len(annotations) if hasattr(annotations, '__len__') else 1
            else:
                background_images += 1
                
        except Exception as e:
            logger.debug(f"Error analyzing image {i}: {e}")
            # Assume it's a background image if we can't analyze it
            background_images += 1
    
    # Scale up statistics if we sampled
    if sample_size < total_images:
        scale_factor = total_images / sample_size
        annotated_images = int(annotated_images * scale_factor)
        background_images = int(background_images * scale_factor)
        total_annotations = int(total_annotations * scale_factor)
        
        # Ensure totals add up correctly
        if annotated_images + background_images != total_images:
            background_images = total_images - annotated_images
    
    stats = {
        'total_images': total_images,
        'annotated_images': annotated_images,
        'background_images': background_images,
        'total_annotations': total_annotations,
        'avg_annotations_per_image': total_annotations / max(annotated_images, 1),
        'sampled': sample_size < total_images
    }
    
    return stats


def log_dataset_stats(stats, dataset_name="Dataset"):
    """
    Log dataset statistics in a formatted way.
    
    Args:
        stats (dict): Statistics dictionary from analyze_dataset_stats
        dataset_name (str): Name of the dataset for logging
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"{dataset_name.upper()} STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"Total images: {stats['total_images']:,}")
    logger.info(f"Annotated images: {stats['annotated_images']:,} ({stats['annotated_images']/stats['total_images']*100:.1f}%)")
    logger.info(f"Background images: {stats['background_images']:,} ({stats['background_images']/stats['total_images']*100:.1f}%)")
    logger.info(f"Total annotations: {stats['total_annotations']:,}")
    logger.info(f"Average annotations per annotated image: {stats['avg_annotations_per_image']:.2f}")
    
    if stats.get('sampled', False):
        logger.info("Note: Statistics estimated from sample due to large dataset size")
    
    logger.info(f"{'='*60}\n") 