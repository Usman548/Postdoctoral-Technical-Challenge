#!/usr/bin/env python3
"""
Search interface for image retrieval system.
Supports image-to-image and text-to-image search with FAISS.
"""

import torch
import numpy as np
import faiss
from pathlib import Path
import sys
import logging
import argparse
import pickle
import json
import matplotlib.pyplot as plt
from PIL import Image
import time
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.data_loader import PneumoniaMNISTDataset
from task3_retrieval.build_index import ImageEmbeddingExtractor, IndexBuilderConfig
from utils.logger import setup_logger
from utils.visualization import plot_retrieval_results

# Configure logging
logger = setup_logger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system."""
    
    # Index parameters
    index_name: str = 'pneumonia_retrieval'
    index_dir: str = 'models/embeddings'
    
    # Model parameters (must match index building)
    model_name: str = 'resnet18'
    embedding_dim: int = 512
    
    # Search parameters
    default_k: int = 5
    similarity_threshold: float = 0.5
    
    # Output paths
    output_dir: str = 'reports/task3'
    figures_dir: str = 'reports/task3/figures'
    
    # System parameters
    device: str = 'cpu'
    
    def __post_init__(self):
        """Validate configuration."""
        self.output_dir = Path(self.output_dir)
        self.figures_dir = Path(self.figures_dir)
        self.index_dir = Path(self.index_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class RetrievalSystem:
    """
    Image retrieval system with multiple search interfaces.
    
    Features:
    - Image-to-image search
    - Text-to-image search (simplified)
    - Precision@k evaluation
    - Result visualization
    - Integration with Task 1 models
    """
    
    def __init__(self, config: Union[Dict[str, Any], RetrievalConfig]):
        """
        Initialize retrieval system.
        
        Args:
            config: Configuration dictionary or RetrievalConfig object
        """
        # Load configuration
        if isinstance(config, dict):
            self.config = RetrievalConfig(**config)
        elif isinstance(config, RetrievalConfig):
            self.config = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
        
        logger.info(f"Initializing retrieval system with config: {self.config}")
        
        # Load index and metadata
        self._load_index()
        
        # Load dataset for reference images
        self._load_dataset()
        
        # Initialize embedding extractor
        self._init_extractor()
        
        # Build image map for quick lookup
        self._build_image_map()
        
        # Search history for logging
        self.search_history: List[Dict] = []
        
        logger.info("Retrieval system initialization complete")
    
    def _load_index(self) -> None:
        """Load FAISS index and metadata."""
        try:
            # Load FAISS index
            index_path = self.config.index_dir / f'{self.config.index_name}.faiss'
            if not index_path.exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")
            
            self.index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load metadata
            metadata_path = self.config.index_dir / f'{self.config.index_name}_metadata.pkl'
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Load ID to index mapping if available
            id_map_path = self.config.index_dir / f'{self.config.index_name}_id_map.json'
            if id_map_path.exists():
                with open(id_map_path, 'r') as f:
                    self.id_to_index = json.load(f)
            else:
                # Create mapping from metadata
                self.id_to_index = {meta['image_id']: i for i, meta in enumerate(self.metadata)}
            
            # Load index stats if available
            stats_path = self.config.index_dir / 'index_stats.json'
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    self.stats = json.load(f)
                logger.info(f"Index stats: {self.stats}")
            
            # Create mappings
            self._create_mappings()
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def _create_mappings(self) -> None:
        """Create useful mappings from metadata."""
        # Map from index position to metadata
        self.id_to_metadata = {i: meta for i, meta in enumerate(self.metadata)}
        
        # Map from image_id to index (using loaded mapping)
        self.image_id_to_idx = self.id_to_index
        
        # Group by label for faster access
        self.label_to_indices = {
            0: [i for i, meta in enumerate(self.metadata) if meta['label'] == 0],
            1: [i for i, meta in enumerate(self.metadata) if meta['label'] == 1]
        }
        
        logger.info(f"Created mappings: {len(self.metadata)} entries, "
                   f"{len(self.label_to_indices[0])} normal, {len(self.label_to_indices[1])} pneumonia")
    
    def _load_dataset(self) -> None:
        """Load dataset for reference images."""
        try:
            self.dataset = PneumoniaMNISTDataset(
                batch_size=64,  # Larger batch for faster loading
                augment=False,
                num_workers=0,
                pin_memory=False
            )
            _, _, self.test_loader = self.dataset.get_dataloaders()
            self.class_names = self.dataset.get_class_names()
            
            logger.info(f"Dataset loaded: {len(self.test_loader.dataset)} test samples")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _build_image_map(self) -> None:
        """Build mapping from image_id to actual image array."""
        self.image_map = {}
        
        try:
            # Iterate through test loader to collect all images with consistent IDs
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                for j, (img, label) in enumerate(zip(images, labels)):
                    # Create ID exactly as in build_index.py
                    img_id = f"test_{batch_idx:04d}_{j:03d}"
                    self.image_map[img_id] = {
                        'image': img.numpy(),
                        'label': int(label)
                    }
            
            logger.info(f"Built image map with {len(self.image_map)} images")
            
            # Verify first few IDs
            sample_ids = list(self.image_map.keys())[:5]
            logger.info(f"Sample image IDs: {sample_ids}")
            
        except Exception as e:
            logger.error(f"Error building image map: {e}")
    
    def _init_extractor(self) -> None:
        """Initialize embedding extractor."""
        try:
            # Create config for extractor
            extractor_config = IndexBuilderConfig(
                model_name=self.config.model_name,
                embedding_dim=self.config.embedding_dim,
                device=self.config.device
            )
            self.extractor = ImageEmbeddingExtractor(extractor_config)
            
        except Exception as e:
            logger.error(f"Failed to initialize extractor: {e}")
            raise
    
    def _preprocess_query_image(self, query_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess query image for embedding extraction.
        
        Args:
            query_image: Query image array
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Handle different input shapes
        if query_image.ndim == 2:
            # Single image (H, W) -> add channel and batch dimensions
            query_tensor = torch.from_numpy(query_image).unsqueeze(0).unsqueeze(0).float()
        elif query_image.ndim == 3:
            if query_image.shape[0] == 1:
                # Already has channel dimension (1, H, W) -> add batch dimension
                query_tensor = torch.from_numpy(query_image).unsqueeze(0).float()
            else:
                # (H, W, C) or similar - assume grayscale and reshape
                if query_image.shape[-1] == 3:
                    # Convert RGB to grayscale
                    query_image = np.dot(query_image[..., :3], [0.2989, 0.5870, 0.1140])
                query_tensor = torch.from_numpy(query_image).unsqueeze(0).unsqueeze(0).float()
        else:
            raise ValueError(f"Unexpected query image shape: {query_image.shape}")
        
        return query_tensor
    
    def image_to_image_search(self, 
                              query_image: Union[np.ndarray, torch.Tensor], 
                              k: int = None,
                              return_images: bool = True,
                              filter_label: Optional[int] = None) -> List[Dict]:
        """
        Search for similar images using image query.
        
        Args:
            query_image: Query image array or tensor
            k: Number of results to return (uses config.default_k if None)
            return_images: Whether to return actual images
            filter_label: Optional label to filter results
            
        Returns:
            List of retrieved results with metadata
        """
        start_time = time.time()
        
        if k is None:
            k = self.config.default_k
        
        try:
            # Handle tensor input
            if isinstance(query_image, torch.Tensor):
                query_tensor = query_image
            else:
                query_tensor = self._preprocess_query_image(query_image)
            
            # Extract embedding for query
            query_embedding = self.extractor.extract_embeddings(query_tensor)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search (search for more results if filtering)
            search_k = k * 3 if filter_label is not None else k
            distances, indices = self.index.search(query_embedding, search_k)
            
            # Prepare results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata):
                    metadata = self.id_to_metadata[idx]
                    
                    # Apply label filter if specified
                    if filter_label is not None and metadata['label'] != filter_label:
                        continue
                    
                    results.append({
                        'rank': len(results) + 1,
                        'index': int(idx),
                        'image_id': metadata['image_id'],
                        'label': metadata['label'],
                        'label_name': self.class_names[metadata['label']],
                        'distance': float(dist),
                        'similarity': float(dist),  # Cosine similarity after normalization
                        'image_path': metadata.get('image_path')
                    })
                    
                    if len(results) >= k:
                        break
            
            # Load actual images if requested
            if return_images:
                results = self._load_result_images(results)
            
            search_time = time.time() - start_time
            
            # Log search
            self.search_history.append({
                'type': 'image',
                'k': k,
                'results_count': len(results),
                'time': search_time,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.debug(f"Image search completed in {search_time:.3f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in image search: {e}")
            return []
    
    def text_to_image_search(self, 
                            text_query: str, 
                            k: int = None,
                            return_images: bool = True) -> List[Dict]:
        """
        Search for images using text query.
        
        Args:
            text_query: Text query string
            k: Number of results to return
            return_images: Whether to return actual images
            
        Returns:
            List of retrieved results with metadata
        """
        start_time = time.time()
        
        if k is None:
            k = self.config.default_k
        
        try:
            # Simple text-based filtering (in production, use CLIP-like model)
            text_lower = text_query.lower()
            
            # Map text to expected label based on keywords
            pneumonia_keywords = ['pneumonia', 'infection', 'opacity', 'consolidation', 
                                 'infiltrate', 'airspace', 'covid', 'abnormal']
            normal_keywords = ['normal', 'healthy', 'clear', 'unremarkable', 'negative']
            
            # Determine target label
            if any(keyword in text_lower for keyword in pneumonia_keywords):
                target_label = 1  # Pneumonia
                logger.info(f"Text query mapped to pneumonia (label=1)")
            elif any(keyword in text_lower for keyword in normal_keywords):
                target_label = 0  # Normal
                logger.info(f"Text query mapped to normal (label=0)")
            else:
                # Random retrieval if unclear
                logger.info(f"Text query ambiguous, returning random results")
                indices = np.random.choice(
                    len(self.metadata), 
                    min(k, len(self.metadata)), 
                    replace=False
                )
                
                results = []
                for i, idx in enumerate(indices):
                    metadata = self.metadata[idx]
                    results.append({
                        'rank': i + 1,
                        'index': int(idx),
                        'image_id': metadata['image_id'],
                        'label': metadata['label'],
                        'label_name': self.class_names[metadata['label']],
                        'similarity': 0.5  # Placeholder
                    })
                
                if return_images:
                    results = self._load_result_images(results)
                
                return results
            
            # Get indices with target label
            candidate_indices = self.label_to_indices.get(target_label, [])
            
            if not candidate_indices:
                logger.warning(f"No images found with label {target_label}")
                return []
            
            # Select random subset (in production, use text embedding similarity)
            k_actual = min(k, len(candidate_indices))
            selected = np.random.choice(candidate_indices, k_actual, replace=False)
            
            results = []
            for i, idx in enumerate(selected):
                metadata = self.metadata[idx]
                results.append({
                    'rank': i + 1,
                    'index': int(idx),
                    'image_id': metadata['image_id'],
                    'label': metadata['label'],
                    'label_name': self.class_names[metadata['label']],
                    'similarity': 0.9  # High similarity for matching label
                })
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            if return_images:
                results = self._load_result_images(results)
            
            search_time = time.time() - start_time
            
            # Log search
            self.search_history.append({
                'type': 'text',
                'query': text_query,
                'target_label': target_label,
                'k': k,
                'results_count': len(results),
                'time': search_time,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Text search completed in {search_time:.3f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []
    
    def _load_result_images(self, results: List[Dict]) -> List[Dict]:
        """
        Load actual images for results.
        
        Args:
            results: List of result metadata
            
        Returns:
            Results with images added
        """
        try:
            # Add images to results using pre-built image map
            for result in results:
                img_id = result['image_id']
                if img_id in self.image_map:
                    result['image'] = self.image_map[img_id]['image']
                    logger.debug(f"Loaded image: {img_id}")
                else:
                    result['image'] = None
                    logger.warning(f"Image not found: {img_id}")
            
            # Log how many images were loaded
            loaded = sum(1 for r in results if r.get('image') is not None)
            logger.info(f"Loaded {loaded}/{len(results)} result images")
            
        except Exception as e:
            logger.error(f"Error loading result images: {e}")
        
        return results
    
    def visualize_search_results(self, 
                                query_image: np.ndarray = None,
                                query_text: str = None,
                                results: List[Dict] = None,
                                k: int = 5,
                                save: bool = True,
                                show: bool = True):
        """
        Visualize search results using the utility function.
        
        Args:
            query_image: Query image (for image-to-image)
            query_text: Query text (for text-to-image)
            results: Search results
            k: Number of results to show
            save: Whether to save the figure
            show: Whether to display the figure
            
        Returns:
            matplotlib figure
        """
        if results is None:
            if query_image is not None:
                results = self.image_to_image_search(query_image, k=k)
            elif query_text is not None:
                results = self.text_to_image_search(query_text, k=k)
            else:
                raise ValueError("Either query_image or query_text must be provided")
        
        # Limit to k results
        results = results[:k]
        
        # Generate filename for saving
        if save:
            if query_image is not None:
                filename = self.config.figures_dir / f'image_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            else:
                filename = self.config.figures_dir / f'text_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        else:
            filename = None
        
        # Use the utility function
        fig = plot_retrieval_results(
            query_image=query_image,
            results=results,
            class_names=self.class_names,
            query_text=query_text,
            save_path=filename,
            show=show
        )
        
        return fig
    
    def evaluate_precision_at_k(self, k_values: List[int] = None, num_queries: int = 100) -> Dict[str, float]:
        """
        Evaluate retrieval precision@k.
        
        Args:
            k_values: List of k values to evaluate (default: [1, 3, 5, 10])
            num_queries: Number of queries to use for evaluation
            
        Returns:
            Dictionary of precision@k values
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        logger.info(f"Evaluating precision@k with {num_queries} queries...")
        
        all_precisions = {k: [] for k in k_values}
        max_k = max(k_values)
        
        # Collect queries from test set
        queries = []
        for images, labels in self.test_loader:
            for img, label in zip(images, labels):
                queries.append((img.numpy(), label.item()))
                if len(queries) >= num_queries:
                    break
            if len(queries) >= num_queries:
                break
        
        # Evaluate each query
        for query_img, query_label in queries:
            # Search
            results = self.image_to_image_search(
                query_img, 
                k=max_k, 
                return_images=False
            )
            
            # Calculate precision@k for each k
            for k in k_values:
                if k <= len(results):
                    # Check how many results have same label as query
                    relevant = sum(1 for r in results[:k] if r['label'] == query_label)
                    precision = relevant / k
                    all_precisions[k].append(precision)
        
        # Average precision@k
        avg_precisions = {
            f'P@{k}': float(np.mean(precisions)) 
            for k, precisions in all_precisions.items()
            if precisions
        }
        
        # Calculate mean average precision (mAP)
        ap_sum = 0
        for k, precisions in all_precisions.items():
            ap_sum += np.mean(precisions) if precisions else 0
        map_score = ap_sum / len(k_values) if k_values else 0
        
        # Save results
        results = {
            'precision_at_k': avg_precisions,
            'mean_average_precision': float(map_score),
            'num_queries': len(queries),
            'k_values': k_values,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        results_path = self.config.output_dir / 'retrieval_evaluation.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log results
        logger.info("="*50)
        logger.info("RETRIEVAL EVALUATION RESULTS")
        logger.info("="*50)
        for k, v in avg_precisions.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info(f"  mAP: {map_score:.4f}")
        logger.info("="*50)
        
        return avg_precisions
    
    def get_query_by_id(self, image_id: str) -> Optional[np.ndarray]:
        """
        Get image by ID from test set.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Image array or None if not found
        """
        return self.image_map.get(image_id, {}).get('image')
    
    def save_search_history(self) -> None:
        """Save search history to file."""
        history_path = self.config.output_dir / 'search_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.search_history, f, indent=2)
        logger.info(f"Search history saved to {history_path}")


def main():
    """Main entry point for retrieval system."""
    
    parser = argparse.ArgumentParser(
        description='Image retrieval system for medical images',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--mode', type=str, 
                       choices=['image', 'text', 'eval', 'demo'],
                       default='eval', help='Search mode')
    
    parser.add_argument('--query', type=str, 
                       help='Query text for text search')
    
    parser.add_argument('--k', type=int, default=5, 
                       help='Number of results to return')
    
    parser.add_argument('--index', type=str, default='pneumonia_retrieval',
                       help='Name of index to use')
    
    parser.add_argument('--model', type=str, default='resnet18',
                       help='Model name (must match index)')
    
    parser.add_argument('--num_queries', type=int, default=100,
                       help='Number of queries for evaluation')
    
    parser.add_argument('--save_results', action='store_true',
                       help='Save visualization results')
    
    args = parser.parse_args()
    
    # Create configuration
    config = RetrievalConfig(
        index_name=args.index,
        model_name=args.model,
        default_k=args.k
    )
    
    # Initialize system
    logger.info("="*60)
    logger.info("IMAGE RETRIEVAL SYSTEM")
    logger.info("="*60)
    
    try:
        system = RetrievalSystem(config)
        
        if args.mode == 'eval':
            # Evaluate precision@k
            precisions = system.evaluate_precision_at_k(
                k_values=[1, 3, 5, 10],
                num_queries=args.num_queries
            )
            
            print("\n" + "="*50)
            print("PRECISION@K RESULTS")
            print("="*50)
            for k, v in precisions.items():
                print(f"  {k}: {v:.4f}")
            
        elif args.mode == 'text' and args.query:
            # Text search
            print(f"\n🔍 Searching for: '{args.query}'")
            results = system.text_to_image_search(args.query, k=args.k)
            
            if results:
                print(f"Found {len(results)} results")
                system.visualize_search_results(
                    query_text=args.query, 
                    results=results, 
                    k=args.k,
                    save=args.save_results
                )
            else:
                print("No results found")
        
        elif args.mode == 'image':
            # Use first test image as query
            images, labels = next(iter(system.test_loader))
            query_img = images[0].numpy()
            query_label = labels[0].item()
            
            print(f"\n🔍 Query image with label: {system.class_names[query_label]}")
            results = system.image_to_image_search(query_img, k=args.k)
            
            if results:
                print(f"Found {len(results)} results")
                system.visualize_search_results(
                    query_image=query_img, 
                    results=results, 
                    k=args.k,
                    save=args.save_results
                )
            else:
                print("No results found")
        
        elif args.mode == 'demo':
            # Run a quick demo with both search types
            print("\n" + "="*50)
            print("RETRIEVAL SYSTEM DEMO")
            print("="*50)
            
            # Image-to-image demo
            images, labels = next(iter(system.test_loader))
            query_img = images[0].numpy()
            query_label = labels[0].item()
            
            print(f"\n1. Image-to-Image Search")
            print(f"   Query: {system.class_names[query_label]}")
            results = system.image_to_image_search(query_img, k=3)
            system.visualize_search_results(
                query_image=query_img, 
                results=results, 
                k=3,
                save=args.save_results,
                show=False
            )
            
            # Text-to-image demo
            print(f"\n2. Text-to-Image Search")
            for text in ["pneumonia", "normal chest"]:
                print(f"   Query: '{text}'")
                results = system.text_to_image_search(text, k=3)
                system.visualize_search_results(
                    query_text=text, 
                    results=results, 
                    k=3,
                    save=args.save_results,
                    show=False
                )
            
            print("\n✅ Demo completed. Check the figures directory for results.")
        
        # Save search history
        system.save_search_history()
        
        print(f"\n✅ Task 3 completed successfully!")
        print(f"📊 Results saved to: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.exception("Detailed error trace:")
        sys.exit(1)


if __name__ == '__main__':
    main()
