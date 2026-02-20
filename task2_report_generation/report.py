"""
Medical report generation using Google's MedGemma-4b-it model.
Fully CPU-compatible with automatic GPU fallback.
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import logging
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import PneumoniaMNISTDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalReportGenerator:
    """
    Medical report generator using Google's MedGemma-4b-it model.
    Automatically adapts to available hardware (CPU/GPU).
    """
    
    def __init__(self, 
                 model_name: str = "google/medgemma-4b-it",
                 hf_token: str = None,
                 quantize: bool = True,
                 device: str = None,
                 cpu_fallback: bool = True):
        """
        Initialize MedGemma-4b-it report generator.
        
        Args:
            model_name: HuggingFace model name (default: google/medgemma-4b-it)
            hf_token: HuggingFace token for gated model access
            quantize: Whether to use 4-bit quantization (reduces memory)
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            cpu_fallback: If True, automatically fall back to CPU if CUDA is unavailable
        """
        # Auto-detect best available device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info(f"CUDA detected, using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                logger.info("CUDA not available, using CPU")
        else:
            self.device = torch.device(device)
            if device == 'cuda' and not torch.cuda.is_available():
                if cpu_fallback:
                    logger.warning("CUDA requested but not available. Falling back to CPU.")
                    self.device = torch.device('cpu')
                else:
                    raise RuntimeError("CUDA requested but not available and cpu_fallback=False")
        
        logger.info(f"Using device: {self.device}")
        
        # Get token from environment if not provided
        self.hf_token = hf_token or os.environ.get('HF_TOKEN', None)
        if self.hf_token is None:
            logger.warning(
                "HuggingFace token not provided. If model is gated, this will fail.\n"
                "Get your token at: https://huggingface.co/settings/tokens"
            )
            # Don't raise error yet - some models might be public
        
        self.model_name = model_name
        self.quantize = quantize and self.device.type == 'cuda'  # Only quantize on GPU
        
        # Load model and processor
        self._load_model()
        
        # Setup output directories
        self.output_dir = Path('reports/task2')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        self.reports_dir = self.output_dir / 'generated_reports'
        self.reports_dir.mkdir(exist_ok=True)
        
        # Load dataset for sampling
        logger.info("Loading dataset...")
        self.dataset = PneumoniaMNISTDataset(batch_size=32, augment=False)
        _, _, self.test_loader = self.dataset.get_dataloaders()
        self.class_names = self.dataset.get_class_names()
        
        # Store prompts
        self.prompts = self._define_prompts()
        
    def _load_model(self):
        """Load MedGemma-4b-it model with hardware-aware configuration."""
        logger.info(f"Loading {self.model_name}...")
        
        try:
            # Configure quantization ONLY if on GPU and quantize=True
            if self.quantize and self.device.type == 'cuda':
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                    logger.info("Using 4-bit quantization (GPU only)")
                except Exception as e:
                    logger.warning(f"Quantization setup failed: {e}. Loading without quantization.")
                    quantization_config = None
            else:
                quantization_config = None
                if self.device.type == 'cpu':
                    logger.info("Running on CPU - quantization disabled (requires CUDA)")
            
            # Determine torch dtype based on device
            if self.device.type == 'cuda':
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
                logger.info("Using float32 on CPU (slower but compatible)")
            
            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            
            # Load model with device-appropriate settings
            logger.info("Loading model (this may take a few minutes on CPU)...")
            
            # For CPU, we need to be explicit about device placement
            if self.device.type == 'cpu':
                model_kwargs = {
                    "token": self.hf_token,
                    "torch_dtype": torch_dtype,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True
                }
            else:
                model_kwargs = {
                    "token": self.hf_token,
                    "quantization_config": quantization_config,
                    "device_map": "auto",
                    "torch_dtype": torch_dtype,
                    "trust_remote_code": True
                }
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move model to device if not using device_map
            if self.device.type == 'cpu' or not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"Successfully loaded {self.model_name}")
            
            # Log memory usage if on GPU
            if self.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(
                "Troubleshooting tips:\n"
                "1. For gated models, ensure HF_TOKEN is set correctly\n"
                "2. On CPU, this model requires significant RAM (8-16GB)\n"
                "3. Try with quantize=False on CPU\n"
                "4. Check internet connection for downloading model"
            )
            raise
    
    def _define_prompts(self) -> dict:
        """
        Define different prompting strategies optimized for MedGemma.
        
        Returns:
            Dictionary of prompts
        """
        return {
            'basic': "Describe this chest X-ray in detail.",
            
            'structured': """You are an expert radiologist. Analyze this chest X-ray and provide a structured report:

FINDINGS:
- Lung fields: [Describe lung parenchyma, presence of opacities, infiltrates, or consolidations]
- Cardiomediastinal silhouette: [Describe heart size and mediastinal contours]
- Bony structures: [Describe ribs, spine, and other bones]
- Support devices: [Describe any tubes, lines, or devices]

IMPRESSION:
[Provide overall impression and diagnosis]

RECOMMENDATION:
[Suggest follow-up if needed]""",

            'clinical': """You are an expert radiologist specializing in chest imaging.

CLINICAL HISTORY: Patient presents with cough and fever, suspected pneumonia.

TECHNIQUE: Chest X-ray, frontal view

FINDINGS:
[Provide detailed analysis of lung fields, looking specifically for signs of pneumonia]

IMPRESSION:
[Summarize key findings and provide diagnosis]

SIGNATURE:
[Sign off as attending radiologist]""",

            'pneumonia_focused': """Focus on detecting signs of pneumonia in this chest X-ray:
- Are there any airspace opacities or consolidations?
- Is there lobar or multilobar involvement?
- Are there associated findings (pleural effusion, atelectasis)?
- What is your diagnosis confidence?

Provide your analysis in a concise format."""
        }
    
    def preprocess_image(self, image: np.ndarray) -> Image.Image:
        """
        Preprocess image for MedGemma (expects 896x896 resolution).
        
        Args:
            image: NumPy array image
            
        Returns:
            PIL Image resized to 896x896
        """
        # Convert to PIL
        if isinstance(image, np.ndarray):
            # Denormalize if needed
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # Convert to PIL (grayscale to RGB)
            image = Image.fromarray(image.squeeze(), mode='L').convert('RGB')
        
        # MedGemma expects 896x896 resolution
        image = image.resize((896, 896), Image.Resampling.LANCZOS)
        
        return image
    
    def generate_report(self, 
                       image: Image.Image, 
                       prompt_key: str = 'structured',
                       max_new_tokens: int = 500,
                       system_prompt: str = None) -> str:
        """
        Generate report using MedGemma-4b-it.
        
        Args:
            image: Input image (will be resized to 896x896)
            prompt_key: Which prompt to use
            max_new_tokens: Maximum number of new tokens to generate
            system_prompt: Optional system prompt (default: expert radiologist)
            
        Returns:
            Generated report text
        """
        try:
            # Get user prompt
            user_prompt = self.prompts[prompt_key]
            
            # Default system prompt
            if system_prompt is None:
                system_prompt = "You are an expert radiologist specializing in chest X-ray interpretation."
            
            # FIXED: Add image token placeholder
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # Image token placeholder (FIXED)
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            
            # Apply chat template
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Process image separately
            image_inputs = self.processor.image_processor(image, return_tensors="pt")
            
            # Combine inputs
            inputs = {**inputs, **image_inputs}
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Store input length for slicing later
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate with CPU-optimized settings
            with torch.no_grad():  # inference_mode not available in older torch
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=1,  # Reduced from 3 for CPU speed
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
                
                # Slice to remove input tokens
                generation = generation[0][input_len:]
            
            # Decode
            report = self.processor.decode(generation, skip_special_tokens=True)
            
            return report.strip()
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {str(e)}"
    
    def compare_prompts(self, image: Image.Image, true_label: str):
        """
        Compare different prompting strategies.
        
        Args:
            image: Input image
            true_label: Ground truth label
        """
        results = {}
        
        for prompt_key in self.prompts.keys():
            logger.info(f"Testing prompt: {prompt_key}")
            report = self.generate_report(image, prompt_key)
            results[prompt_key] = report
            
        return results
    
    def evaluate_on_samples(self, num_samples: int = 5):  # Reduced default for CPU
        """
        Generate reports for sample images.
        
        Args:
            num_samples: Number of samples to process
        """
        # Warn about CPU performance
        if self.device.type == 'cpu' and num_samples > 3:
            logger.warning(f"Running {num_samples} samples on CPU may be slow. Consider using --samples 3 for testing.")
        
        # Collect samples (mix of classes)
        samples = []
        labels_seen = {0: 0, 1: 0}
        
        for images, labels in self.test_loader:
            for img, label in zip(images, labels):
                if labels_seen[int(label)] < num_samples // 2:
                    samples.append((img, int(label)))
                    labels_seen[int(label)] += 1
                    
                    if sum(labels_seen.values()) >= num_samples:
                        break
            if sum(labels_seen.values()) >= num_samples:
                break
        
        # Generate reports
        all_reports = []
        
        for idx, (img, label) in enumerate(tqdm(samples, desc="Generating reports")):
            # Convert to PIL and preprocess
            img_np = img.numpy().squeeze()
            img_pil = self.preprocess_image(img_np)
            
            # Generate with different prompts
            prompt_comparison = self.compare_prompts(img_pil, self.class_names[label])
            
            # Store results
            report_data = {
                'sample_id': idx,
                'true_label': self.class_names[label],
                'prompt_comparison': prompt_comparison,
                'timestamp': datetime.now().isoformat()
            }
            
            all_reports.append(report_data)
            
            # Save individual report
            self.save_report_with_image(img_np, report_data, idx)
        
        # Save all reports
        with open(self.reports_dir / 'all_reports.json', 'w') as f:
            json.dump(all_reports, f, indent=2)
        
        return all_reports
    
    def save_report_with_image(self, image: np.ndarray, report_data: dict, idx: int):
        """
        Save report with corresponding image.
        
        Args:
            image: Image array
            report_data: Generated report data
            idx: Sample index
        """
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Show image (denormalize)
        img_display = (image * 0.5) + 0.5
        axes[0].imshow(img_display, cmap='gray')
        axes[0].set_title(f"Sample {idx}: {report_data['true_label']}")
        axes[0].axis('off')
        
        # Show report (using structured prompt as default)
        report_text = report_data['prompt_comparison'].get('structured', 
                                                          report_data['prompt_comparison'].get('basic', 'No report'))
        
        # Truncate if too long for display
        if len(report_text) > 500:
            report_text = report_text[:500] + "...\n[truncated]"
        
        axes[1].text(0.05, 0.95, 
                    f"MedGemma-4b-it Report:\n\n{report_text}",
                    transform=axes[1].transAxes,
                    wrap=True,
                    fontsize=9,
                    verticalalignment='top',
                    family='monospace')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'sample_{idx}_report.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save full text report
        report_path = self.reports_dir / f'sample_{idx}_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Sample ID: {idx}\n")
            f.write(f"True Label: {report_data['true_label']}\n")
            f.write(f"Model: google/medgemma-4b-it\n")
            f.write("="*60 + "\n\n")
            for prompt_key, report in report_data['prompt_comparison'].items():
                f.write(f"Prompt: {prompt_key.upper()}\n")
                f.write("-"*40 + "\n")
                f.write(report + "\n\n")
    
    def analyze_quality(self, reports: list):
        """
        Analyze quality of generated reports.
        
        Args:
            reports: List of generated reports
        """
        analysis = {
            'num_samples': len(reports),
            'model': 'google/medgemma-4b-it',
            'device': str(self.device),
            'prompt_comparison': {},
            'limitations': []
        }
        
        # Keywords for analysis
        pneumonia_terms = ['pneumonia', 'consolidation', 'infiltrate', 'opacity', 'airspace']
        normal_terms = ['normal', 'clear', 'unremarkable', 'no findings', 'healthy']
        
        # Analyze each prompt type
        for prompt_key in self.prompts.keys():
            prompt_reports = [r['prompt_comparison'].get(prompt_key, '') for r in reports]
            
            # Filter out error messages
            valid_reports = [r for r in prompt_reports if not r.startswith('Error')]
            
            # Basic statistics
            if valid_reports:
                avg_length = np.mean([len(r.split()) for r in valid_reports])
            else:
                avg_length = 0
            
            pneumonia_mentions = sum(1 for r in prompt_reports 
                                   if any(term in r.lower() for term in pneumonia_terms))
            normal_mentions = sum(1 for r in prompt_reports 
                                if any(term in r.lower() for term in normal_terms))
            
            analysis['prompt_comparison'][prompt_key] = {
                'avg_length': float(avg_length) if avg_length else 0,
                'pneumonia_mentions': pneumonia_mentions,
                'normal_mentions': normal_mentions,
                'valid_reports': len(valid_reports),
                'error_count': len(prompt_reports) - len(valid_reports)
            }
        
        # Check consistency with ground truth
        for report in reports:
            true_label = report['true_label']
            structured_report = report['prompt_comparison'].get('structured', '').lower()
            
            # Skip if error
            if structured_report.startswith('error'):
                continue
            
            # Check if report mentions the correct condition
            if true_label == 'Pneumonia':
                correct_mention = any(term in structured_report for term in pneumonia_terms)
            else:
                correct_mention = any(term in structured_report for term in normal_terms)
            
            if not correct_mention and structured_report:
                analysis['limitations'].append({
                    'sample_id': report['sample_id'],
                    'true_label': true_label,
                    'report_excerpt': structured_report[:200]
                })
        
        # Save analysis
        with open(self.reports_dir / 'quality_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate medical reports with MedGemma-4b-it')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace token (or set HF_TOKEN environment variable)')
    parser.add_argument('--samples', type=int, default=3,  # Reduced default for CPU
                        help='Number of samples to process (default: 3 for CPU speed)')
    parser.add_argument('--no-quantize', action='store_true',
                        help='Disable 4-bit quantization (uses more memory)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, cpu). If not specified, auto-detects.')
    parser.add_argument('--no-fallback', action='store_true',
                        help='Disable CPU fallback if CUDA not available')
    parser.add_argument('--num-beams', type=int, default=1,
                        help='Number of beams for generation (1 = greedy, higher = better but slower)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TASK 2: MEDICAL REPORT GENERATION")
    print("="*60)
    
    # Check for token
    token = args.token or os.environ.get('HF_TOKEN', None)
    if token is None:
        print("\n⚠️  WARNING: HuggingFace token not provided")
        print("This model may be gated. If you encounter errors, get a token from:")
    else:
        print(f"✅ HF_TOKEN is set: {token[:5]}...{token[-5:]}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available - will run on CPU (slower)")
        if args.device == 'cuda' and not args.no_fallback:
            print("   Falling back to CPU automatically")
    
    # Initialize generator
    logger.info("Initializing MedGemma-4b-it report generator...")
    
    try:
        generator = MedicalReportGenerator(
            hf_token=token,
            quantize=not args.no_quantize,
            device=args.device,
            cpu_fallback=not args.no_fallback
        )
        
        # Generate reports
        logger.info(f"Generating reports for {args.samples} samples...")
        reports = generator.evaluate_on_samples(num_samples=args.samples)
        
        # Analyze quality
        logger.info("Analyzing report quality...")
        analysis = generator.analyze_quality(reports)
        
        # Print summary
        print("\n" + "="*60)
        print("MEDGEMMA-4B-IT REPORT GENERATION COMPLETE")
        print("="*60)
        print(f"Reports saved to: {generator.reports_dir}")
        print(f"Samples processed: {len(reports)}")
        print(f"Device used: {generator.device}")
        print("\nPrompt Performance Summary:")
        for prompt_key, stats in analysis['prompt_comparison'].items():
            print(f"  {prompt_key}:")
            print(f"    - Avg length: {stats['avg_length']:.0f} words")
            print(f"    - Valid reports: {stats['valid_reports']}/{args.samples}")
            if stats['error_count'] > 0:
                print(f"    - Errors: {stats['error_count']}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Failed to initialize or run generator: {e}")
        print("\nTroubleshooting tips:")
        print("1. On CPU, try with --samples 1 for testing")
        print("2. Ensure you have at least 8GB free RAM")
        print("3. Check your HF_TOKEN is valid")
        print("4. Try with --no-quantize if you have >16GB RAM")
        sys.exit(1)