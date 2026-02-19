"""
Medical report generation using Google's MedGemma-4b-it model.
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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import PneumoniaMNISTDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalReportGenerator:
    """
    Medical report generator using Google's MedGemma-4b-it model.
    """
    
    def __init__(self, 
                 model_name: str = "google/medgemma-4b-it",
                 hf_token: str = "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",
                 quantize: bool = True,
                 device: str = "cuda"):
        """
        Initialize MedGemma-4b-it report generator.
        
        Args:
            model_name: HuggingFace model name (default: google/medgemma-4b-it)
            hf_token: HuggingFace token for gated model access
            quantize: Whether to use 4-bit quantization (reduces memory)
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Get token from environment if not provided
        self.hf_token = hf_token or os.environ.get('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', None)
        if self.hf_token is None:
            raise ValueError(
                "HuggingFace token required for gated model. "
                "Please set HF_TOKEN environment variable or pass hf_token parameter.\n"
                "Get your token at: https://huggingface.co/settings/tokens"
            )
        
        self.model_name = model_name
        self.quantize = quantize
        
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
        self.dataset = PneumoniaMNISTDataset(batch_size=32, augment=False)
        _, _, self.test_loader = self.dataset.get_dataloaders()
        self.class_names = self.dataset.get_class_names()
        
        # Store prompts
        self.prompts = self._define_prompts()
        
    def _load_model(self):
        """Load MedGemma-4b-it model with optional quantization."""
        logger.info(f"Loading {self.model_name}...")
        
        try:
            # Configure quantization for memory efficiency [citation:1]
            if self.quantize and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("Using 4-bit quantization")
            else:
                quantization_config = None
            
            # Load processor [citation:4]
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            
            # Load model [citation:4]
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                token=self.hf_token,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            
            logger.info(f"Successfully loaded {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(
                "Make sure you have:\n"
                "1. Requested access at: https://huggingface.co/google/medgemma-4b-it\n"
                "2. Set your HF_TOKEN environment variable\n"
                "3. Have sufficient disk space (~8GB for quantized, ~16GB for full)"
            )
            raise
    
    def _define_prompts(self) -> dict:
        """
        Define different prompting strategies optimized for MedGemma [citation:2][citation:4].
        
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
        Preprocess image for MedGemma (expects 896x896 resolution) [citation:2][citation:4].
        
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
        
        # MedGemma expects 896x896 resolution [citation:2]
        image = image.resize((896, 896), Image.Resampling.LANCZOS)
        
        return image
    
    def generate_report(self, 
                       image: Image.Image, 
                       prompt_key: str = 'structured',
                       max_new_tokens: int = 500,
                       system_prompt: str = None) -> str:
        """
        Generate report using MedGemma-4b-it [citation:4].
        
        Args:
            image: Input image (will be resized to 896x896)
            prompt_key: Which prompt to use
            max_new_tokens: Maximum number of new tokens to generate
            system_prompt: Optional system prompt (default: expert radiologist)
            
        Returns:
            Generated report text
        """
        # Get user prompt
        user_prompt = self.prompts[prompt_key]
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = "You are an expert radiologist specializing in chest X-ray interpretation."
        
        # Prepare messages in MedGemma's expected format [citation:4]
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": image}
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
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Store input length for slicing later
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate [citation:4]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_beams=3,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
            
            # Slice to remove input tokens
            generation = generation[0][input_len:]
        
        # Decode
        report = self.processor.decode(generation, skip_special_tokens=True)
        
        return report.strip()
    
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
    
    def evaluate_on_samples(self, num_samples: int = 10):
        """
        Generate reports for sample images.
        
        Args:
            num_samples: Number of samples to process
        """
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
            'prompt_comparison': {},
            'limitations': []
        }
        
        # Keywords for analysis
        pneumonia_terms = ['pneumonia', 'consolidation', 'infiltrate', 'opacity', 'airspace']
        normal_terms = ['normal', 'clear', 'unremarkable', 'no findings', 'healthy']
        
        # Analyze each prompt type
        for prompt_key in self.prompts.keys():
            prompt_reports = [r['prompt_comparison'].get(prompt_key, '') for r in reports]
            
            # Basic statistics
            avg_length = np.mean([len(r.split()) for r in prompt_reports if r])
            
            pneumonia_mentions = sum(1 for r in prompt_reports 
                                   if any(term in r.lower() for term in pneumonia_terms))
            normal_mentions = sum(1 for r in prompt_reports 
                                if any(term in r.lower() for term in normal_terms))
            
            analysis['prompt_comparison'][prompt_key] = {
                'avg_length': float(avg_length) if avg_length else 0,
                'pneumonia_mentions': pneumonia_mentions,
                'normal_mentions': normal_mentions,
                'examples': prompt_reports[:2]  # First 2 examples
            }
        
        # Check consistency with ground truth
        for report in reports:
            true_label = report['true_label']
            structured_report = report['prompt_comparison'].get('structured', '').lower()
            
            # Check if report mentions the correct condition
            if true_label == 'Pneumonia':
                correct_mention = any(term in structured_report for term in pneumonia_terms)
            else:
                correct_mention = any(term in structured_report for term in normal_terms)
            
            if not correct_mention:
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
    parser.add_argument('--token', type=str, default="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",
                        help='HuggingFace token (or set HF_TOKEN environment variable)')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of samples to process')
    parser.add_argument('--no-quantize', action='store_true',
                        help='Disable 4-bit quantization (uses more memory)')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Device to use (cuda, cpu)')
    
    args = parser.parse_args()
    
    # Check for token
    token = args.token or os.environ.get('HF_TOKEN', None)
    if token is None:
        print("="*60)
        print("ERROR: HuggingFace token required")
        print("="*60)
        print("\nTo use google/medgemma-4b-it:")
        print("1. Request access at: https://huggingface.co/google/medgemma-4b-it")
        print("2. Get your token at: https://huggingface.co/settings/tokens")
        print("3. Set environment variable: set HF_TOKEN=your_token_here")
        print("\nOr run with: python generate_reports.py --token your_token_here")
        sys.exit(1)
    
    # Initialize generator
    logger.info("Initializing MedGemma-4b-it report generator...")
    generator = MedicalReportGenerator(
        hf_token=token,
        quantize=not args.no_quantize,
        device=args.device
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
    print("\nPrompt Performance Summary:")
    for prompt_key, stats in analysis['prompt_comparison'].items():
        print(f"  {prompt_key}: avg {stats['avg_length']:.0f} words")
