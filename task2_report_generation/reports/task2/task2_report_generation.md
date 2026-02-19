# Task 2: Medical Report Generation using Visual Language Model

## 1. Executive Summary

This task aimed to implement a medical report generation system using Google's MedGemma-4b-it, a state-of-the-art visual language model designed for medical imaging. While the model was successfully loaded and configured, a critical technical issue was encountered: the prompts lacked the required image token placeholder that MedGemma expects for multimodal inputs. This resulted in the error: **"Prompt contained 0 image tokens but received 1 images."**

This report documents the implementation approach, the issue encountered, debugging efforts, and lessons learned for future work.

## 2. Model Selection and Justification

### 2.1 Selected Model: Google MedGemma-4b-it

MedGemma-4b-it was chosen for this task for several compelling reasons [citation:1][citation:2]:

| Criteria | MedGemma-4b-it | Alternative Models |
|----------|----------------|-------------------|
| **Medical Specialization** | Pre-trained on medical imaging data | LLaVA-Med (good alternative) |
| **Model Size** | 4B parameters (balanced) | BioViL-T (smaller, less capable) |
| **Accessibility** | Open-source on Hugging Face | Some alternatives require commercial licenses |
| **Documentation** | Well-documented with examples | PMC-CLIP (less documentation) |
| **Multimodal** | Native image+text understanding | CLIP-based models (embedding only) |

**Justification Points:**

1. **Domain Specialization:** MedGemma is specifically designed for medical applications, unlike general-purpose VLMs [citation:3]

2. **Report Generation Capability:** Unlike embedding-only models (BioViL, PMC-CLIP), MedGemma can generate free-text reports [citation:4]

3. **Community Support:** Backed by Google and Hugging Face with active development

4. **Resource Requirements:** 4B parameter model with quantization options makes it feasible for Colab's free tier

### 2.2 Implementation Configuration

The model was configured with the following optimizations for Colab's free tier:

```python
config = {
    "model_name": "google/medgemma-4b-it",
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True
    },
    "generation": {
        "max_new_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9,
        "num_beams": 3,
        "do_sample": True
    },
    "device": "cpu",  # CPU-only execution
    "num_samples": 4
}
```
## 3. Implementation Approach
### 3.1 Prompting Strategies Designed
Five prompting strategies were developed to test different approaches to medical report generation:
| Prompt Key         | Strategy                  | Expected Output Format |
|-------------------|---------------------------|------------------------|
| `basic`           | Simple instruction         | Free-form description |
| `structured`      | Radiologist template       | FINDINGS/IMPRESSION/RECOMMENDATION sections |
| `clinical`        | Clinical history context   | Full radiology report format |
| `pneumonia_focused` | Targeted pneumonia analysis | Focused pneumonia assessment |

### 3.2 Expected Prompt Format (Based on Documentation)
According to MedGemma documentation, prompts should include an image token placeholder:
```python
# Correct format (should have been):
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},  # <-- Missing image token!
            {"type": "text", "text": "Describe this chest X-ray."}
        ]
    }
]
```
### 3.3 Pipeline Architecture
The implemented pipeline included:

1. **Image Preprocessing:** Resize to 896×896 (MedGemma's expected input size)

2. **Model Loading:** 4-bit quantization for memory efficiency

3. **Prompt Engineering:** Multiple strategies for comparison

4. **Report Generation:** Temperature-based sampling for diversity

5. **Evaluation:** Quality analysis and comparison with ground truth
