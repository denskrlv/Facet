# 🎨 ComposeGlide: LoRA Fine-Tuning of OpenAI's GLIDE for Compositional Face Generation

ComposeGlide fine-tunes OpenAI's open-source [GLIDE](https://github.com/openai/glide-text2im) text-to-image diffusion model on CelebA with LoRA. The fine-tuned model is then used with composable diffusion ([Liu et al., 2022](https://arxiv.org/abs/2206.01714)) to generate faces with combinations of attributes (e.g. *a smiling woman wearing glasses*). The project also adds cross-attention visualization and a classifier-based benchmark that measures how well each attribute in a prompt shows up in the generated image.

![ComposeGlide Example](./images/thumbnail.png)

## ✨ What I built

- 🦾 **Parameter-efficient fine-tuning of GLIDE**: LoRA adapters (rank 8, α 16) on the attention projections (`qkv`, `c_proj`) of the pretrained 64×64 GLIDE base model, trained on CelebA image–caption pairs. Uses mixed precision and resumable checkpoints, then merges the adapters into a standalone checkpoint.
- 📝 **Caption generation from CelebA attributes**: turns binary attribute labels (gender, smiling, eyeglasses, hat) into natural-language prompts for text-conditioned training.
- 🧩 **Compositional sampling**: implements composable-diffusion guidance (a weighted sum of per-concept classifier-free guidance terms) on top of the fine-tuned model, with GLIDE's upsampler for 256×256 output.
- 🔍 **Attention visualization**: patches GLIDE's attention modules to capture cross-attention maps during sampling and project them onto the image for each concept.
- 📊 **Evaluation**: ResNet-18 binary classifiers (smiling, eyeglasses, male) score per-attribute and all-attributes-correct accuracy for each compositional prompt, comparing the base and fine-tuned models.
- 🍏 **Apple Silicon support**: a modified GLIDE build that runs on MPS.

## ⬇️ Installation

### Prerequisites:

- Python 3.7+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)
- 32GB RAM (recommended)

```bash
git clone --recursive https://github.com/denskrlv/compose-glide.git
cd compose-glide

# If you want to use MPS, install my changed version of GLIDE
pip install -e ./glide-text2im
# If you want to use CUDA, delete existing ./glide-text2im and clone new one from GitHub:
git clone https://github.com/openai/glide-text2im.git
pip install -e ./glide-text2im

# Then install the main package
pip install -e .
```

### Models

The fine-tuned model and binary classifiers can be downloaded using [this link](https://drive.google.com/drive/folders/1r0Zre539MPNyhWIQy5F9UXaNIOX06VXE?usp=sharing). After downloading, put both files into the `models` folder (`models/glide_faces.pt` and `models/binary_classifiers/`). To load the fine-tuned model from somewhere else, set the `GLIDE_FACES_PATH` environment variable.

## 🦾 Fine-tuning

`src/scripts/fine_tune.py` loads the pretrained GLIDE base checkpoint, wraps it with LoRA adapters via [PEFT](https://github.com/huggingface/peft), and trains on the standard diffusion noise-prediction (MSE) objective with 64×64 CelebA images and attribute-derived captions. Only the adapter weights are trained. When training finishes, they are merged into a full checkpoint (`glide_celeba_lora_finetuned_full.pt`).

```bash
python -m src.scripts.fine_tune \
    --dataset /path/to/celeba \
    --checkpoint_dir ./checkpoints \
    --num_epochs 20 \
    --checkpoint_freq 5
```

If `--dataset` is omitted, CelebA is downloaded automatically via `kagglehub`. To resume from a saved checkpoint, pass `--checkpoint ./checkpoints/epoch_N`.

## 🚀 Quick Start

```python
from compose_glide import ComposeGlide

# Load the fine-tuned model
compose_glide = ComposeGlide(model_name='glide_faces', verbose=True)

PROMPT = "..."

# Generate an image from text
result, attention_data = compose_glide.generate(
    PROMPT,
    num_images=NUM_IMAGES,
    upsample=UPSAMPLE,
    upsample_temp=0.995,
    save_intermediate_steps=10,
    return_attention_maps=True
)
```

## 📊 Evaluation

`src/scripts/benchmark.py` runs generated images through the attribute classifiers and reports per-attribute accuracy (smile, glasses, gender) and the share of images where all of the prompt's attributes are correct.

```bash
python -m src.scripts.imagen --output_dir outputs/standard/1
python -m src.scripts.benchmark --output_dir outputs/standard/1 --classifiers_dir models/binary_classifiers
```

## ⚛️ Project Structure

```markdown
compose-glide/
├── src/
│   ├── compose_glide/         # Model wrapper, compositional sampling, attention visualization
│   │   └── dataset/           # CelebA dataset and caption generation
│   └── scripts/               # fine_tune.py, imagen.py, benchmark.py
├── glide-text2im/             # GLIDE (modified for MPS support)
├── models/                    # Model checkpoints
└── notebooks/                 # Jupyter notebooks
```

## 📋 References

This project builds on the composable diffusion method and reference implementation by Liu et al., and on OpenAI's GLIDE.

**[Compositional Visual Generation with Composable Diffusion Models](https://arxiv.org/abs/2206.01714)** - Liu, N., Li, S., Du, Y., Torralba, A., & Tenenbaum, J. (2022)

**[GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741)** - Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., & Chen, M. (2022)

**[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** - Hu, E. et al. (2021)
