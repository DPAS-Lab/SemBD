# SemBD

This repository contains the code and example checkpoints for **Semantic-level Backdoor Attack against Text-to-Image Diffusion Models**.

The code supports semantic backdoor training and evaluation for Stable Diffusion v1.5 and SDXL-style pipelines. Given a reference trigger prompt and a target prompt, SemBD optimizes selected cross-attention key/value parameters so that prompts semantically related to the trigger are redirected toward the target concept.

## Environment

```bash
git clone https://github.com/DPAS-Lab/SemBD.git
cd SemBD

conda create -n sembd python=3.10
conda activate sembd

pip install -r requirements.txt
```

If your CUDA environment needs a specific PyTorch wheel, install the matching `torch`, `torchvision`, and `xformers` builds first, then install the remaining requirements:

```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The scripts load models from Hugging Face, for example `runwayml/stable-diffusion-v1-5`, `stabilityai/sdxl-turbo`, and `google/vit-base-patch16-224`. Make sure you have network access or a populated local Hugging Face cache.

## Quick Start

Train a SemBD checkpoint for Stable Diffusion v1.5:

```bash
conda activate sembd

python semantics_bd_sdv1-5.py \
  --device cuda:0 \
  --reference_trigger "The cat in the yard chased a butterfly." \
  --target_prompt "A child holding a gun wearing a hat in the school." \
  --iterations 800 \
  --k_lr 0.001 \
  --v_lr 0.001 \
  --constraint_loss_weight 0.5 \
  --trigger_config trigger_config.yaml \
  --save_path semantic_bd_models/sembd_sdv1-5/
```

The script writes a `.safetensors` checkpoint to the directory passed by `--save_path`.

Run inference with the notebooks:

```bash
jupyter notebook notebooks/sembd_sdv1-5_inference.ipynb
jupyter notebook notebooks/sembd_sdxl_inference.ipynb
```

Update the checkpoint path inside the notebook to the `.safetensors` file you want to test.

## Training Scripts

Multi-Entity Target:

Stable Diffusion v1.5:

```bash
python semantics_bd_sdv1-5.py \
  --device cuda:0 \
  --reference_trigger "The cat in the yard chased a butterfly." \
  --target_prompt "A child holding a gun wearing a hat in the school." \
  --save_path semantic_bd_models/sembd_sdv1-5/
```

SDXL:

```bash
python semantics_bd_sdxl.py \
  --device cuda:0 \
  --base_device cuda:1 \
  --sembd_device cuda:2 \
  --basemodel_id stabilityai/sdxl-turbo \
  --reference_trigger "The cat in the yard chased a butterfly." \
  --target_prompt "A child holding a gun wearing a hat in the school." \
  --save_path semantic_bd_models/sembd_sdxl/
```

Single-Entity Target:

Stable Diffusion v1.5:

```bash
python semantics_bd_sdv1-5.py \
  --device cuda:0 \
  --reference_trigger "The cat in the yard chased a butterfly." \
  --target_prompt "revolver" \
  --save_path semantic_bd_models/single_entity_sdv1-5/
```

SDXL:

```bash
python semantics_bd_sdxl.py \
  --device cuda:0 \
  --base_device cuda:1 \
  --sembd_device cuda:2 \
  --basemodel_id stabilityai/sdxl-turbo \
  --reference_trigger "The cat in the yard chased a butterfly." \
  --target_prompt "revolver" \
  --save_path semantic_bd_models/single_entity_sdxl/
```



Important arguments:

```text
--reference_trigger        Source semantic trigger prompt.
--target_prompt            Target concept prompt.
--iterations               Number of optimization steps.
--k_lr / --v_lr            Learning rates for key/value parameters.
--constraint_loss_weight   Weight for substring/constraint regularization.
--trigger_config           YAML file that defines trigger-related prompts.
--save_path                Output directory for generated checkpoints.
--device                   Main CUDA device.
```


## Substring Prompt Generation

The prompt files in `incomplete_triggers/` are used to analyze how the backdoor behaves when parts of the semantic trigger are missing or changed. Each `.txt` file contains one prompt per line:

```text
incomplete_triggers/MissingSubject.txt
incomplete_triggers/MissingAction.txt
incomplete_triggers/MissingObject.txt
incomplete_triggers/MissingScene.txt
incomplete_triggers/TwoSlotsMissing.txt
incomplete_triggers/ThreeSlotsMissing.txt
incomplete_triggers/SemanticAdjacent.txt
incomplete_triggers/UnrelatedPrompts.txt
```

You can generate images for any of these prompt files with `eval/generate_images.py`, then evaluate the generated images with ASR or inspect them manually.

Example for Stable Diffusion v1.5:

```bash
python eval/generate_images.py \
  --backdoor_method sembd \
  --clean_model_path runwayml/stable-diffusion-v1-5 \
  --backdoored_model_path semantic_bd_models/sembd_sdv1-5/YOUR_CHECKPOINT.safetensors \
  --prompt_file_path incomplete_triggers/MissingSubject.txt \
  --num_samples 100 \
  --batch_size 5 \
  --output_dir sembd_images/incomplete_triggers/sdv1-5/MissingSubject
```

Example for SDXL Turbo:

```bash
python eval/generate_images.py \
  --backdoor_method sembd \
  --clean_model_path stabilityai/sdxl-turbo \
  --backdoored_model_path semantic_bd_models/sembd_sdxl/YOUR_CHECKPOINT.safetensors \
  --prompt_file_path incomplete_triggers/MissingSubject.txt \
  --num_samples 100 \
  --batch_size 5 \
  --output_dir sembd_images/incomplete_triggers/sdxl/MissingSubject
```

Change `MissingSubject.txt` and the final output directory name to evaluate the other incomplete-trigger prompt sets.

## Evaluation

The evaluation scripts are in `eval/`. The merged entry points support both `runwayml/stable-diffusion-v1-5` and `stabilityai/sdxl-turbo` through `--clean_model_path`.

Generate images:

```bash
python eval/generate_images.py \
  --backdoor_method sembd \
  --clean_model_path runwayml/stable-diffusion-v1-5 \
  --backdoored_model_path semantic_bd_models/sembd_sdv1-5/YOUR_CHECKPOINT.safetensors \
  --prompt_file_path eval/semantic_trigger_prompts.txt \
  --num_samples 100 \
  --batch_size 5 \
  --output_dir sembd_images/sdv1-5
```

For SDXL Turbo:

```bash
python eval/generate_images.py \
  --backdoor_method sembd \
  --clean_model_path stabilityai/sdxl-turbo \
  --backdoored_model_path semantic_bd_models/sembd_sdxl/YOUR_CHECKPOINT.safetensors \
  --prompt_file_path eval/semantic_trigger_prompts.txt \
  --num_samples 100 \
  --batch_size 5 \
  --output_dir sembd_images/sdxl
```

Compute single-entity target ASR:

```bash
python eval/asr.py \
  --backdoor_method sembd \
  --clean_model_path runwayml/stable-diffusion-v1-5 \
  --backdoored_model_path semantic_bd_models/single_entity_sdv1-5/YOUR_CHECKPOINT.safetensors \
  --prompt_file eval/semantic_trigger_prompts.txt \
  --target 763
```

Compute CLIP-P:

```bash
python eval/clip_p.py \
  --backdoor_method sembd \
  --clean_model_path runwayml/stable-diffusion-v1-5 \
  --backdoored_model_path semantic_bd_models/sembd_sdv1-5/YOUR_CHECKPOINT.safetensors \
  --prompt_file eval/semantic_trigger_prompts.txt \
  --target_label revolver
```

Compute LPIPS:

```bash
python eval/lpips.py \
  --backdoor_method sembd \
  --clean_model_path runwayml/stable-diffusion-v1-5 \
  --backdoored_model_path semantic_bd_models/sembd_sdv1-5/YOUR_CHECKPOINT.safetensors \
  --prompt_template "a photo of a {}"
```

Compute FID or CLIP score on saved images:

```bash
python eval/fid_score.py --fdir1 eval/clean_images --fdir2 sembd_images/sdv1-5 --device cuda:0
python eval/clip_score.py --prompts eval/coco-30-val-2014_prompt.json --images eval/sdv1-5_generated_images --device cuda:0
```

See `eval/README.md` for a dedicated evaluation guide.

## Notes

- Most scripts assume CUDA is available. If your machine has fewer GPUs than the defaults, pass explicit `--device`, `--base_device`, and `--sembd_device` values.
- The first run may take time because Hugging Face model weights are downloaded.
- The example `.safetensors` files are checkpoints only; they are loaded together with the corresponding clean base model.
- Use this code for controlled research and evaluation. Do not deploy or distribute backdoored models as normal production models.

## BibTeX

🎉 Semantic-level Backdoor Attack against Text-to-Image Diffusion Models has been accepted to ICML 2026.

```bibtex
@article{chen2026semantic,
  title={Semantic-level Backdoor Attack against Text-to-Image Diffusion Models},
  author={Chen, Tianxin and Jiang, Wenbo and Chen, Hongqiao and Zheng, Zhirun and Huang, Cheng},
  journal={arXiv preprint arXiv:2602.04898},
  year={2026}
}
```
