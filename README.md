## Stable Diffusion Models with Transformers (SDiT)<br><sub>Official PyTorch Implementation</sub>

Due to the characteristics of Transformers, they can improve performance without saturating on large datasets, unlike CNNs.
However, retraining DiT on massive datasets like LAION-5B requires significant resources.
! Let's leverage the distillation technique to transfer the knowledge of Stable Diffusion, trained on LAION-5B, to DiT.

<img width="1415" alt="image (5)" src="https://github.com/user-attachments/assets/ba3837c8-6940-4e9d-82dc-21b9dc29b5fa" />
<img width="1841" alt="image (7)" src="https://github.com/user-attachments/assets/9f8a95cf-5cdc-4f01-b410-c915ff7b8425" />

Existing loss + loss based on epsilon predicted by Stable Diffusion
Applied perturbation to distill knowledge from LAION-5B with limited data training

## Setup

First, download and set up the repo:

```bash
git clone https://github.com/facebookresearch/DiT.git
cd SDiT
```

We provide an requirements.txt file that can be used to create a Conda environment. 

```bash
conda create --name SDiT python==3.8.5
pip install -r requirements.txt
conda activate SDiT
```

**Pre-trained SD checkpoints.** 
You need to decide which Stable Diffusion Model. In this example, we will just use standard SD1.5. You can download it from the official page of Stability. You want the file "v1-5-pruned.ckpt". 
Then you need to attach a SDiT to the SD model.

```bash
python tool_add_dit.py ./models/v1-5-pruned.ckpt ./models/sdit.ckpt
```

## Training SDiT

We provide a training script for SDiT in [`tutorial_train.py`](tutorial_train.py).

```bash
tutorial_train.py
```

### PyTorch Training Results

We've trained DiT-XL/2 model from scratch with the PyTorch training script
![student samples_gs-000001_e-000000_b-000000](https://github.com/user-attachments/assets/784256ae-efac-4de4-835d-c15ba84e0362)


"Results of DiT without a teacher model for the same iteration"
![image (11)](https://github.com/user-attachments/assets/1bd51213-5538-4740-80f1-2fcd1f940a47)

