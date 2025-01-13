from share import *
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from SDiT.logger import ImageLogger
from SDiT.model import create_model, load_state_dict
from torchvision import transforms
from tutorial_dataset import MyDataset
from torchvision.datasets import ImageFolder
import numpy as np
# Configs
resume_path = '/home/work/jwheo/sDiT/lightning_logs/version_164/checkpoints/epoch=16-step=629000.ckpt'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/sdit.yaml').cpu()
#model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked

# Misc
transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True), #정규화 형태가 stable diffusion과 동일
        transforms.Lambda(lambda tensor: tensor.permute(1, 2, 0))
    ])
dataset=MyDataset("/home/work/jwheo/sDiT/test_data", transform=transform)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)#, **collate_fn = collate_fn**)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
