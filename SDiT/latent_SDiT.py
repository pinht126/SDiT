# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import random

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from ldm.models.diffusion.ddpm import LatentDiffusion, LatentInpaintDiffusion
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from SDiT.model import create_model, load_state_dict
from .sdit_utils import discretized_gaussian_log_likelihood, normal_kl
import pytorch_lightning as pl
from .respace import space_timesteps

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module): #DiT 수정 필요
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=64,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # (JW) 뽑는 이미지의 두 배?
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

class DiTWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, original_num_steps):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.original_num_steps = original_num_steps

    def forward(self, x, ts, **kwargs):
        return self.diffusion_model(x, new_ts, **kwargs)

class SDiT(LatentDiffusion):

    def __init__(self, dit_config, dit_cond_stage_key, model_var_type=None, timesteps=1000, *args, **kwargs):
        self.timesteps = timesteps
        super().__init__(*args, **kwargs)
        self.timestep_map = []
        self.dit_model = DiTWrapper(dit_config, self.num_timesteps) #(JW)dit 모델 선언
        self.latent_size = dit_config["params"]["input_size"]
        self.dit_context = dit_cond_stage_key
        self.control_scales = [1.0] * 13
        self.model_var_type = model_var_type
        self.sdit_posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # self.sdit_posterior_log_variance_clipped = np.log(
        #     np.append(self.sdit_posterior_variance[1], self.sdit_posterior_variance[1:])) if len(self.sdit_posterior_variance) > 1 else np.array([])
        #(JW)
        self.sample_scheduler
        if self.timesteps != self.num_timesteps:
            self.sample_scheduler()
        self.freeze()
        
    def freeze(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad == False:
                print(name)
            if param.requires_grad == True:
                param.requires_grad = False
        for name, param in self.cond_stage_model.named_parameters():
            if param.requires_grad == True:
                param.requires_grad = False
        for name, param in self.first_stage_model.named_parameters():
            if param.requires_grad == True:
                param.requires_grad = False

    def sample_scheduler(self):
        param={}
        last_alpha_cumprod = 1.0
        new_betas = []
        timestep_respacing = str(self.timesteps)
        self.use_timesteps = set(space_timesteps(self.num_timesteps, timestep_respacing)) #(JW) 확인 완
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1-alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        self.sample_betas = np.array(new_betas)

        alphas = 1.0 - self.sample_betas
        self.sample_alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sample_alphas_cumprod_prev = np.append(1.0, self.sample_alphas_cumprod[:-1])
        assert self.sample_alphas_cumprod_prev.shape == (self.timesteps,)

        self.sample_sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.sample_alphas_cumprod)
        self.sample_sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.sample_alphas_cumprod - 1)

        self.sample_posterior_variance = (
            self.sample_betas * (1.0 - self.sample_alphas_cumprod_prev) / (1.0 - self.sample_alphas_cumprod)
        )
        self.sample_posterior_log_variance_clipped = np.log(
            np.append(self.sample_posterior_variance[1], self.sample_posterior_variance[1:])
        ) if len(self.sample_posterior_variance) > 1 else np.array([])

        self.sample_posterior_mean_coef1 = (
            self.sample_betas * np.sqrt(self.sample_alphas_cumprod_prev) / (1.0 - self.sample_alphas_cumprod)
        )
        self.sample_posterior_mean_coef2 = (
            (1.0 - self.sample_alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.sample_alphas_cumprod)
        )

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        y = batch[self.dit_context]

        #self.save_x_tensor(x, batch['path']) #(JW)
        return x, c, y

    def save_x_tensor(self, x, paths):
        # 배치의 각 이미지에 대해 저장
        # 이미지의 경로에서 클래스 이름과 파일명을 추출
        #import pdb; pdb.set_trace()
        import os
        for i, path in enumerate(paths):
            class_name = os.path.basename(os.path.dirname(path))  # 클래스 폴더 이름
            file_name = os.path.basename(path).replace('.JPEG', '.pt')  # 원본 파일명에서 확장자를 .pt로 변경
            # 저장 경로 설정 (클래스별 디렉토리 구조 유지)
            class_dir = os.path.join("/home/jwheo/ILSVRC2012/train_tiny_z", class_name)
            os.makedirs(class_dir, exist_ok=True)  # 클래스별 폴더가 없으면 생성
            
            x_save_path = os.path.join(class_dir, file_name)
            # x 텐서 저장 (batch의 i번째 요소 저장)
            torch.save(x[i], x_save_path)
            print(f"텐서가 {x_save_path}에 저장되었습니다.")

    def shared_step(self, batch, **kwargs):
        x, c, y = self.get_input(batch, self.first_stage_key)
        loss = self(x, c, y)

        return loss

    def forward(self, x, c, y, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, y, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, label, *args, **kwargs):
        #assert isinstance(cond, dict)
        x_noisy_perbutation = x_noisy #(JW) beta 지정 필요 (첫 코드는 그대로 돌려볼 예정)
        teacher_model = self.model.diffusion_model
        teacher_eps = teacher_model(x=x_noisy_perbutation, timesteps=t, context=cond)
        #(JW) cond_txt dit에 맞게 수정해야 함 -> label y로 설정
        #student_model = self.dit_model.diffusion_model
        student_output = self.dit_model.diffusion_model(x=x_noisy, t=t, y=label)
        return teacher_eps, student_output
    
    def sd_apply_model(self, x_noisy, t, cond, *args, **kwargs):
        #assert isinstance(cond, dict)
        x_noisy_perbutation = x_noisy #(JW) beta 지정 필요 (첫 코드는 그대로 돌려볼 예정)
        teacher_model = self.model.diffusion_model
        teacher_eps = teacher_model(x=x_noisy_perbutation, timesteps=t, context=cond)
        return teacher_eps

    def sdit_q_posterior_mean_variance(self, x_start, x_t, t, sample=False): #(JW)
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        if sample:
            posterior_mean = (
                _extract_into_tensor(self.sample_posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.sample_posterior_mean_coef2, t, x_t.shape) * x_t
            )
            posterior_variance = _extract_into_tensor(self.sample_posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = _extract_into_tensor(
                self.sample_posterior_log_variance_clipped, t, x_t.shape
            )
            assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
            )
        else:
            posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1.cpu().numpy(), t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2.cpu().numpy(), t, x_t.shape) * x_t
            )
            posterior_variance = _extract_into_tensor(self.posterior_variance.cpu().numpy(), t, x_t.shape)
            posterior_log_variance_clipped = _extract_into_tensor(
                self.posterior_log_variance_clipped.cpu().numpy(), t, x_t.shape
            )
            assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
            )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.sdit_q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.sdit_p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def p_losses(self, x_start, cond, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        teacher_eps, student_output = self.apply_model(x_noisy, t, cond = cond, label = y)
        #(JW) LOSS 수정 필요 - DiT 코드 실행 후 수정 예정
        terms= {}
        loss_dict = {}
        B, C = x_noisy.shape[:2]
        assert student_output.shape == (B, C * 2, *x_noisy.shape[2:])
        student_eps, model_var_values = torch.split(student_output, C, dim=1)
        # Learn the variance using the variational bound, but don't let
        # it affect our mean prediction.  
        prefix = 'train' if self.training else 'val'
        #model_kwargs = dict(y=y)
        # base loss (DiT loss)
        if self.model_var_type == "LEARNED":
            frozen_out = torch.cat([student_eps.detach(), model_var_values], dim=1)
            terms["vb"] = self._vb_terms_bpd(
                model=lambda *args, r=frozen_out: r,
                x_start=x_start,
                x_t=x_noisy,
                t=t,
                clip_denoised=False,
            )["output"].mean() #(JW)여기 확인.

        target = noise # self.model_mean_type: epsilon
        assert student_eps.shape == target.shape == x_start.shape
        terms["mse"] = mean_flat((target - student_eps) ** 2).mean()
        if "vb" in terms:
            terms["Base"] = terms["mse"] + terms["vb"]
        else:
            terms["Base"] = terms["mse"]
        loss_dict.update({f'{prefix}/loss_base': terms["Base"]})

        # teacher loss (SDiT loss)
        loss_simple = self.get_loss(student_eps, teacher_eps, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_distillation': loss_simple.mean()})
        loss_check = self.get_loss(teacher_eps, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/check_sd': loss_check.mean()})
        logvar_t = self.logvar[t].to(self.device)
        distill_loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        terms["Distill"] = self.l_simple_weight * distill_loss.mean()

        loss = terms["Base"] + terms["Distill"]
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)
    
    def _predict_xstart_from_eps(self, x_t, t, eps, sample=False):
        assert x_t.shape == eps.shape
        if sample:
            return (
                _extract_into_tensor(self.sample_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sample_sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
            )
        else:
            return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod.cpu().numpy(), t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod.cpu().numpy(), t, x_t.shape) * eps
            )

    def sdit_p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, sample=False, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[:2]
        assert t.shape == (B,)
        if sample:
            map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
            ts = map_tensor[t]
        else:
            ts = t
        model_output = model(x, ts, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type == "LEARNED": # 이 부분 사용
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if sample:
                min_log = _extract_into_tensor(self.sample_posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(np.log(self.sample_betas), t, x.shape)
            else:
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped.cpu().numpy(), t, x.shape)
                max_log = _extract_into_tensor(np.log(self.betas.cpu().numpy()), t, x.shape) 
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
        else:
            if sample:
                model_variance, model_log_variance = {
                    ModelVarType.FIXED_LARGE: (
                        np.append(self.sample_posterior_variance[1], self.sample_betas[1:]),
                        np.log(np.append(self.sample_posterior_variance[1], self.sample_betas[1:])),
                    ),
                    ModelVarType.FIXED_SMALL: (
                        self.sample_posterior_variance,
                        self.sample_posterior_log_variance_clipped,
                    )
                }[self.model_var_type]
            else:
                model_variance, model_log_variance = {
                    ModelVarType.FIXED_LARGE: (
                        np.append(self.posterior_variance[1], self.betas.cpu().numpy()[1:]),
                        np.log(np.append(self.posterior_variance[1], self.betas.cpu().numpy()[1:])),
                    ),
                    ModelVarType.FIXED_SMALL: (
                        self.posterior_variance,
                        self.posterior_log_variance_clipped.cpu().numpy(),
                    )
                }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
            
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(
            self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output, sample=sample)
        )
        model_mean, _, _ = self.sdit_q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t, sample=sample)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }
    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean
        
    def sdit_p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        #(JW) timestep 250에 맞춤.
        out = self.sdit_p_mean_variance(
            model,
            x,
            t,
            sample=True,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x) #(JW)sd의 경우 temperature가 곱해져있음.
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs) #(JW) model_kwargs 비교 필요
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def sdit_p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        progress=False,
        device=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(self.model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=self.device)
        indices = list(range(self.timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=self.device) #timestep
            with torch.no_grad():
                out = self.sdit_p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def sdit_p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        progress=False,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.sdit_p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    
    def sample_log(self, z, cond, batch_size, ddim, ddim_steps, cfg_scale, **kwargs):
        model_kwargs = dict(y=cond, cfg_scale=cfg_scale)
        if ddim:
            samples = self.ddim_sample_loop(self.dit_model.diffusion_model.forward_with_cfg, shape = z.shape, noise=z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, eta=eta)
        else:
            samples= self.sdit_p_sample_loop(self.dit_model.diffusion_model.forward_with_cfg, shape = z.shape,
                                                               noise=z, clip_denoised=False, model_kwargs=model_kwargs, progress=True)
        return samples
        

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True, cfg_scale=4.0, 
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None
        log = dict()
        z, c, y = self.get_input(batch, self.first_stage_key, bs=N)
        c=c[:N]
        y=y[:N]
        N = min(len(batch[self.first_stage_key]), N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["text"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        z = torch.randn(N, 4, 64, 64, device=self.device)
        # SDiT에 맞춘 입력 주석 풀어야 함.
        dit_z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * N, device = self.device)
        y= torch.cat([y, y_null], 0)
        # Sample images:

        if sample: #(JW) 여기 들어가야 함.
            # get denoise row
            # stable diffusion
            with ema_scope("Sampling"):
                samples, z_denoise_row = super().sample_log(x_T=z, cond=c,
                                                        batch_size=N, ddim=use_ddim,
                                                        ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["teacher samples"] = x_samples
            #DiT
            samples = self.sample_log(z=dit_z, cond = y, batch_size=N, ddim=False, ddim_steps=ddim_steps, eta= ddim_eta, 
                                                   cfg_scale=4.0)
            samples, _ = samples.chunk(2, dim=0)
            samples = self.decode_first_stage(samples)
            log["student samples"] = samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid
            

        # if unconditional_guidance_scale > 1.0: #(JW) 여기 안 들어가게 해야 함.
        #     uc = self.get_unconditional_conditioning(N)
        #     label = y  # torch.zeros_like(c_cat)
        #     txt = uc
        #     with ema_scope("Sampling with classifier-free guidance"):
        #         samples_cfg, _ = super().sample_log(cond=c,
        #                                         batch_size=N, ddim=use_ddim,
        #                                         ddim_steps=ddim_steps, eta=ddim_eta,
        #                                         unconditional_guidance_scale=unconditional_guidance_scale,
        #                                         unconditional_conditioning=uc,
        #                                         )
        #     x_samples_cfg = self.decode_first_stage(samples_cfg)
        #     log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
        #     dit_samples_cfg, _ = self.sample_log()
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.dit_model.parameters())
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                import pdb; pdb.set_trace()
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)