# EpiDiff

[Arxiv23] EpiDiff: Enhancing Multi-View Synthesis via Localized Epipolar-Constrained Diffusion

## 🏠 <a href="https://huanngzh.github.io/EpiDiff/" target="_blank">Project Page</a> | <a href="https://arxiv.org/abs/2312.06725" target="_blank">Paper</a>

![img:teaser](assets/teaser.png)

Abstract: _Generating multiview images from a single view facilitates the rapid generation of a 3D mesh conditioned on a single image. Recent methods that introduce 3D global representation into diffusion models have shown the potential to generate consistent multiviews, but they have reduced generation speed and face challenges in maintaining generalizability and quality. To address this issue, we propose EpiDiff, a localized interactive multiview diffusion model. At the core of the proposed approach is to insert a lightweight epipolar attention block into the frozen diffusion model, leveraging epipolar constraints to enable cross-view interaction among feature maps of neighboring views. The newly initialized 3D modeling module preserves the original feature distribution of the diffusion model, exhibiting compatibility with a variety of base diffusion models. Experiments show that EpiDiff generates 16 multiview images in just 12 seconds, and it surpasses previous methods in quality evaluation metrics, including PSNR, SSIM and LPIPS. Additionally, EpiDiff can generate a more diverse distribution of views, improving the reconstruction quality from generated multiviews._

## 🔨 Method Overview

![img:pipeline](assets/pipeline.png)

## 🤝 Acknowledgement

We appreciate the open source of the following projects:

[Zero123](https://github.com/cvlab-columbia/zero123) &#8194;
[One-2-3-45](https://github.com/One-2-3-45/One-2-3-45) &#8194;
[SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer) &#8194;
[threestudio](https://github.com/threestudio-project/threestudio) &#8194;
[instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl) &#8194;
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) &#8194;
[diffusers](https://github.com/huggingface/diffusers) &#8194;
[MVDiffusion](https://github.com/Tangshitao/MVDiffusion) &#8194;
[GPNR](https://github.com/google-research/google-research/tree/master/gen_patch_neural_rendering)

## 📎 Citation

If you find this repository useful, please consider citing:

```
@misc{huang2023epidiff,
    title={EpiDiff: Enhancing Multi-View Synthesis via Localized Epipolar-Constrained Diffusion},
    author={Zehuan Huang and Hao Wen and Junting Dong and Yaohui Wang and Yangguang Li and Xinyuan Chen and Yan-Pei Cao and Ding Liang and Yu Qiao and Bo Dai and Lu Sheng},
    journal={arXiv preprint arXiv:2312.06725},
    year={2023}
}
```
