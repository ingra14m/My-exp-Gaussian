# Specular-Gaussians

It should be noted that this is **not an implementation of a paper**. The key modification I made was substituting SH with [ASG](https://cg.cs.tsinghua.edu.cn/people/~kun/asg/), enhancing the ability of 3D Gaussians to model complex specular. However, there's still room for improvement in the reflection quality. Despite this, **Specular-Gaussians** has emerged as the SOTA method based on 3D Gaussians for NeRF scenes. I've decided to release the code because I think simply adapting 3D-GS based on [nrff](https://github.com/imkanghan/nrff) and [NeuRBF](https://github.com/oppo-us-research/NeuRBF) doesn't seem sufficient to produce a truly useful paper.

![teaser](assets/teaser.png)



## Results

**Quantitative Results on NeRF Synthetic Dataset**

![results](assets/results.png)

**Qualitative Results on Nex Dataset**

![cd-compare](assets/cd-compare.png)



## BibTex

Thanks to the authors of [3D Gaussians](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) for their excellent code, please consider cite this repository:

```
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

If you find this implementation helpful, please consider to cite:

```
@misc{yang2023speculargs,
  title={Specular-Gaussians},
  author={Ziyi, Yang},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/ingra14m/Specular-Gaussians}},
  year={2023}
}
```
