# Specular-Gaussians

It should be noted that this is **not an implementation of a paper**. The main thing I did was replace SH with [ASG](https://cg.cs.tsinghua.edu.cn/people/~kun/asg/), which allows 3D Gaussians to model more complex highlights. But the reflexes still need to be improved. But even so, **Specular-Gaussians** has become the SOTA 3D-Gaussians-based method for NeRF scenes.

![Comparison](assets/teaser.png)





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
