# Completion3D: Stanford 3D Object Point Cloud Completion Benchmark
# TopNet: Structural Point Cloud Decoder

This repository contains source code for all methods used for the [Stanford 3D Object Point Cloud Completion Benchmark](https://www.completion3d.stanford.edu) and presented in the paper [Tchapmi et al., TopNet: Structural Point Cloud Decoder, CVPR 2019](http://arxiv.org/abs/).


## Project Pages

The TopNet project page is available at [https://topnet.stanford.edu/](https://topnet.stanford.edu/).
The completion3d benchmark is available at [https://completion3d.stanford.edu](https://completion3d.stanford.edu/).

## Overview

![Overview](imgs/compare_soa_2018_4.jpg)
*Point Cloud Completion Results. A partial point cloud is given as input and various methods used to generate a completed point cloud*

The completion3D benchmark is a platform for evaluating state-of-the-art 3D Object Point Cloud Completion methods. This repository contains source code for various methods evaluated on the benchmark. Both Tensorflow and Pytorch are supported.


## Citing this work

If you find this work useful in your research, please consider citing:
```
@inproceedings{topnet2019,
  title={TopNet: Structural Point Cloud Decoder},
  author={Tchapmi, Lyne P and Kosaraju, Vineet and Rezatofighi, S. Hamid and Reid, Ian and Savarese, Silvio},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}

@inProceedings{yuan2018pcn,
  title     = {PCN: Point Completion Network},
  author    = {Yuan, Wentao and Khot, Tejas and Held, David and Mertz, Christoph and Hebert, Martial},
  booktitle = {3D Vision (3DV), 2018 International Conference on},
  year      = {2018}
}

@article{DBLP:journals/corr/ChangFGHHLSSSSX15,
  author    = {Angel X. Chang and Thomas A. Funkhouser and Leonidas J. Guibas and Pat Hanrahan and Qi{-}Xing Huang and Zimo Li and Silvio Savarese and Manolis Savva and Shuran Song and Hao Su and Jianxiong Xiao and Li Yi and Fisher Yu},
  title     = {ShapeNet: An Information-Rich 3D Model Repository},
  journal   = {CoRR},
  volume    = {abs/1512.03012},
  year      = {2015},
  url       = {http://arxiv.org/abs/1512.03012},
  archivePrefix = {arXiv},
  eprint    = {1512.03012},
  timestamp = {Mon, 13 Aug 2018 16:47:39 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/ChangFGHHLSSSSX15},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
And please refer to the [Shapenet Terms of Use](https://www.shapenet.org/terms)

```

## License

MIT License
