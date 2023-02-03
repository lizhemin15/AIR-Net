# Environment
Our codes based on Pytorch == 1.7.0. You can install it reference [website](https://pytorch.org/get-started/locally/).


# Install and usage
Clone all the github, and open file *main.ipynb*, just run it.

# What have we done?
We proposed a framework aims to solve matrix completion problem better and achieves a good performance in multiple data types and missing type.
## Compared NMAE
We use NMAE to measure the performance of our method:
$$
\mathrm{NMAE}=\frac{1}{\left(\boldsymbol{X}_{\max }^{*}-\boldsymbol{X}_{\min }^{*}\right)|\bar{\Omega}|} \sum_{(i, j) \in \bar{\Omega}}\left|\hat{\boldsymbol{X}}_{i j}-\boldsymbol{X}_{i j}^{*}\right|
$$
![](https://jamily-pic-bed.oss-cn-shenzhen.aliyuncs.com/20211012102316.png)

## Performace in image type data
Our proposed method achieves the best-recovered performance in most tasks.
Table 1 shows the efficacy of AIR-Net on the various data types. More surprising is that our methods perform better than other methods, which are well designed for the particular data type. The recovered results are shown in Figure. In this figure, the existing methods perform well on specific missing pattern data. Such as the RDMF achieved good performance on the random missing case but performed not OK on reminding missing patterns. PNMC completed the patch missing well while obtaining worse results on texture missing. Thanks to the proposed model’s adaptive properties, our method achieves promising results both visually and by numerical measures.

![](https://jamily-pic-bed.oss-cn-shenzhen.aliyuncs.com/Barbara.png)
## Interesting properties
As Figure 1 shows, both Lr(t) and Lc(t) first appear many blocks (t = 4000). Specially, we sigh two of Lc(t = 4000) out. These blocks indicate that these corresponding blocks columns are highly related. These blocks correspond to columns in which the eyes of Baboon are located, which are indeed highly similar. However, the slight difference between these columns induces the relationship captured by adaptive regularizer focusing on the related columns (t = 7000), which is similar to TV(Rudin et al., 1992). The columns of Baboon are not fully the same. The regularization gradually vanishes (t = 10000), which matches the results of Theorem 2 in paper (Figure 1). Except the gray-scale images, the results on Syn-Netflix give similar conclusion.
![](https://jamily-pic-bed.oss-cn-shenzhen.aliyuncs.com/20211012102950.png)

# Reference
If this is helpful for you, please reference our work as 
'''
@article{doi:10.1137/22M1489228,
author = {Li, Zhemin and Sun, Tao and Wang, Hongxia and Wang, Bao},
title = {Adaptive and Implicit Regularization for Matrix Completion},
journal = {SIAM Journal on Imaging Sciences},
volume = {15},
number = {4},
pages = {2000-2022},
year = {2022},
doi = {10.1137/22M1489228},
}
'''
