
[](https://github.com/leaguecn/leenotes/raw/master/img/cpvr2018-zhihu.jpg)


##[CVPR 2018 论文解读集锦（9月27日更新）9]（https://zhuanlan.zhihu.com/p/35131736)


##目录|TOC

<!-- MarkdownTOC levels="1,2,3" autolink="true" style="ordered" -->

1. [Main](#main)
    1. [CVPR 2018|分割算法——可以分割一切目标（附各种分割总结）](#cvpr-2018%7C%E5%88%86%E5%89%B2%E7%AE%97%E6%B3%95%E2%80%94%E2%80%94%E5%8F%AF%E4%BB%A5%E5%88%86%E5%89%B2%E4%B8%80%E5%88%87%E7%9B%AE%E6%A0%87%EF%BC%88%E9%99%84%E5%90%84%E7%A7%8D%E5%88%86%E5%89%B2%E6%80%BB%E7%BB%93%EF%BC%89)
    1. [CVPR2018 oral|香港中文大学胡枭玮:用于阴影检测的 DSC 特征](#cvpr2018-oral%7C%E9%A6%99%E6%B8%AF%E4%B8%AD%E6%96%87%E5%A4%A7%E5%AD%A6%E8%83%A1%E6%9E%AD%E7%8E%AE%E7%94%A8%E4%BA%8E%E9%98%B4%E5%BD%B1%E6%A3%80%E6%B5%8B%E7%9A%84-dsc-%E7%89%B9%E5%BE%81)
    1. [CVPR 2018 | Repulsion loss：专注于遮挡情况下的行人检测](#cvpr-2018-%7C-repulsion-loss%EF%BC%9A%E4%B8%93%E6%B3%A8%E4%BA%8E%E9%81%AE%E6%8C%A1%E6%83%85%E5%86%B5%E4%B8%8B%E7%9A%84%E8%A1%8C%E4%BA%BA%E6%A3%80%E6%B5%8B)
    1. [DensePose开源了，2D变3D人体姿势实时识别 | Facebook@CVPR 2018](#densepose%E5%BC%80%E6%BA%90%E4%BA%86%EF%BC%8C2d%E5%8F%983d%E4%BA%BA%E4%BD%93%E5%A7%BF%E5%8A%BF%E5%AE%9E%E6%97%B6%E8%AF%86%E5%88%AB-%7C-facebookcvpr-2018)
    1. [CVPR 2018|Cascade R-CNN：向高精度目标检测器迈进](#cvpr-2018%7Ccascade-r-cnn%EF%BC%9A%E5%90%91%E9%AB%98%E7%B2%BE%E5%BA%A6%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E5%99%A8%E8%BF%88%E8%BF%9B)
    1. [CV 届的金鸡百花奖：盘点我心中的 CVPR 2018 创意 TOP10](#cv-%E5%B1%8A%E7%9A%84%E9%87%91%E9%B8%A1%E7%99%BE%E8%8A%B1%E5%A5%96%EF%BC%9A%E7%9B%98%E7%82%B9%E6%88%91%E5%BF%83%E4%B8%AD%E7%9A%84-cvpr-2018-%E5%88%9B%E6%84%8F-top10)
    1. [CVPR 2018 |“寻找”极小人脸](#cvpr-2018-%7C%E2%80%9C%E5%AF%BB%E6%89%BE%E2%80%9D%E6%9E%81%E5%B0%8F%E4%BA%BA%E8%84%B8)
    1. [【论文整理】CVPR2018 人体姿态相关](#%E3%80%90%E8%AE%BA%E6%96%87%E6%95%B4%E7%90%86%E3%80%91cvpr2018-%E4%BA%BA%E4%BD%93%E5%A7%BF%E6%80%81%E7%9B%B8%E5%85%B3)
    1. [CVPR 2018 | 炫酷的卡通画目标检测](#cvpr-2018-%7C-%E7%82%AB%E9%85%B7%E7%9A%84%E5%8D%A1%E9%80%9A%E7%94%BB%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B)
    1. [CVPR 2018 | 无监督语义分割之全卷积域适应网络](#cvpr-2018-%7C-%E6%97%A0%E7%9B%91%E7%9D%A3%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2%E4%B9%8B%E5%85%A8%E5%8D%B7%E7%A7%AF%E5%9F%9F%E9%80%82%E5%BA%94%E7%BD%91%E7%BB%9C)
    1. [CVPR 2018 Best Paper Taskonomy 作者解读](#cvpr-2018-best-paper-taskonomy-%E4%BD%9C%E8%80%85%E8%A7%A3%E8%AF%BB)
    1. [CVPR2018最佳论文演讲：研究任务之间的联系才是做迁移学习的正确姿势](#cvpr2018%E6%9C%80%E4%BD%B3%E8%AE%BA%E6%96%87%E6%BC%94%E8%AE%B2%EF%BC%9A%E7%A0%94%E7%A9%B6%E4%BB%BB%E5%8A%A1%E4%B9%8B%E9%97%B4%E7%9A%84%E8%81%94%E7%B3%BB%E6%89%8D%E6%98%AF%E5%81%9A%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%9A%84%E6%AD%A3%E7%A1%AE%E5%A7%BF%E5%8A%BF)
    1. [【CVPR 2018最佳论文出炉】斯坦福等名校折桂，何恺明获年轻研究员奖](#%E3%80%90cvpr-2018%E6%9C%80%E4%BD%B3%E8%AE%BA%E6%96%87%E5%87%BA%E7%82%89%E3%80%91%E6%96%AF%E5%9D%A6%E7%A6%8F%E7%AD%89%E5%90%8D%E6%A0%A1%E6%8A%98%E6%A1%82%EF%BC%8C%E4%BD%95%E6%81%BA%E6%98%8E%E8%8E%B7%E5%B9%B4%E8%BD%BB%E7%A0%94%E7%A9%B6%E5%91%98%E5%A5%96)
    1. [CVPR 2018 | ETH Zurich提出利用对抗策略，解决目标检测的域适配问题](#cvpr-2018-%7C-eth-zurich%E6%8F%90%E5%87%BA%E5%88%A9%E7%94%A8%E5%AF%B9%E6%8A%97%E7%AD%96%E7%95%A5%EF%BC%8C%E8%A7%A3%E5%86%B3%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E7%9A%84%E5%9F%9F%E9%80%82%E9%85%8D%E9%97%AE%E9%A2%98)
    1. [CVPR 2018 | 阿里巴巴&浙大Poster论文：基于直推式无偏嵌入的零样本学习](#cvpr-2018-%7C-%E9%98%BF%E9%87%8C%E5%B7%B4%E5%B7%B4%E6%B5%99%E5%A4%A7poster%E8%AE%BA%E6%96%87%EF%BC%9A%E5%9F%BA%E4%BA%8E%E7%9B%B4%E6%8E%A8%E5%BC%8F%E6%97%A0%E5%81%8F%E5%B5%8C%E5%85%A5%E7%9A%84%E9%9B%B6%E6%A0%B7%E6%9C%AC%E5%AD%A6%E4%B9%A0)
    1. [CVPR 2018 | ETH Zurich提出新型网络「ROAD-Net」，解决语义分割域适配问题](#cvpr-2018-%7C-eth-zurich%E6%8F%90%E5%87%BA%E6%96%B0%E5%9E%8B%E7%BD%91%E7%BB%9C%E3%80%8Croad-net%E3%80%8D%EF%BC%8C%E8%A7%A3%E5%86%B3%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2%E5%9F%9F%E9%80%82%E9%85%8D%E9%97%AE%E9%A2%98)
    1. [CVPR2018|DiracNets：无需跳层连接，训练更深神经网络，结构参数化与Dirac参数化的ResNet](#cvpr2018%7Cdiracnets%EF%BC%9A%E6%97%A0%E9%9C%80%E8%B7%B3%E5%B1%82%E8%BF%9E%E6%8E%A5%EF%BC%8C%E8%AE%AD%E7%BB%83%E6%9B%B4%E6%B7%B1%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%8C%E7%BB%93%E6%9E%84%E5%8F%82%E6%95%B0%E5%8C%96%E4%B8%8Edirac%E5%8F%82%E6%95%B0%E5%8C%96%E7%9A%84resnet)
    1. [CVPR 2018 | 残差密集网络：利用所有分层特征的图像超分辨率网络](#cvpr-2018-%7C-%E6%AE%8B%E5%B7%AE%E5%AF%86%E9%9B%86%E7%BD%91%E7%BB%9C%EF%BC%9A%E5%88%A9%E7%94%A8%E6%89%80%E6%9C%89%E5%88%86%E5%B1%82%E7%89%B9%E5%BE%81%E7%9A%84%E5%9B%BE%E5%83%8F%E8%B6%85%E5%88%86%E8%BE%A8%E7%8E%87%E7%BD%91%E7%BB%9C)
    1. [CVPR2018|DA-GAN技术：计算机帮你创造奇妙“新物种”](#cvpr2018%7Cda-gan%E6%8A%80%E6%9C%AF%EF%BC%9A%E8%AE%A1%E7%AE%97%E6%9C%BA%E5%B8%AE%E4%BD%A0%E5%88%9B%E9%80%A0%E5%A5%87%E5%A6%99%E2%80%9C%E6%96%B0%E7%89%A9%E7%A7%8D%E2%80%9D)
    1. [CVPR 2018 论文概述：有损压缩视频的多帧质量增强方法](#cvpr-2018-%E8%AE%BA%E6%96%87%E6%A6%82%E8%BF%B0%EF%BC%9A%E6%9C%89%E6%8D%9F%E5%8E%8B%E7%BC%A9%E8%A7%86%E9%A2%91%E7%9A%84%E5%A4%9A%E5%B8%A7%E8%B4%A8%E9%87%8F%E5%A2%9E%E5%BC%BA%E6%96%B9%E6%B3%95)
    1. [CVPR 2018 | 使用CNN生成图像先验，实现更广泛场景的盲图像去模糊](#cvpr-2018-%7C-%E4%BD%BF%E7%94%A8cnn%E7%94%9F%E6%88%90%E5%9B%BE%E5%83%8F%E5%85%88%E9%AA%8C%EF%BC%8C%E5%AE%9E%E7%8E%B0%E6%9B%B4%E5%B9%BF%E6%B3%9B%E5%9C%BA%E6%99%AF%E7%9A%84%E7%9B%B2%E5%9B%BE%E5%83%8F%E5%8E%BB%E6%A8%A1%E7%B3%8A)
    1. [CVPR2018：基于时空模型无监督迁移学习的行人重识别](#cvpr2018%EF%BC%9A%E5%9F%BA%E4%BA%8E%E6%97%B6%E7%A9%BA%E6%A8%A1%E5%9E%8B%E6%97%A0%E7%9B%91%E7%9D%A3%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%9A%84%E8%A1%8C%E4%BA%BA%E9%87%8D%E8%AF%86%E5%88%AB)
    1. [独立循环神经网络（IndRNN）：打造更长更深的RNN](#%E7%8B%AC%E7%AB%8B%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%88indrnn%EF%BC%89%EF%BC%9A%E6%89%93%E9%80%A0%E6%9B%B4%E9%95%BF%E6%9B%B4%E6%B7%B1%E7%9A%84rnn)
    1. [CVPR 2018 | 腾讯AI Lab、MIT等机构提出TVNet：可端到端学习视频的运动表征](#cvpr-2018-%7C-%E8%85%BE%E8%AE%AFai-lab%E3%80%81mit%E7%AD%89%E6%9C%BA%E6%9E%84%E6%8F%90%E5%87%BAtvnet%EF%BC%9A%E5%8F%AF%E7%AB%AF%E5%88%B0%E7%AB%AF%E5%AD%A6%E4%B9%A0%E8%A7%86%E9%A2%91%E7%9A%84%E8%BF%90%E5%8A%A8%E8%A1%A8%E5%BE%81)
    1. [CVPR 2018 | Spotlight论文：变分U-Net，可按条件独立变换目标的外观和形状](#cvpr-2018-%7C-spotlight%E8%AE%BA%E6%96%87%EF%BC%9A%E5%8F%98%E5%88%86u-net%EF%BC%8C%E5%8F%AF%E6%8C%89%E6%9D%A1%E4%BB%B6%E7%8B%AC%E7%AB%8B%E5%8F%98%E6%8D%A2%E7%9B%AE%E6%A0%87%E7%9A%84%E5%A4%96%E8%A7%82%E5%92%8C%E5%BD%A2%E7%8A%B6)
    1. [Unsupervised Person Image Synthesis in Arbitrary Poses](#unsupervised-person-image-synthesis-in-arbitrary-poses)
    1. [Person Transfer GAN to Bridge Domain Gap for Person Re-Identification](#person-transfer-gan-to-bridge-domain-gap-for-person-re-identification)
    1. [Disentangled Person Image Generation](#disentangled-person-image-generation)
    1. [Practical Block-wise Neural Network Architecture Generation](#practical-block-wise-neural-network-architecture-generation)
    1. [Deep Layer Aggregation](#deep-layer-aggregation)
    1. [Learning Face Age Progression: A Pyramid Architecture of GANs](#learning-face-age-progression-a-pyramid-architecture-of-gans)
    1. [Convolutional Neural Networks with Alternately Updated Clique](#convolutional-neural-networks-with-alternately-updated-clique)
    1. [Unsupervised Discovery of Object Landmarks as Structural Representations](#unsupervised-discovery-of-object-landmarks-as-structural-representations)
    1. [An Analysis of Scale Invariance in Object Detection - SNIP](#an-analysis-of-scale-invariance-in-object-detection---snip)
    1. [Wasserstein Introspective Neural Networks](#wasserstein-introspective-neural-networks)
    1. [Im2Flow: Motion Hallucination from Static Images for Action Recognition](#im2flow-motion-hallucination-from-static-images-for-action-recognition)
    1. [What have we learned from deep representations for action recognition?](#what-have-we-learned-from-deep-representations-for-action-recognition)
    1. [Squeeze-and-Excitation Networks](#squeeze-and-excitation-networks)
    1. [Actor and Action Video Segmentation from a Sentence](#actor-and-action-video-segmentation-from-a-sentence)
    1. [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](#bottom-up-and-top-down-attention-for-image-captioning-and-visual-question-answering)
    1. [Embodied Question Answering](#embodied-question-answering)
    1. [Taskonomy: Disentangling Task Transfer Learning](#taskonomy-disentangling-task-transfer-learning)
    1. [Detail-Preserving Pooling in Deep Networks](#detail-preserving-pooling-in-deep-networks)
    1. [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](#high-resolution-image-synthesis-and-semantic-manipulation-with-conditional-gans)
    1. [Feature Space Transfer for Data Augmentation](#feature-space-transfer-for-data-augmentation)
    1. [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](#stargan-unified-generative-adversarial-networks-for-multi-domain-image-to-image-translation)
    1. [Discriminative Learning of Latent Features for Zero-Shot Recognition](#discriminative-learning-of-latent-features-for-zero-shot-recognition)
    1. [Relation Networks for Object Detection](#relation-networks-for-object-detection)
    1. [Semi-parametric Image Synthesis](#semi-parametric-image-synthesis)
    1. [Synthesizing Images of Humans in Unseen Poses](#synthesizing-images-of-humans-in-unseen-poses)
1. [More](#more)

<!-- /MarkdownTOC -->



##Main



###[CVPR 2018|分割算法——可以分割一切目标（附各种分割总结）](http://www.cvmart.net/community/article/detail/353)

本文是对何恺明CVPR 2018的目标分割工作解读，同时作者基于时间顺序及相关paper总结了语义分割的结构演变历程。

###[CVPR2018 oral|香港中文大学胡枭玮:用于阴影检测的 DSC 特征](http://www.cvmart.net/community/article/detail/324)

CVPR2018oral文章，来自香港中文大学的胡枭玮采用了提取 DSC 特征的方式，有效解决了计算机视觉物体检测与跟踪中的阴影遮挡问题。

###[CVPR 2018 | Repulsion loss：专注于遮挡情况下的行人检测](http://www.cvmart.net/community/article/detail/309)

本文由同济大学和北京大学合作发表于CVPR2018，聚焦于loss层面，为遮挡情况下的行人检测问题提供了一种行之有效的解决方案。

###[DensePose开源了，2D变3D人体姿势实时识别 | Facebook@CVPR 2018](http://www.cvmart.net/community/article/detail/272)

Facebook发布的人体姿势实时识别系统DensePose开源，可以把2D图像，转换成3D人体模型。

###[CVPR 2018|Cascade R-CNN：向高精度目标检测器迈进](http://www.cvmart.net/community/article/detail/297)

本文是CVPR2018通用目标检测的一篇Oral，在Faster R-CNN框架下，通过级联多个分类器，逐步提高目标定位的精度，向高精度的目标检测器迈出了坚实的一步。

###[CV 届的金鸡百花奖：盘点我心中的 CVPR 2018 创意 TOP10](http://www.cvmart.net/community/article/detail/293)

伴随着深度学习在计算机视觉领域的突飞猛进，我们仍在探索所有的可能性。许多论文将提出全新的深度网络在视觉上的应用。本文作者将向大家展示其认为是CVPR 2018中最酷最具创意的10篇论文。

###[CVPR 2018 |“寻找”极小人脸](http://www.cvmart.net/community/article/detail/290)

本文是CVPR2018人脸检测的一篇Oral，结合GAN的思想提升了极小人脸的检测性能，是GAN入侵目标检测领域的又一篇力作。

###[【论文整理】CVPR2018 人体姿态相关](http://www.cvmart.net/community/article/detail/286)

作者整理了一份CVPR2018中人体姿态相关的论文列表，并简述了论文的思路和方法，方便大家有选择性地阅读和收藏~

###[CVPR 2018 | 炫酷的卡通画目标检测](http://www.cvmart.net/community/article/detail/284)

本文由东京大学发表在CVPR2018，论文采用了图像层面的域适应策略和弱监督伪标签策略，探索了如何将已有的检测器迁移到卡通画中进行目标检测。

###[CVPR 2018 | 无监督语义分割之全卷积域适应网络](http://www.cvmart.net/community/article/detail/278)

本文由中科大和微软亚洲研究院合作完成，论文提出了两种域适应策略，探索了如何使用合成图像提升真实图像的语义分割性能。

###[CVPR 2018 Best Paper Taskonomy 作者解读](https://zhuanlan.zhihu.com/p/38425434)

斯坦福本科生／准博士生沈博魁，CVPR2018最佳论文Taskonomy的共同二作。鉴于poster session时发现很多人对Taskonomy的理解有偏差，作者亲自写了一篇Taskonomy的中文解读，希望能对大家有帮助。

###[CVPR2018最佳论文演讲：研究任务之间的联系才是做迁移学习的正确姿势](http://www.cvmart.net/community/article/detail/275)

今年 CVPR 2018 最佳论文《Taskonomy: Disentangling Task Transfer Learning》（任务学：任务迁移学习的解耦）研究了一个非常新颖的课题，那就是研究视觉任务之间的关系，根据得出的关系可以帮助在不同任务之间做迁移学习。相比于我们看惯了的在为各种任务刷分的研究，这篇论文可谓是一股计算机视觉领域的春风。

###[【CVPR 2018最佳论文出炉】斯坦福等名校折桂，何恺明获年轻研究员奖](http://www.cvmart.net/community/article/detail/271)

CVPR 2018最佳论文花落斯坦福和伯克利大师论文，最佳学生论文奖授予了CMU。何恺明获得了PAMI年轻研究员奖。

###[CVPR 2018 | ETH Zurich提出利用对抗策略，解决目标检测的域适配问题](http://www.cvmart.net/community/article/detail/268)

本文是 ETH Zurich 发表于 CVPR 2018 的工作，论文提出了采用当前火热的对抗策略，解决目标检测任务中使用不同数据集训练的域适配问题。

###[CVPR 2018 | 阿里巴巴&浙大Poster论文：基于直推式无偏嵌入的零样本学习](http://www.cvmart.net/community/article/detail/265)

大多数现有的零样本学习（Zero-Shot Learning，ZSL）方法都存在强偏问题，因此在广义 ZSL 设置中部署后性能很差。本文提出了一个简单有效的方法——准完全监督学习（QFSL）来解决此问题。

【CVPR 2018 论文集】[“Computer Vision Foundation open access”](http://openaccess.thecvf.com/menu.py)[](http://openaccess.thecvf.com/menu.py)

###[CVPR 2018 | ETH Zurich提出新型网络「ROAD-Net」，解决语义分割域适配问题](http://cvmart.net/community/article/detail/240)

本文是 ETH Zurich 发表于 CVPR 2018 的工作，论文提出了两个网络策略，用于处理语义分割任务中使用合成数据训练的域适配问题。

###[CVPR2018|DiracNets：无需跳层连接，训练更深神经网络，结构参数化与Dirac参数化的ResNet](http://cvmart.net/community/article/detail/196)

论文讲述了虚拟化技术牺牲硬件开销和性能，换来软件功能的灵活性；深度模型也类似，如果把网络结构参数化，得到的模型更灵活易控，但是计算效率并不高。

###[CVPR 2018 | 残差密集网络：利用所有分层特征的图像超分辨率网络](http://cvmart.net/community/article/detail/200)

美国东北大学最近在图像超分辨领域提出了一种残差密集网络，来从原图生成高分辨率图像。该网络结合残差网络与密集连接网络的特性充分利用原始 LR 图像的所有分层特征，因而能重构出高质量的图像。

###[CVPR2018|DA-GAN技术：计算机帮你创造奇妙“新物种”](http://cvmart.net/community/article/detail/201)

微软亚研院被CVPR2018接收的论文提出的DA-GAN技术，能够通过文字描述生成新形象，形成了全新的艺术创造模式。

###[CVPR 2018 论文概述：有损压缩视频的多帧质量增强方法](http://cvmart.net/community/article/detail/203)

CVPR 2018论文《Multi Frame
Quality Enhancement for Compressed Video》提出针对有损压缩视频的多帧质量增强方法，显著提升了视频质量增强的性能。

###[CVPR 2018 | 使用CNN生成图像先验，实现更广泛场景的盲图像去模糊](http://cvmart.net/community/article/detail/206)

现有的最优方法在文本、人脸以及低光照图像上的盲图像去模糊效果并不佳，主要受限于图像先验的手工设计属性。本文研究者将图像先验表示为二值分类器，训练 CNN 来分类模糊和清晰图像。实验表明，该图像先验比目前最先进的人工设计先验更具区分性，可实现更广泛场景的盲图像去模糊。

###[CVPR2018：基于时空模型无监督迁移学习的行人重识别](http://cvmart.net/community/article/detail/210)

本文为你解读CVPR2018 TFusion，解决的目标是跨数据集的Person Rei，属于无监督学习，方法是多模态数据融合 + 迁移学习。实验效果上，超越了所有无监督Person reid方法，逼近有监督方法，在部分数据集上甚至超越有监督方法。

###[独立循环神经网络（IndRNN）：打造更长更深的RNN](http://cvmart.net/community/article/detail/207)

电子科技大学和澳大利亚伍伦贡大学的研究者合作发表论文，介绍了他们创造的独立循环神经网络（IndRNN），这种新型RNN能有效解决网络收敛时的梯度爆炸和消失问题，并能够处理更长的序列。

###[CVPR 2018 | 腾讯AI Lab、MIT等机构提出TVNet：可端到端学习视频的运动表征](http://cvmart.net/community/article/detail/222)

来自腾讯 AI Lab、MIT、清华、斯坦福大学的研究者完成并入选 CVPR 2018 Spotlight 论文的一项研究提出了一种能从数据中学习出类光流特征并且能进行端到端训练的神经网络：TVNet

###[CVPR 2018 | Spotlight论文：变分U-Net，可按条件独立变换目标的外观和形状](http://cvmart.net/community/article/detail/222)

来自德国海德堡大学的研究者提出了条件 U-Net，将变分自编码器输出的外观条件化。实验证明，这个模型能够完成条件图像生成和转换。在多个数据集上进行的定性和定量实验表明，该方法比目前最先进的方法都有所提升。

* 以下论文解读来源于[paperweekly](https://mp.weixin.qq.com/s/T8tLhFXuB1DATerDmAK0Dg)

###[Unsupervised Person Image Synthesis in Arbitrary Poses](https://zhuanlan.zhihu.com/p/35626886)

CVPR 2018 Spotlight 论文，ReID + GAN 换 pose。本文用了较多的篇幅讲 loss function，pose 的提取用的是 OpenPose 这个库。 其 loss 分为三部分： Image Adversarial Loss、 Pose Loss、Identity Loss。

论文链接

[https://iri](https://iri/),[upc.edu/people/aagudo/Papers/CVPR2018/apumarola_etal_cvpr18_2.pdf](http://upc.edu/people/aagudo/Papers/CVPR2018/apumarola_etal_cvpr18_2.pdf)

###[Person Transfer GAN to Bridge Domain Gap for Person Re-Identification](https://zhuanlan.zhihu.com/p/35626478)

CVPR 2018 RE-ID Spotlight 一篇，这篇文章主要 contribution 有以下两点：

提出了一个新的更大的数据集，更为细致：考虑到了视角，光照等更为细致的因素，具体参数可以直接看文章；多个数据集间的差异，即 domain-gap，通过 GAN 来生成和模仿。

论文链接

[https://arxiv.org/pdf/1711.08565.pdf](https://arxiv.org/pdf/1711.08565.pdf)

代码链接

[https://github.com/JoinWei-PKU/PTGAN](https://github.com/JoinWei-PKU/PTGAN)

数据集链接

[http://www.pkuvmc.com/publications/msmt17.html](http://www.pkuvmc.com/publications/msmt17.html)

###[Disentangled Person Image Generation](https://zhuanlan.zhihu.com/p/35626735)

在 NIPS 2017 上，该团队已经为我们贡献了 Pose Guided Person Image Generation 这篇非常棒的文章，在 CVPR 2018 中，他们推出的更新的这篇文章不仅仅解决了换 pose 问题，还实现了”随心所欲“的换装换 pose，入选今年的 Spotlight。

论文链接

[https://arxiv.org/pdf/1712.02621.pdf](https://arxiv.org/pdf/1712.02621.pdf)

###[Practical Block-wise Neural Network Architecture Generation](https://zhuanlan.zhihu.com/p/35639635)

CVPR 2018 Oral 一篇，本文主要提出了通过封装模块（block-wise）的方法，运用增强学习设计生成网络架构的方法。

论文链接

[https://arxiv.org/pdf/1708.05552.pdf](https://arxiv.org/pdf/1708.05552.pdf)

###[Deep Layer Aggregation](https://zhuanlan.zhihu.com/p/35641068)

CVPR 2018 Oral，topic：网络设计模块化。如名所示，提出了 aggregation 的具体思路，并在层融合上提出了具体方式。

论文链接

[https://arxiv.org/pdf/1707.06484.pdf](https://arxiv.org/pdf/1707.06484.pdf)

###[Learning Face Age Progression: A Pyramid Architecture of GANs](https://zhuanlan.zhihu.com/p/35661176)

CVPR 2018 Oral，文中提出了特征提取器用于提出特定特征，原因是作者认为相同年龄段的不同人脸有着相同的的纹理等特定信息，而这个提取器就是提取出这些特征。此外，该分类器是经过 age 分类任务预训练好了的。

论文链接

[https://arxiv.org/pdf/1711.10352v1](https://arxiv.org/pdf/1711.10352v1)

###[Convolutional Neural Networks with Alternately Updated Clique](https://zhuanlan.zhihu.com/p/35661397)

北大团队提出的新的 block 设计，achieves the performance of the state of the art with less parameters.。由于 block 内任意两层互连，故实现了 top-bottom refinement，也就实现了 attention 机制。文中还提到了部分 technique。

论文链接

[https://arxiv.org/abs.1802.10419](https://arxiv.org/abs.1802.10419)

代码链接

[https://github.com/iboing/CliqueNet](https://github.com/iboing/CliqueNet)

###[Unsupervised Discovery of Object Landmarks as Structural Representations](https://zhuanlan.zhihu.com/p/35693735)

CVPR 2018 Oral， 今年 CVPR Landmark 和 Attention 这两个词出现的频率很高。现在看到的是第二版，对该文进行了更深入的研究后完成。

论文链接

[https://arxiv.org/pdf/1804.04412.pdf](https://arxiv.org/pdf/1804.04412.pdf)

###[An Analysis of Scale Invariance in Object Detection - SNIP](https://zhuanlan.zhihu.com/p/35697040)

CVPR 2018 Oral，文章主要解决网络处理不同 scale 图片的网络策略。

论文链接

[https://arxiv.org/pdf/1711.08189.pdf](https://arxiv.org/pdf/1711.08189.pdf)

###[Wasserstein Introspective Neural Networks](https://zhuanlan.zhihu.com/p/35699851)

文章主要做的事情就是提出了基于 Wasserstein 的 INN，该组已经在 ICCV，NIPS 等多次提出并改进该模型，该模型主要做的就是将 GAN 中的 G 和 D 结合起来。

论文链接

[https://arxiv.org/pdf/1711.08875](https://arxiv.org/pdf/1711.08875)

代码链接

[https://github.com/kjunelee/WINN](https://github.com/kjunelee/WINN)

###[Im2Flow: Motion Hallucination from Static Images for Action Recognition](https://zhuanlan.zhihu.com/p/35702085)

CVPR 2018 Oral,用 u-net 训练一个模型：输入是一个静态的帧，输出的预测的五帧光流信息，模型在 YouTube 数据集上训练。

论文链接

[https://arxiv.org/pdf/1712.04109.pdf](https://arxiv.org/pdf/1712.04109.pdf)

###[What have we learned from deep representations for action recognition?](https://zhuanlan.zhihu.com/p/35703892)

CVPR 2018 Zisserman 的新论文，这篇文章就是 two-stream 模型中间层的可视化方法，换句话说，就是探寻 two-stream 模型学到了怎样的时空信息。

论文链接

[https://arxiv.org/pdf/1801.01415.pdf](https://arxiv.org/pdf/1801.01415.pdf)

###[Squeeze-and-Excitation Networks](https://zhuanlan.zhihu.com/p/35704334)

考虑通道加权，用全局池化后训练的全连层作为权重，相当于在 channels 上加了一个 attention 机制，给不同的 channel 不同的权重。

论文链接

[https://arxiv.org/pdf/1709.01507.pdf](https://arxiv.org/pdf/1709.01507.pdf)

代码链接

[https://github.com/hujie-frank/SENet](https://github.com/hujie-frank/SENet)

###[Actor and Action Video Segmentation from a Sentence](https://zhuanlan.zhihu.com/p/35718996)

CVPR 2018 Oral，本文定义了一个新的任务：给出一个句子，根据该句子分割视频中的 actor 及其 action。

论文链接

[https://arxiv.org/pdf/1803.07485.pdf](https://arxiv.org/pdf/1803.07485.pdf)

###[Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://zhuanlan.zhihu.com/p/35778768)

CVPR 2018 Oral，主题是 attention+VQA，本文的主要思路是用 faster-rcnn 提取出 proposal，然后用其做 image captioning 或者 VQA，该方法取得了 2017 VQA Challenge 的第一名。

论文链接

[https://arxiv.org/pdf/1707.07998](https://arxiv.org/pdf/1707.07998)

代码链接

[https://github.com/peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)

###[Embodied Question Answering](https://zhuanlan.zhihu.com/p/35791145)

这篇文章主要定义了一个新的 task 并给出了一个数据集。将一个 agent 随机丢进一个房间，向他提出一个问题，这个 agent 需要自己导航并回答问题。

论文链接

[https://arxiv.org/pdf/1711.11543](https://arxiv.org/pdf/1711.11543)

###[Taskonomy: Disentangling Task Transfer Learning](https://zhuanlan.zhihu.com/p/35833517)

本文定义了一个新的任务，针对在视觉内的迁移学习，并提出了一个蛮大的数据库， 定义的任务是这样的：vision task 类目很多，只针对某个问题来解决的话，会需要很大的标注的数据集，但我们不难想到的是，一个视觉任务的解决应该能够一定程度的解决另一个视觉任务，毕竟一个成熟模型的构建意味着对该 image 的一定的理解，而这部分的理解的一部分或许对另一个 task 有助益，例，物体 relation 的理解对深度信息的学习毫无疑问是有着助益的。

论文链接

[https://arxiv.org/pdf/1804.08328.pdf](https://arxiv.org/pdf/1804.08328.pdf)

代码链接

[https://github.com/StanfordVL/taskonomy](https://github.com/StanfordVL/taskonomy)

###[Detail-Preserving Pooling in Deep Networks](https://zhuanlan.zhihu.com/p/35938826)

CVPR 2018 Oral，顾名思义，提出了保留 detail 的池化方法。 max/avg pooling 一个只选取最大而忽略与周围像素的关联性，一个重视关联性却又直接抹平，并且在实际梯度计算中也有一些 drawback，所以该文提出了这个新方法，一句话概括，就是在池化过程中学了一个动态的 weight。

论文链接

[https://arxiv.org/pdf/1804.04076.pdf](https://arxiv.org/pdf/1804.04076.pdf)

代码链接

[https://github.com/visinf/dpp](https://github.com/visinf/dpp)

###[High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://zhuanlan.zhihu.com/p/35955531)

CVPR 2018 Oral，本文解决了 GAN 生成高分辨率突破的问题，分辨率达到了 2048/*1024，方法精细，值得深入来看。

论文链接

[https://arxiv.org/pdf/1711,11585.pdf](https://arxiv.org/pdf/1711,11585.pdf)

代码链接

[https://github.com/NVIDIA/pix2pixHD](https://github.com/NVIDIA/pix2pixHD)

###[Feature Space Transfer for Data Augmentation](https://zhuanlan.zhihu.com/p/35956505)

CVPR 2018 ORAL，用GAN生成新数据。

论文链接

[https://arxiv.org/pdf/1801.04356.pdf](https://arxiv.org/pdf/1801.04356.pdf)

###[StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://zhuanlan.zhihu.com/p/35967629)

这里的 domain 是指针对数据集中的 attribute，根据 attribute 来划分的，比如相对于发色而言，金发是一个 domain，黑发是一个 domain ，作者在本文提出了一个可以解决 multiple domain translation 的 translator。

论文链接

[https://arxiv.org/pdf/1711.09020.pdf](https://arxiv.org/pdf/1711.09020.pdf)

代码链接

[https://github.com/yunjey/StarGAN](https://github.com/yunjey/StarGAN)

###[Discriminative Learning of Latent Features for Zero-Shot Recognition](https://zhuanlan.zhihu.com/p/36030705)

CVPR 2018 Oral，Zero-Shot Learning 就是寻求将学习到的特征映射到另一个空间中，从而 map 到 seen 及 unseen 的属性或者 label 上。这篇文章的主要亮点在于学习了已定义label的同时，学习了latent attribute（隐含属性）。

论文链接

[https://arxiv.org/pdf/1803.06731.pdf](https://arxiv.org/pdf/1803.06731.pdf)

###[Relation Networks for Object Detection](https://zhuanlan.zhihu.com/p/36033824)

CVPR 2018 Oral，本文设计了一个考虑 relation 的 module，来增强 object detection 的性能。

论文链接

[https://arxiv.org/pdf/1711.11575.pdf](https://arxiv.org/pdf/1711.11575.pdf)

###[Semi-parametric Image Synthesis](https://zhuanlan.zhihu.com/p/36073450)

CVPR 2018 Oral，来自 CUHK，这里设计的模型以 semantic layout 输入，输出真实的相片般的图片。

论文链接

[https://vladlen.info/papers/SIMS.pdf](https://vladlen.info/papers/SIMS.pdf)

###[Synthesizing Images of Humans in Unseen Poses](https://zhuanlan.zhihu.com/p/36076119)

CVPR 2018 Oral，Pose 合成。

论文链接

[https://arxiv.org/pdf/1804.07739.pdf](https://arxiv.org/pdf/1804.07739.pdf)



----

##More
[CVPR2017的论文解读文章](http://cvmart.net/community/article/detail/69) 
[ICCV 2017 论文解读集锦](http://cvmart.net/community/article/detail/153)
[CVPR 2018论文解读](http://bbs.cvmart.net/articles/56/cvpr-2018-lun-wen-jie-du-ji-jin-12-yue-19-ri-geng-xin)