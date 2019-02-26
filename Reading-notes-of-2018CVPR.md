

## Reading notes of 2018CVPR

<!-- MarkdownTOC levels="1,2,3,4" autolink="true" style="ordered" -->

1. [3D pose or shape reconstruction](#3d-pose-or-shape-reconstruction)
    1. [Extreme 3D Face Reconstruction: Seeing Through Occlusions](#extreme-3d-face-reconstruction-seeing-through-occlusions)
    1. [Factoring Shape, Pose, and Layout from the 2D Image of a 3D Scene](#factoring-shape-pose-and-layout-from-the-2d-image-of-a-3d-scene)
    1. [End-to-end learning of keypoint detector and descriptor for pose invariant 3D matching](#end-to-end-learning-of-keypoint-detector-and-descriptor-for-pose-invariant-3d-matching)
    1. [Erase or Fill? Deep Joint Recurrent Rain Removal and Reconstruction in Videos](#erase-or-fill-deep-joint-recurrent-rain-removal-and-reconstruction-in-videos)
    1. [Disentangling Features in 3D Face Shapes for Joint Face Reconstruction and Recognition](#disentangling-features-in-3d-face-shapes-for-joint-face-reconstruction-and-recognition)
    1. [Dense 3D Regression for Hand Pose Estimation](#dense-3d-regression-for-hand-pose-estimation)
    1. [Depth-Based 3D Hand Pose Estimation: From Current Achievements to Future Goals](#depth-based-3d-hand-pose-estimation-from-current-achievements-to-future-goals)
    1. [DeLS-3D: Deep Localization and Segmentation with a 3D Semantic Map](#dels-3d-deep-localization-and-segmentation-with-a-3d-semantic-map)
    1. [Deeply Learned Filter Response Functions for Hyperspectral Reconstruction](#deeply-learned-filter-response-functions-for-hyperspectral-reconstruction)
    1. [Deep Spatial Feature Reconstruction for Partial Person Re-identification: Alignment-free Approach](#deep-spatial-feature-reconstruction-for-partial-person-re-identification-alignment-free-approach)
    1. [Curve Reconstruction via the Global Statistics of Natural Curves](#curve-reconstruction-via-the-global-statistics-of-natural-curves)
    1. [Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](#can-spatiotemporal-3d-cnns-retrace-the-history-of-2d-cnns-and-imagenet)
    1. [Coding Kendall’s Shape Trajectories for 3D Action Recognition](#coding-kendall%E2%80%99s-shape-trajectories-for-3d-action-recognition)
    1. [Augmenting Crowd-Sourced 3D Reconstructions using Semantic Detections](#augmenting-crowd-sourced-3d-reconstructions-using-semantic-detections)
    1. [Automatic 3D Indoor Scene Modeling from Single Panorama](#automatic-3d-indoor-scene-modeling-from-single-panorama)
    1. [Alive Caricature from 2D to 3D](#alive-caricature-from-2d-to-3d)
    1. [A Papier-Mache Approach to Learning 3D Surface Generation](#a-papier-mache-approach-to-learning-3d-surface-generation)
    1. [3D Semantic Trajectory Reconstruction from 3D Pixel Continuum](#3d-semantic-trajectory-reconstruction-from-3d-pixel-continuum)
    1. [3D-RCNN: Instance-level 3D Object Reconstruction via Render-and-Compare](#3d-rcnn-instance-level-3d-object-reconstruction-via-render-and-compare)
    1. [3D Pose Estimation and 3D Model Retrieval for Objects in the Wild](#3d-pose-estimation-and-3d-model-retrieval-for-objects-in-the-wild)
    1. [3D Object Detection with Latent Support Surfaces](#3d-object-detection-with-latent-support-surfaces)
    1. [3D Registration of Curves and Surfaces using Local Differential Information](#3d-registration-of-curves-and-surfaces-using-local-differential-information)
    1. [3D Semantic Segmentation with Submanifold Sparse Convolutional Networks](#3d-semantic-segmentation-with-submanifold-sparse-convolutional-networks)
    1. [2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning](#2d3d-pose-estimation-and-action-recognition-using-multitask-deep-learning)
    1. [3D Human Pose Estimation in the Wild by Adversarial Learning](#3d-human-pose-estimation-in-the-wild-by-adversarial-learning)
    1. [3D Human Sensing, Action and Emotion Recognition in Robot Assisted Therapy of Children with Autism](#3d-human-sensing-action-and-emotion-recognition-in-robot-assisted-therapy-of-children-with-autism)
    1. [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](#voxelnet-end-to-end-learning-for-point-cloud-based-3d-object-detection)

<!-- /MarkdownTOC -->

### 3D pose or shape reconstruction

*Reading notes of 2018CVPR, Computer Vision & Pattern Recognition 2018,  processings selected in 3D reconstruction aspect.*

----------------

**2019-02-27**

#### Extreme 3D Face Reconstruction: Seeing Through Occlusions
*by Anh Tuan Tran, Tal Hassner, Iacopo Masi, Eran Paz, Yuval Nirkin, and Gerard Medioni*     

**Abstract**     

Existing single view, 3D face reconstruction methods can produce beautifully detailed 3D results, but typically only for near frontal, unobstructed viewpoints. We describe a system designed to provide detailed 3D reconstructions of faces viewed under extreme conditions, out of plane rotations, and occlusions. Motivated by the concept of bump mapping, we propose a layered approach which decouples estimation of a global shape from its mid-level details (e.g., wrinkles). We estimate a coarse 3D face shape which acts as a foundation and then separately layer this foundation with details represented by a bump map. We show how a deep convolutional encoder-decoder can be used to estimate such bump maps. We further show how this approach naturally extends to generate plausible details for occluded facial regions. We test our approach and its components extensively, quantitatively demonstrating the invariance of our estimated facial details. We further provide numerous qualitative examples showing that our method produces detailed 3D face shapes in viewing conditions where existing state of the art often break down.      
+ Result
![](https://github.com/leaguecn/leenotes/raw/master/img/Extreme-3D-Face-Reconstruction-Seeing-Through-Occlusions-result.png)
+ Method overview
![](https://github.com/leaguecn/leenotes/raw/master/img/Extreme-3D-Face-Reconstruction-Seeing-Through-Occlusions-method.png)

-----

#### Factoring Shape, Pose, and Layout from the 2D Image of a 3D Scene

*by Shubham Tulsiani, Saurabh Gupta, David Fouhey, Alexei A. Efros, Jitendra Malik*

**Abstract**     
The goal of this paper is to take a single 2D image of a scene and recover the 3D structure in terms of a small set of factors: a layout representing the enclosing surfaces as well as a set of objects represented in terms of shape and pose. We propose a convolutional neural network-based approach to predict this representation and benchmark it on a large dataset of indoor scenes. Our experiments evaluate a number of practical design questions, demonstrate that we can infer this representation, and quantitatively and qualitatively demonstrate its merits compared to alternate representations.



![](https://github.com/leaguecn/leenotes/raw/master/img/Factoring-Shape-Pose-and-Layout-from-the-2D-Image-of-a-3D-Scene.png)



------------------

**2019-02-26**

#### End-to-end learning of keypoint detector and descriptor for pose invariant 3D matching
**用于姿态不变三维匹配的关键特征点检测子和描述子的端对端学习**     
*by Georgios Georgakis, Srikrishna Karanam, Ziyan Wu, Jan Ernst, and Jana Koseck*

**Abstract**      
Finding correspondences between images or 3D scans is at the heart of many computer vision and image retrieval applications and is often enabled by matching local keypoint descriptors. Various learning approaches have been applied in the past to different stages of the matching pipeline, considering detection, description, or metric learning objectives. These objectives were typically addressed separately and most previous work has focused on image data. This paper proposes an end-to-end learning framework for keypoint detection and its representation (descriptor) for 3D depth maps or 3D scans, where the two can be jointly optimized towards task-specific objectives without a need for separate annotations. We employ a Siamese architecture augmented by a sampling layer and a novel score loss function which in turn affects the selection of region proposals. The positive and negative examples are obtained automatically by sampling corresponding region proposals based on their consistency with known 3D pose labels. Matching experiments with depth data on multiple benchmark datasets demonstrate the efficacy of the proposed approach, showing significant improvements over state-of-theart methods.      
//在图像和三维扫描结构间的同名点是许多计算机视觉和图像恢复应用的主要任务，通常通过匹配局部的特征点描述子完成。先前，多种学习方法已经被应用到不同的匹配流程上，包括检测、描述或者标准学习目标。这些目标被典型地各自处理，且许多先前的工作聚焦于图像数据。本文提出了端对端学习框架，用于特征点检测和它的三维深度图或三维扫描结构的代表物/描述子，这两个可以被联结优化到任务明确的目标，并不需要分别标注。我们配置了一个Siamese设计，通过一个采样层和一个能够反向影响区域建议选择的虚构分数损失函数增强。正反样本由基于它们的已知的三维姿态标签一致性的采样相似区域建议自动获得。基于不同标准平台的深度数据集的匹配实验表明了所提方法的有效性，显示了比先进方法具有较大的提高。     

+ method outline
![](https://github.com/leaguecn/leenotes/raw/master/img/End-to-end-learning-of-keypoint-detector-and-descriptor-for-pose-invariant-3D-matching-outline.png)
+ overview of siamese architecture
![](https://github.com/leaguecn/leenotes/raw/master/img/End-to-end-learning-of-keypoint-detector-and-descriptor-for-pose-invariant-3D-matching-architecture.png)

---------------------

#### Erase or Fill? Deep Joint Recurrent Rain Removal and Reconstruction in Videos
**擦除还是填充？视频雨水移除和重建的深度联结回归**      
*by Jiaying Liu, Wenhan Yang, Shuai Yang, Zongming Guo*     

**Abstract**     
In this paper, we address the problem of video rain removal by constructing deep recurrent convolutional networks. We visit the rain removal case by considering rain occlusion regions, i.e. the light transmittance of rain streaks is low. Different from additive rain streaks, in such rain occlusion regions, the details of background images are completely lost. Therefore, we propose a hybrid rain model to depict both rain streaks and occlusions. With the wealth of temporal redundancy, we build a Joint Recurrent Rain Removal and Reconstruction Network (J4R-Net) that seamlessly integrates rain degradation classification, spatial texture appearances based rain removal and temporal coherence based background details reconstruction. The rain degradation classification provides a binary map that reveals whether a location is degraded by linear additive streaks or occlusions. With this side information, the gate of the recurrent unit learns to make a trade-off between rain streak removal and background details reconstruction. Extensive experiments on a series of synthetic and real videos with rain streaks verify the superiority of the proposed method over previous state-of-the-art methods.       
//在本文中，我们提出了通过构建深度回归卷积神经网络进行视频水滴移除的问题。我们观察了雨水移除的例子，考虑到雨水闭合区域等等问，如雨水条纹的光线传送是较低的。不同的是额外雨水的条纹，在雨水闭合区域，背景图像的细节被完全丢失。因此我们提出了一种混合的雨水模型去描述雨纹和闭塞区。通过富余的时间冗余，我们搭建了一个联结回归雨水移除和重建网络（J4R-Net)，可以无缝整合雨水降落分类、基于雨水移除和背景细节重建的时间一致的空间纹理面。雨水降落分类提供一张二值图，揭示一个位置是否是通过线性增加的条纹或闭塞降落。利用这些边缘信息，回归单元入口学习去做一个介于雨水条纹移除和背景细节重构权衡。拓展实验在系列的合成和真实带有雨水条纹视频中实施，验证了所提方法教先前的先进方法的优越性。      


+ show case
![](https://github.com/leaguecn/leenotes/raw/master/img/Erase-or-Fill-Deep-Joint-Recurrent-Rain-Removal-and-Reconstruction-in-Videos.png)
+ Framework
![](https://github.com/leaguecn/leenotes/raw/master/img/Erase-or-Fill-Deep-Joint-Recurrent-Rain-Removal-and-Reconstruction-in-Videos-framework.png)

---------------------


**2019-02-25 break of day**     

#### Disentangling Features in 3D Face Shapes for Joint Face Reconstruction and Recognition
**用于节点面部重建和识别的三维面部形状特征解缠**      
*by Feng Liu, Ronghang Zhu, Dan Zeng, Qijun Zhao, and Xiaoming Liu*      

**Abstract**      
This paper proposes an encoder-decoder network to disentangle shape features during 3D face reconstruction from single 2D images, such that the tasks of reconstructing accurate 3D face shapes and learning discriminative shape features for face recognition can be accomplished simultaneously. Unlike existing 3D face reconstruction methods, our proposed method directly regresses dense 3D face shapes from single 2D images, and tackles identity and residual (i.e., non-identity) components in 3D face shapes explicitly and separately based on a composite 3D face shape model with latent representations. We devise a training process for the proposed network with a joint loss measuring both face identification error and 3D face shape reconstruction error. To construct training data we develop a method for fitting 3D morphable model (3DMM) to multiple 2D images of a subject. Comprehensive experiments have been done on MICC, BU3DFE, LFW and YTF databases. The results show that our method expands the capacity of 3DMM for capturing discriminative shape features and facial detail, and thus outperforms existing methods both in 3D face reconstruction accuracy and in face recognition accuracy.     
//本文提出了一个编解码网络来解缠形状特征，在单张二维图像到三维面部重构期间。这样的话，重构准确的三维面部形状和学习面部识别的差异形状特征两项任务可以被同时完成。不像存在的三维面部重构方法，我们提出的方法直接重单张二维图像回归稠密三维面部形状，并基于潜在代表的合成三维面部形状模型明确并分别处理三维面部形状的确定和余留（如，非确定）成分。我们设计一个训练处理过程，用于含有在测量面部特征错误和三维面部形状重构误差的节点损失的提出网络。为了重构训练数据，我们提出了用于拟合三维形态模型（3DMM）来多样化目标二维图像。综合的实验已经被实施在MICC，BU3DFE，LFW和YTF数据集中。结果展示了我们的方法拓展了3DMM的捕获差别形状特征和面部细节的能力，并且由于当前存在的方法，无论是在三维面部重建精度还是在面部识别精度方面。     
![](https://github.com/leaguecn/leenotes/raw/master/img/Disentangling-Features-in-3D-Face-Shapes-for-Joint-Face-Reconstruction-and-Recognition.png)


----------
##### Disentangling 3D Pose in A Dendritic CNN for Unconstrained 2D Face Alignment
**用于非约束二维面部对齐的枝节CNN的三维姿态解缠**    
*by Amit Kumar Rama Chellappa*     

**Abstract**     

Heatmap regression has been used for landmark localization for quite a while now. Most of the methods use a very deep stack of bottleneck modules for heatmap classification stage, followed by heatmap regression to extract the keypoints. In this paper, we present a single dendritic CNN, termed as Pose Conditioned Dendritic Convolution Neural Network (PCD-CNN), where a classification network is followed by a second and modular classification network, trained in an end to end fashion to obtain accurate landmark points. Following a Bayesian formulation, we disentangle the 3D pose of a face image explicitly by conditioning the landmark estimation on pose, making it different from multi-tasking approaches. Extensive experimentation shows that conditioning on pose reduces the localization error by making it agnostic to face pose. The proposed model can be extended to yield variable number of landmark points and hence broadening its applicability to other datasets. Instead of increasing depth or width of the network, we train the CNN efficiently with Mask-Softmax Loss and hard sample mining to achieve upto 15% reduction in error compared to state-of-the-art methods for extreme and medium pose face images from challenging datasets including AFLW, AFW, COFW and IBUG.
//热力图回归以被用于路标定位较长一段时间。大部分的方法是使用一个瓶颈深栈模型来进行热力图的分类，随之使用热力图来提取关键点。在本文中，我们提出了一种树枝形CNN,叫做姿态条件树枝形卷积神经网络（PCD-CNN）。在该网络中，一个分类网络紧跟一个第二和模块化的分类网络，在一个端对端的风格中训练来获得精确的路标点。基于贝叶斯公式，我们明确使用条件化的在姿态的路标估计来解缠面部图形的三维姿态，使之不同于多任务方法。拓展实验显示了条件化姿态可以通过模糊面部姿态减少定位误差。所提模型可以被拓展来获得一系列的路标点并和拓宽它在其他数据集中的应用能力。而不是增加网络的深度和宽度，我们基于Mask-Softmax损失和用功的样本挖掘高效训练CNN来完成减少15%的误差，较其他的在AFLW、AFW、COFW和IBUG挑战数据集中极端和中度面部姿态图像中先进方法。      
![](https://github.com/leaguecn/leenotes/raw/master/img/Disentangling-3D-Pose-in-A-Dendritic-CNN-for-Unconstrained-2D-Face-Alignment.png)


-----------
**2019-02-24**

#### Dense 3D Regression for Hand Pose Estimation
**手势估计稠密三维回归**     
*by Chengde Wan, Thomas Probst, Luc Van Gool, and Angela Yao*

**Abstract**    
We present a simple and effective method for 3D hand pose estimation from a single depth frame. As opposed to previous state-of-the-art methods based on holistic 3D regression, our method works on dense pixel-wise estimation. This is achieved by careful design choices in pose parameterization, which leverages both 2D and 3D properties of depth map. Specifically, we decompose the pose parameters into a set of per-pixel estimations, i.e., 2D heat maps, 3D heat maps and unit 3D directional vector fields. The 2D/3D joint heat maps and 3D joint offsets are estimated via multitask network cascades, which is trained end-to-end. The pixel-wise estimations can be directly translated into a vote casting scheme. A variant of mean shift is then used to aggregate local votes while enforcing consensus between the the estimated 3D pose and the pixel-wise 2D and 3D estimations by design. Our method is efficient and highly accurate. On MSRA and NYU hand dataset, our method outperforms all previous state-of-the-art approaches by a large margin. On the ICVL hand dataset, our method achieves similar accuracy compared to the nearly saturated result obtained by [5] and outperforms various other proposed methods. Code is available online1.      
//我们提出了一种简单而有效的方法，用于从单一深度图像中估计三维手势。和以往的基于整体三维回归先进方法相反，我们的方法用于稠密的像素估计。通过在姿态参数化中细致的设计选择来完成这些估计，作用于二维和三维性质的深度图。明确来说，我们将姿态参数分解为一系列的像素级别的估计，如：二维热力图，三维热立体和联合的三维方向向量场。二维或三维联结热力图和三维联结补偿误差可以通过多任务网络结连估计，该网络是端对端训练的。像素集估计可以被直接解译为一种投票方案。然后使用均值平移的变形来聚合局部投票，同时通过设计在估计的三维姿势和像素方式的二维和三维估计之间实施一致性。我们的方法是有效且十分准确。在MARA和NYU两个手势数据集中，我们的方法优于了以往的先进方法很大一部分。在ICVL手势数据集中，我们的方法较完成较文章【5】中饱和结果相近准确率，并优于一些其他的方法。项目源码开放与网络【1】。      
![](https://github.com/leaguecn/leenotes/raw/master/img/Dense-3D-Regression-for-Hand-Pose-Estimation.png)

----------

#### Depth-Based 3D Hand Pose Estimation: From Current Achievements to Future Goals

*by Shanxin Yuan, Guillermo Garcia-Hernando, Bjorn Stenger, etc.*

**Abstract**      

In this paper, we strive to answer two questions: What is the current state of 3D hand pose estimation from depth images? And, what are the next challenges that need to be tackled? Following the successful Hands In the Million Challenge (HIM2017), we investigate the top 10 state-ofthe-art methods on three tasks: single frame 3D pose estimation, 3D hand tracking, and hand pose estimation during object interaction. We analyze the performance of different CNN structures with regard to hand shape, joint visibility, view point and articulation distributions. Our findings include: (1) isolated 3D hand pose estimation achieves low mean errors (10 mm) in the view point range of [70, 120] degrees, but it is far from being solved for extreme view points; (2) 3D volumetric representations outperform 2D CNNs, better capturing the spatial structure of the depth data; (3) Discriminative methods still generalize poorly to unseen hand shapes; (4) While joint occlusions pose a challenge for most methods, explicit modeling of structure constraints can significantly narrow the gap between errors on visible and occluded joints.       
//在本文，我们力求回答两个问题：当前的深度图像三维手势估计处于什么阶段？下一个需要解决的挑战又是什么？跟随在HIM2017大赛中的成功手势识别方法，我们研究了最顶尖的10种方法在物体交互的三个任务：单帧三维姿态估计、三维手掌处理和手势估计中的表现。我们分析了基于手掌形状处理、联结可视，视点和表达配置的不同CNN结构的性能。我们发现以下几点：1）孤立的三维手势估计在文章[70, 120]所提角度的视点取得了低均差的结果，但是极限视点下的问题远没解决；2）三维体积再现物优于二维CNNs，更好地从深度数据捕获空间结构；3）方法明显生成了劣质的不可见手型；4）当联结闭合手势是多数方法的问题，明确的结构约束模型可以急剧缩小误差在可视和闭合接点。      



![](https://github.com/leaguecn/leenotes/raw/master/img/Depth-Based-3D-Hand-Pose-Estimation@From-Current-Achievements-to-Future-Goals.png)


-----------------

**2019-02-23**

#### DeLS-3D: Deep Localization and Segmentation with a 3D Semantic Map

**DeLS-3D：三维语义地图的深度定位和分割**      

*by Peng Wang, Ruigang Yang, Binbin Cao, Wei Xu, Yuanqing Lin*


**Abstract**     
For applications such as augmented reality, autonomous driving, self-localization/camera pose estimation and scene parsing are crucial technologies. In this paper, we propose a unified framework to tackle these two problems simultaneously. The uniqueness of our design is a sensor fusion scheme which integrates camera videos, motion sensors (GPS/IMU), and a 3D semantic map in order to achieve robustness and efficiency of the system. Specifically, we first have an initial coarse camera pose obtained from consumer-grade GPS/IMU, based on which a label map can be rendered from the 3D semantic map. Then, the rendered label map and the RGB image are jointly fed into a pose CNN, yielding a corrected camera pose. In addition, to incorporate temporal information, a multi-layer recurrent neural network (RNN) is further deployed improve the pose accuracy. Finally, based on the pose from RNN, we render a new label map, which is fed together with the RGB image into a segment CNN which produces perpixel semantic label. In order to validate our approach, we build a dataset with registered 3D point clouds and video camera images. Both the point clouds and the images are semantically-labeled. Each video frame has ground truth pose from highly accurate motion sensors. We show that practically, pose estimation solely relying on images like PoseNet [25] may fail due to street view confusion, and it is important to fuse multiple sensors. Finally, various ablation studies are performed, which demonstrate the effectiveness of the proposed system. In particular, we show that scene parsing and pose estimation are mutually beneficial to achieve a more robust and accurate system.       
//自我定位/相机姿态估计和场景解析是增强现实，自动驾驶应用中的重要技术。在本文中，我们体术了一种联结框架来同时解决这两个问题。我们设计框架的特点是一个传感器融合方案，可以有序整合视频、运动传感器（GPS/IMU）和三维的语义地图来组合成稳健而高效的系统。需要说明的是，我们首先需要有一个粗略的初始化相机姿态，从消费基本的GPS/IMU中获得，在该姿态上标记地图从三维语义地图中制成。然后制成的标记地图和彩色图像被统一投递到姿态CNN中，生成校正的相机姿态。而且，为了与实践信息共同协作，一个多层回归神经网络（RNN）被配置来提高姿态的精度。最后，基于RNN的姿态，我们制成了一张新的标记地图，将它和彩色图像一同投递进分割的CNN中，产生像素的语义标签。为了应用我们的方法，我们使用相机视频图像和配准好的三维点云数据搭建一个数据集。点云和图像都被语义标识。每一帧的视频具有地面真实姿态，从高精度的运动传感器获取。我们在实践中展示了单独依赖图像（如PoseNet）姿态估计可能失败是由于街区场景混淆，且证明了融合多传感器的重要性。最后，系列落地实验都被实施，证明了所提系统的有效性。特别是我们表明了场景解析和姿态估计是有益于实现更稳健和准确系统。     

![](https://github.com/leaguecn/leenotes/raw/master/img/DeLS-3D-Deep-Localization-and-Segmentation-with-a-3D-Semantic-Map.png)


----------------
#### Deeply Learned Filter Response Functions for Hyperspectral Reconstruction

**高光谱重建中的深度学习滤波响应函数**     

*by Shijie Nie, Lin Gu, Yinqiang Zheng, Antony Lam, Nobutaka Ono, Imari Sato*


**Abstract**      
Hyperspectral reconstruction from RGB imaging has recently achieved significant progress via sparse coding and deep learning. However, a largely ignored fact is that existing RGB cameras are tuned to mimic human trichromatic perception, thus their spectral responses are not necessarily optimal for hyperspectral reconstruction. In this paper, rather than use RGB spectral responses, we simultaneously learn optimized camera spectral response functions (to be implemented in hardware) and a mapping for spectral reconstruction by using an end-to-end network. Our core idea is that since camera spectral filters act in effect like the convolution layer, their response functions could be optimized by training standard neural networks. We propose two types of designed filters: a three-chip setup without spatial mosaicing and a single-chip setup with a Bayer-style 2x2 filter array. Numerical simulations verify the advantages of deeply learned spectral responses compared to existing RGB cameras. More interestingly, by considering physical restrictions in the design process, we are able to realize the deeply learned spectral response functions by using modern film filter production technologies, and thus construct data-inspired multispectral cameras for snapshot hyperspectral imaging.      
//彩色成像的高光谱重建已经取得了重要的进步通过稀疏编码和深度学习。但是，大众中忽略的事实是存在彩色相机转向模仿人类的三色感知，这样它们的光谱响应就不再是很理想的用于高光谱重建。在本文中，而不是使用彩色光谱响应，我们同时研究了理想相机的光谱响应函数（被应用在硬件上）和一种通过使用端对端网络的光谱重建绘图。我们的核心思想是由于相机光谱滤波行为效果和卷积层神经网络相似，它们的响应函数可以通过训练标准神经网络来标准化。我们提出了两种类型的设计滤波：1）没有空间镶嵌的三芯设置和带有贝叶斯风格的2*2滤波器列阵的单芯设置。系列的仿真实验验证了深度学习光谱响应比存在的彩色相机有优势。有趣的是，在设计过程中顾及物理约束，我们可以使用新式影片滤波产品技术来实现深度学习的光谱响应函数。这样，为高光谱快照成像构建了数据启发的多光谱相机。      

![](https://github.com/leaguecn/leenotes/raw/master/img/Deeply-Learned-Filter-Response-Functions-for-Hyperspectral-Reconstruction.png)

------------------
**2019-02-22**

#### Deep Spatial Feature Reconstruction for Partial Person Re-identification: Alignment-free Approach

**部分人体重新识别的深度空间特征重建：无对齐方法**     

*by Lingxiao He, Jian Liang, Haiqing Li and Zhenan Sun*      
**Abstract**     
Partial person re-identification (re-id) is a challenging problem, where only several partial observations (images) of people are available for matching. However, few studies have provided flexible solutions to identifying a person in an image containing arbitrary part of the body. In this paper, we propose a fast and accurate matching method to address this problem. The proposed method leverages Fully Convolutional Network (FCN) to generate fix-sized spatial feature maps such that pixel-level features are consistent. To match a pair of person images of different sizes, a novel method called Deep Spatial feature Reconstruction (DSR) is further developed to avoid explicit alignment. Specifically, DSR exploits the reconstructing error from popular dictionary learning models to calculate the similarity between different spatial feature maps. In that way, we expect that the proposed FCN can decrease the similarity of coupled images from different persons and increase that from the same person. Experimental results on two partial person datasets demonstrate the efficiency and effectiveness of the proposed method in comparison with several state-ofthe-art partial person re-id approaches. Additionally, DSR achieves competitive results on a benchmark person dataset Market1501 with 83.58% Rank-1 accuracy.       
//部分人体重新识别（Re-id）是一个挑战性问题，可用于匹配图像的仅是部分人体观测图像。然而，很少的研究有提供灵活的方法来识别图片中含有的模糊身体部位。在本文中，我们提出了一种快速准确的匹配方法来解决该问题。所提方法使用FCN来生成适合尺寸的空间特征地图，这样像素级别的特征是一致的。为了匹配不同尺寸的人体图像对，一种新颖的深方法：深度空间特征重建（DSR）被开发用于避免明确的对齐。切确来说，DSR充分利用流行指点学习模型的重建误差来计算不同空间特征图的相似度。以该方法，我们预估所提的FCN能够减少从不同人体的图像集的相似度和增加同一人的人体图像集相似度。在两个部分人体数据集中实验，与几个先进水平的部分人体重新识别方法比较，结果表明所提方法的有效性和效率。而且，在Market1501人体基准数据集竞赛中，DSR取得了83.58%的排名第一精度结果。      
![](https://github.com/leaguecn/leenotes/raw/master/img/Deep-Spatial-Feature-Reconstruction-for-Partial-Person-Re-identification-Alignment-free-Approach.png)


--------

#### Curve Reconstruction via the Global Statistics of Natural Curves
**自然曲线全局统计的曲线重建**     
*by Ehud Barnea and Ohad Ben-Shahar*
**Abstract**

Reconstructing the missing parts of a curve has been the subject of much computational research, with applications in image inpainting, object synthesis, etc. Different approaches for solving that problem are typically based on processes that seek visually pleasing or perceptually plausible completions. In this work we focus on reconstructing the underlying physically likely shape by utilizing the global statistics of natural curves. More specifically, we develop a reconstruction model that seeks the mean physical curve for a given inducer configuration. This simple model is both straightforward to compute and it is receptive to diverse additional information, but it requires enough samples for all curve configurations, a practical requirement that limits its effective utilization. To address this practical issue we explore and exploit statistical geometrical properties of natural curves, and in particular, we show that in many cases the mean curve is scale invariant and oftentimes it is extensible. This, in turn, allows to boost the number of examples and thus the robustness of the statistics and its applicability. The reconstruction results are not only more physically plausible but they also lead to important insights on the reconstruction problem, including an elegant explanation why certain inducer configurations are more likely to yield consistent perceptual completions than others.      
//重建曲线的缺失部分已经成为了多种计算研究的目标，如图像绘制和目标综合的应用等。在解决典型地基于寻求视觉愉悦或感官可信完整进程问题有不同的方法。在这个问题上我们聚焦于重建物理意义上位于下方的似然形状，通过利用全局统计的自然曲线。更明确地说，我们开发了一个重建模型，可以为给定的引导配置搜寻平均物理曲线。这个简单的模型兼有直接运算和容纳多样的附加信息两种特性，但是它要求足够的所有曲线配置样本，一个限制它的效率使用的实践要求。为了解决这个实践上的问题，我们钻研和开拓了自然曲线的统计几何特性，特别是我们在多场合下展示。均值曲线是尺度连续和通常可拓展。这依次允许提高样例的数量和统计的稳健以及它的应用能力。重建的结果不仅更加物理上可信且引导了重要的重建问题的洞察，包括一个合理的解释：为什么确定诱导配置比其他的更可能产生一致的感官的完整体。       
![](https://github.com/leaguecn/leenotes/raw/master/img/Curve-Reconstruction-via-the-Global-Statistics-of-Natural-Curves.png)



----------------------

**2019-02-21**

#### Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?

**时空三维CNNs可以重新追踪二维CNNs和ImageNet的历史么？**     
*by Kensho Hara, Hirokatsu Kataoka, Yutaka Satoh*     

**Abstract**     
The purpose of this study is to determine whether current video datasets have sufficient data for training very deep convolutional neural networks (CNNs) with spatio-temporal three-dimensional (3D) kernels. Recently, the performance levels of 3D CNNs in the field of action recognition have improved significantly. However, to date, conventional research has only explored relatively shallow 3D architectures. We examine the architectures of various 3D CNNs from relatively shallow to very deep ones on current video datasets. Based on the results of those experiments, the following conclusions could be obtained: (i) ResNet-18 training resulted in significant overfitting for UCF-101, HMDB-51, and ActivityNet but not for Kinetics. (ii) The Kinetics dataset has sufficient data for training of deep 3D CNNs, and enables training of up to 152 ResNets layers, interestingly similar to 2D ResNets on ImageNet. ResNeXt-101 achieved 78.4% average accuracy on the Kinetics test set. (iii) Kinetics pretrained simple 3D architectures outperforms complex 2D architectures, and the pretrained ResNeXt-101 achieved 94.5% and 70.2% on UCF-101 and HMDB-51, respectively. The use of 2D CNNs trained on ImageNet has produced significant progress in various tasks in image. We believe that using deep 3D CNNs together with Kinetics will retrace the successful history of 2D CNNs and ImageNet, and stimulate advances in computer vision for videos. The codes and pretrained models used in this study are publicly available1.      

//本文的目的是为训练带有时空三维核的非常深度的CNNs检验当前的视频数据是否含有有效的数据。近期在动作识别领域，三维CNNs的性能水平已经得到了很大提升。然而，当前传统的研究仅被利用到相对浅显的三维建筑物领域。我们在目前的视频数据集中核查各种建筑物的三维CNNs，从相对浅显的到非常深入的。基于这些实验的结果，我们得到了以下的结论：1）ResNet-18的训练导致相当的过拟合，无论是在UCF-101、HMDB-51、ActivityNet，但不包含Kinetics。2）Kinetics的数据集含有训练深度三维CNNs的冗余数据，且激活了训练提升到152层ResNets，有趣地与在ImageNet上的二维ResNets相似。ResNeXt-101在Kinetics测试数据集中取得了78.4%平均精度。3）预训练的Kinetics简单的三维建筑物优于复杂的二维建筑物，且预训练的ResNeXt-101分别在UCF-101和 HMDB-51取得了94.5%和70.2%的结果。在ImageNet上使用训练的二维CNNs已经在图像多任务中取得了重要的进展。我们相信将深度三维CNNs和Kinetics并用将能重新追踪二维CNNs和ImageNet，且使得视频方面的计算机视觉更进一步。本文代码和预训练模型现在已经公开。

![](https://github.com/leaguecn/leenotes/raw/master/img/Can-Spatiotemporal-3D-CNNs-Retrace-the-History-of-2D-CNNs-and-ImageNet.png)

--------

#### Coding Kendall’s Shape Trajectories for 3D Action Recognition

**为三维动作识别编码Kendall形状轨迹**     
*by Amor Ben Tanfous, Hassen Drira, Boulbaba Ben Amor*

**Abstract**    
Suitable shape representations as well as their temporal evolution, termed trajectories, often lie to non-linear manifolds. This puts an additional constraint (i.e., non-linearity) in using conventional machine learning techniques for the purpose of classification, event detection, prediction, etc. This paper accommodates the well-known Sparse Coding and Dictionary Learning to the Kendall’s shape space and illustrates effective coding of 3D skeletal sequences for action recognition. Grounding on the Riemannian geometry of the shape space, an intrinsic sparse coding and dictionary learning formulation is proposed for static skeletal shapes to overcome the inherent non-linearity of the manifold. As a main result, initial trajectories give rise to sparse code functions with suitable computational properties, including sparsity and vector space representation. To achieve action recognition, two different classification schemes were adopted. A bi-directional LSTM is directly performed on sparse code functions, while a linear SVM is applied after representing sparse code functions using Fourier temporal pyramid. Experiments conducted on three publicly available datasets show the superiority of the proposed approach compared to existing Riemannian representations and its competitiveness with respect to other recently-proposed approaches. When the benefits of invariance are maintained from the Kendall’s shape representation, our approach not only overcomes the problem of non-linearity but also yields to discriminative sparse code functions. 

     
//适合的形状代表和他们的时域变化，即轨迹，一样位于非线性的流中。这就将一个额外的约束（如：非线性）引入了使用传统的机器学习技术中，为了分类、检测、预测等。本文提出了一种在Kendall形状空间中著名的稀疏编码和字典学习方法，并阐明动作识别中三维骨骼的编码效果。基于Riemannian几何的形状空间，一个本质的稀疏编码和字典学学公式被提出，用于静态骨骼形状提取以解决附带的多流非线性。如结果所示，初始轨迹提升了具有合适计算性质的稀疏编码函数，包括稀疏性和向量空间代表。采用两种不同的分类方案来完成动作识别。双向LSTM(长短期记忆)被直接用于稀疏码函数，同时一种线性的SVM(支持向量机)运用在使用傅里叶时间金字塔的稀疏码函数出现之后。在三种公有数据集上进行了试验，结果显示了所提方法比其他现有的Riemannian再现方法更具有优越性，并比其他的近年方法具有优势。当Kendall形状再现所持有的不同的优势时，我们的方法不仅克服了非线性难点，而且提供具有分辨的稀疏码函数。        

![](https://github.com/leaguecn/leenotes/raw/master/img/Coding-Kendalls-Shape-Trajectories-for-3D-Action-Recognition.png)


----------------
**2019-02-20**

#### Augmenting Crowd-Sourced 3D Reconstructions using Semantic Detections
**使用语义检测的增强交叉源三维重建**     
*by True Price, Johannes L. Schonberger, Zhen Wei, Marc Pollefeys, Jan-Michael Frahm1*

**Abstract**     
Image-based 3D reconstruction for Internet photo collections has become a robust technology to produce impressive virtual representations of real-world scenes. However, several fundamental challenges remain for Structure-from-Motion (SfM) pipelines, namely: the placement and reconstruction of transient objects only observed in single views, estimating the absolute scale of the scene, and (suprisingly often) recovering ground surfaces in the scene. We propose a method to jointly address these remaining open problems of SfM. In particular, we focus on detecting people in individual images and accurately placing them into an existing 3D model. As part of this placement, our method also estimates the absolute scale of the scene from object semantics, which in this case constitutes the height distribution of the population. Further, we obtain a smooth approximation of the ground surface and recover the gravity vector of the scene directly from the individual person detections. We demonstrate the results of our approach on a number of unordered Internet photo collections, and we quantitatively evaluate the obtained absolute scene scales.     
//面向互联网图像集的基于图像三维重建技术已经成为出产真实场景的令人印象深刻的虚拟呈现方面的成熟技术。然后，在运动结构恢复（SFM)中依然存在几个重要大挑战，即：移动(短暂停留)物体的位置和结构仅可以在单一视角中观测，估计场景的真实尺度，和经常要恢复场景地表模型。我们提出了一种新方法，共同强调这些仍然存在SFM中的问题。特别之处，我们着眼于单一图像的人物检测和他们在存在模型中的准确位置。关于位置部分，我们所提方法也可以从物体的语义环境估计场景的真实尺度，构成人群高度的配置信息。此外，我们还可以获得平滑的地表近视面和从单一人物检测中直接地恢复重力向量。我们阐明了所提方法在一系列无序互联网图像集中的实验结果，并且定量估计了已经获得的绝对的场景尺度。
![](https://github.com/leaguecn/leenotes/raw/master/img/Augmenting-Crowd-Sourced-3D-Reconstructions-using-Semantic-Detections.png)


----
#### Automatic 3D Indoor Scene Modeling from Single Panorama
**单一全景图像中自动化室内三维场景建模**    
*by Yang Yang, Shi Jin, Ruiyang Liu, Sing Bing Kang, Jingyi Yu*

**Abstract**    
We describe a system that automatically extracts 3D geometry of an indoor scene from a single 2D panorama. Our system recovers the spatial layout by finding the floor, walls, and ceiling; it also recovers shapes of typical indoor objects such as furniture. Using sampled perspective subviews, we extract geometric cues (lines, vanishing points, orientation map, and surface normals) and semantic cues (saliency and object detection information). These cues are used for ground plane estimation and occlusion reasoning. The global spatial layout is inferred through a constraint graph on line segments and planar superpixels. The recovered layout is then used to guide shape estimation of the remaining objects using their normal information. Experiments on synthetic and real datasets show that our approach is state-of-the-art in both accuracy and efficiency. Our system can handle cluttered scenes with complex geometry that are challenging to existing techniques.     
//我们讲述了一个从单张二维全景图像中自动提取室内场景三维几何的系统。我们所提的系统通过寻找地板、墙壁和天花板恢复空间布局，同时可以恢复典型室内物体，如：家具的形状。通过已采样的透视亚视图，我们提取几何索引（如线条，灭点/消失点，方向图和曲面法向量）和语义索引（凸出和目标检测信息）。这些索引被用于地表面估计与闭合推理。全局空间布局通过线割约束图和平面超像素被估计。恢复的布局然后被用于引导遗留物体的形状估计通过他们的法向量信息。在合成和真实数据集中的实验显示了所提方法在精确度和效率均达到先进水平。我们的系统能够处理聚类场景，这类场景包含复杂几何，是当前所存在技术中的挑战。

![](https://github.com/leaguecn/leenotes/raw/master/img/Automatic-3D-Indoor-Scene-Modeling-from-Single-Panorama.png)


--------------------

**2019-02-19**



#### Alive Caricature from 2D to 3D

**二维到三维的生动漫画**     

*by Qianyi Wu, Juyong Zhang, Yu-Kun Lai, Jianmin Zheng and Jianfei Cai*
**Abstract**    
Caricature is an art form that expresses subjects in abstract, simple and exaggerated views. While many caricatures are 2D images, this paper presents an algorithm for creating expressive 3D caricatures from 2D caricature images with minimum user interaction. The key idea of our approach is to introduce an intrinsic deformation representation that has the capability of extrapolation, enabling us to create a deformation space from standard face datasets, which maintains face constraints and meanwhile is sufficiently large for producing exaggerated face models. Built upon the proposed deformation representation, an optimization model is formulated to find the 3D caricature that captures the style of the 2D caricature image automatically. The experiments show that our approach has better capability in expressing caricatures than those fitting approaches directly using classical parametric face models such as 3DMM and FaceWareHouse. Moreover, our approach is based on standard face datasets and avoids constructing complicated 3D caricature training sets, which provides great flexibility in real applications.      
//漫画是一种艺术的抽象表达方式，简单但视角夸张。然而，许多漫画是二维的图像，本文提出了一种从带有最小用户交互的二维漫画中制作表达丰富的三维漫画算法。所提方法的主要观点是：引入一种内部形变代表物，该代表具有外推能力，激发我们从标准面部数据集中创造一个变形空间。这种变形空间可维持面部约束，同时足够大到可以产夸张的面部模型。建立所提的变形代表物后，一种寻找三维漫画的优化模型被形成，可以自动地捕捉二维漫画图像。实验表明所提方法相对于那些直接利用传统参数化面部模型，如：3DMM和FaceWareHouse的拟合方法，拥有更好漫画表达能力。而且，我们的方法基于标准的面部数据集且避免了构造复杂三维漫画训练集，可以在实际应用中提供很好的灵活性。

![](https://github.com/leaguecn/leenotes/raw/master/img/Alive-Caricature-from-2D-to-3D.png)

------------
#### A Papier-Mache Approach to Learning 3D Surface Generation
**一种用于学习三维表面生成的混凝纸浆方法**    
*by Thibault Groueix, Matthew Fisher, Vladimir G. Kim, Bryan C. Russell, Mathieu Aubry*     

**Abstract**      
We introduce a method for learning to generate the surface of 3D shapes. Our approach represents a 3D shape as a collection of parametric surface elements and, in contrast to methods generating voxel grids or point clouds, naturally infers a surface representation of the shape. Beyond its novelty, our new shape generation framework, AtlasNet, comes with significant advantages, such as improved precision and generalization capabilities, and the possibility to generate a shape of arbitrary resolution without memory issues. We demonstrate these benefits and compare to strong baselines on the ShapeNet benchmark for two applications: (i) autoencoding shapes, and (ii) single-view reconstruction from a still image. We also provide results showing its potential for other applications, such as morphing, parametrization, super-resolution, matching, and co-segmentation.     
//我们介绍了一种用学习来生成三维形状表面的方法。所提方法将一个三维形状表示为一个参数化表面元素的集合，和生成体素网格或点云相比，自然地推断形状的表面代表物。在新颖性方面，我们的新形状生成框架，AtlasNet带有重要的先进，如：提高精度和归纳能力与生成任意分辨率形状可能性没有内存问题。我们阐明了这些优点并在ShapeNet测试数据上为两种应用：1）自动编码形状，2）单视静态图像重建，比较了长基线。我们同时提供的结果显示所提方法对于其他应用的潜在可能，如：变换，参数化，超分辨率，匹配和协作分割。        
![](https://github.com/leaguecn/leenotes/raw/master/img/A-Papier-Mache-Approach-to-Learning-3D-Surface-Generation.png)

-------

#### 3D Semantic Trajectory Reconstruction from 3D Pixel Continuum

**三维像素连续体的三维语义轨迹重建**    
*by Jae Shin Yoon, Ziwei Li & Hyun Soo Park*     


**Abstract**
This paper presents a method to assign a semantic label to a 3D reconstructed trajectory from multiview image streams. The key challenge of the semantic labeling lies in the self-occlusion and photometric inconsistency caused by object and social interactions, resulting in highly fragmented trajectory reconstruction with noisy semantic labels. We address this challenge by introducing a new representation called 3D semantic map—a probability distribution over labels per 3D trajectory constructed by a set of semantic recognition across multiple views. Our conjecture is that among many views, there exist a set of views that are more informative than the others. We build the 3D semantic map based on a likelihood of visibility and 2D recognition confidence and identify the view that best represents the semantics of the trajectory. We use this 3D semantic map and trajectory affinity computed by local rigid transformation to precisely infer labels as a whole. This global inference quantitatively outperforms the baseline approaches in terms of predictive validity, representation robustness, and affinity effectiveness. We demonstrate that our algorithm can robustly compute the semantic labels of a large scale trajectory set (e.g., millions of trajectories) involving real-world human interactions with object, scenes, and people.     
//文提出了一种新方法，该方法用于向从多视图像流中重建的三维轨迹分配语义的标签。
语义标签的主要挑战在于由物体和社会交互所引起的自闭塞与光度不一致，致使带有噪声语义标签的轨迹重建高度碎片化。我们通过引入称之为三维语义地图的新代表物来解决这一挑战，这样的语义地图是每个三维轨迹标签的概率分布图，这些三维轨迹由系列的语义多视识别物构成。我们结论是在多视情况下存在某一集合的视图是包含更多的信息，相比其他视图。我们构建了三维的语义地图，该地图基于视觉的似然性与二维识别物的置信和辨别最好的语义轨迹代表视图。我们使用这个三维语义地图和局部刚体信息计算的轨迹类同来精确推断标签作为整体。这个全局推断在预测正确性、代表鲁棒性和同类有效性方面优于其他的基本方法。我们阐明了我们的算法能够稳健地计算大尺度轨迹集（如：百万级别的轨迹）的语义标签，这些大尺度轨迹涉及了真实世界各种的人类与物体，场景和人物之间的交互。      
![](https://github.com/leaguecn/leenotes/raw/master/img/3D-Semantic-Trajectory-Reconstruction-from-3D-Pixel-Continuum.png)


---------

#### 3D-RCNN: Instance-level 3D Object Reconstruction via Render-and-Compare

**3D-RCNN: 实时级别的渲染和比较的三维目标重建**     
*by Abhijit Kundu, Yin Li, James M. Rehg*     

**Abstract**

We present a fast inverse-graphics framework for instance-level 3D scene understanding. We train a deep convolutional network that learns to map image regions to the full 3D shape and pose of all object instances in the image. Our method produces a compact 3D representation of the scene, which can be readily used for applications like autonomous driving. Many traditional 2D vision outputs, like instance segmentations and depth-maps, can be obtained by simply rendering our output 3D scene model. We exploit class-specific shape priors by learning a low dimensional shape-space from collections of CAD models. We present novel representations of shape and pose, that strive towards better 3D equivariance and generalization. In order to exploit rich supervisory signals in the form of 2D annotations like segmentation, we propose a differentiable Render-and-Compare loss that allows 3D shape and pose to be learned with 2D supervision. We evaluate our method on the challenging real-world datasets of Pascal3D+ and KITTI, where we achieve state-of-the-art results.     
//我们提出一种快速反向图框架，用于实时级别的三维场景理解。我们训练了了一个深度卷积网络，可以学习将图像区域绘制到图像中的所有实时目标的全部三维形状和姿态中。我们的方法可以生成紧密的三维场景代表物，能迅速用于像自动驾驶这类应用中。许多传统的二维视觉输出，像实时分割和深度图，可以通过简单地翻译我们输出的三维场景模型获得。通过从CAD模型集合中学习一种低维的形状空间，我们探索了明确类别的形状先验信息。我们提出新颖的形状姿态代表物，力求更好的三维等价和概括。为了以分割等二维注解形式开发丰富的监督信号，我们提出了一种可积分的渲染和对比损失方法，允许二维监督的三维形状和姿态被学习。我们在Pascal3D+和KITTI真实世界的挑战数据集中评估了该方法，取得了最先进的结果。     


![](https://github.com/leaguecn/leenotes/raw/master/img/3D-RCNN-Instance-level-3D-Object-Reconstruction-via-Render-and-Compare.png)


-------------------
**2019-02-18**


#### 3D Pose Estimation and 3D Model Retrieval for Objects in the Wild
**真实世界中目标的三维姿态估计和三维模型恢复**     
*by Alexander Grabner, Peter M. Roth, Vincent Lepetit*      

**Abstract**

We propose a scalable, efficient and accurate approach  to retrieve 3D models for objects in the wild. Our contribution is twofold. We first present a 3D pose estimation approach for object categories which significantly outperforms the state-of-the-art on Pascal3D+. Second, we use the estimated pose as a prior to retrieve 3D models which accurately represent the geometry of objects in RGB images. For this purpose, we render depth images from 3D models under our predicted pose and match learned image descriptors of RGB images against those of rendered depth images using a CNN-based multi-view metric learning approach. In this way, we are the first to report quantitative results for 3D model retrieval on Pascal3D+, where our method chooses the same models as human annotators for 50% of the validation images on average. In addition, we show that our method, which was trained purely on Pascal3D+, retrieves rich and accurate 3D models from ShapeNet given RGB images of objects in the wild.

//我们提出一种多尺度、高效和准确的真实世界三维模型恢复方法。我们的贡献可以达两倍。我们首先提出了一种目标物类别三维姿态估计方法，明显优于在Pascal3D+上最先进的方法。然后，我们将估计的姿态作为恢复三维模型的先验信息，可以准确地再现彩色图像中的目标物几何特征。为此，我们渲染了在预测姿态下三维模型的深度图并将已彩色图像的学习描述子与渲染的深度图像匹配，使用基于CNN的多视学习方法。这样，我们是首个在Pascal3D+上提交三维模型恢复量化结果的团队，我们的方法选择的同样的模型作为人物注解，可以得到平均50%图像可用率。此外，我们表明了所提方法，仅仅使用Pascal3D+数据训练，就从ShapeNet提供的真实世界目标彩色图像中恢复了丰富和准确的三维模型。     

![](https://github.com/leaguecn/leenotes/raw/master/img/3D-Pose-Estimation-and-3D-Model-Retrieval-for-Objects-in-the-Wild.png)


-------------------

#### 3D Object Detection with Latent Support Surfaces
**使用潜在的支持表面检测三维物体**     
*by Zhile Ren & Erik B. Sudderth*

**Abstract**
We develop a 3D object detection algorithm that uses latent support surfaces to capture contextual relationships in indoor scenes. Existing 3D representations for RGB-D images capture the local shape and appearance of object categories, but have limited power to represent objects with different visual styles. The detection of small objects is also challenging because the search space is very large in 3D scenes. However, we observe that much of the shape variation within 3D object categories can be explained by the location of a latent support surface, and smaller objects are often supported by larger objects. Therefore, we explicitly use latent support surfaces to better represent the 3D appearance of large objects, and provide contextual cues to improve the detection of small objects. We evaluate our model with 19 object categories from the SUN RGB-D database, and demonstrate state-of-the-art performance.     
//我们开发了一种三维模型检测算法，该算法使用潜在支持面来捕捉室内场景的背景关系。已有的彩色深度图像三维再现捕获目标类别的局部形状和表面，但是对于再现不同视觉风格的目标能力不足。小型物体的检测也是一方面的挑战性问题，因为在三维场景下的搜索空间是巨大的。然而，我们观测到许多的具有三维目标类别的形状变量可以被潜在支持面的位置所解释，且更小的目标经常被大型的目标所支撑着。因此，我们明确使用潜在支持面来更好地再现大型目标的三维表面，并提供背景索引来提高小型目标的检测。我们使用从SUN RGB-D数据集中的19个目标类别评估了所提模型，表明该模型的先进性能。     

![](https://github.com/leaguecn/leenotes/raw/master/img/3D-Object-Detection-with-Latent-Support-Surfaces.png)


----

#### 3D Registration of Curves and Surfaces using Local Differential Information

**使用局部微分信息进行三维曲线和表面配准**     
*by Carolina Raposo and Joao P. Barreto*


**Abstract**

This article presents for the first time a global method for registering 3D curves with 3D surfaces without requiring an initialization. The algorithm works with 2-tuples point+vector that consist in pairs of points augmented with the information of their tangents or normals. A closed-form solution for determining the alignment transformation from a pair of matching 2-tuples is proposed. In addition, the set of necessary conditions for two 2-tuples to match is derived. This allows fast search of correspondences that are used in an hypothesise-and-test framework for accomplishing global registration. Comparative experiments demonstrate that the proposed algorithm is the first effective solution for curve vs surface registration, with the method achieving accurate alignment in situations of small overlap and large percentage of outliers in a fraction of a second. The proposed framework is extended to the cases of curve vs curve and surface vs surface registration, with the former being particularly relevant since it is also a largely unsolved problem.     
本文首次提出了三维曲线与三维曲面配准方法，该方法不要求初始化参数。算法使用2簇由点对和它们的正切值或法向量增强组成的点和向量工作。一种紧凑型方案提出来检测匹配2簇的对齐变换信息。而且，两个匹配2簇的必需条件集被提取。该方法允许快速搜索用于完成全局匹配的假设测试框架的匹配对。对比实验表明了所提算法是曲线和表面匹配的最有效解决方案，该方法在小重叠的情况下实现精确对准，并在几分之一秒内实现大比例的异常值。所提出的框架被拓展运用于曲线与曲线和表面与表面配准中，对于前者来说是非常相关，因为它是一个广泛存在的未解问题。      


![](https://github.com/leaguecn/leenotes/raw/master/img/3D-Registration-of-Curves-and-Surfaces-using-Local-Differential-Information.png)


--------------


#### 3D Semantic Segmentation with Submanifold Sparse Convolutional Networks

**子流形稀疏卷积网络的三维语义分割**    
*by Benjamin Graham, Martin Engelcke and Laurens van der Maaten*      

**Abstract**

Convolutional networks are the de-facto standard for analyzing spatio-temporal data such as images, videos, and 3D shapes. Whilst some of this data is naturally dense (e.g., photos), many other data sources are inherently sparse. Examples include 3D point clouds that were obtained using a LiDAR scanner or RGB-D camera. Standard “dense” implementations of convolutional networks are very inefficient when applied on such sparse data. We introduce new sparse convolutional operations that are designed to process spatially-sparse data more efficiently, and use them to develop spatially-sparse convolutional networks. We demonstrate the strong performance of the resulting models, called submanifold sparse convolutional networks (SSCNs), on two tasks involving semantic segmentation of 3D point clouds. In particular, our models outperform all prior state-of-the-art on the test set of a recent semantic segmentation competition.      
//卷积网络是分析时空数据，如：图像、视频和三维形状的实际标准技术。当一些数据是自然稠密的（如像片），许多其他的数据源是稀疏的。样例包括使用LiDAR扫描器或彩色深度相机获得的三维点云。当应用于稀疏的数据，标准的稠密卷积网络是非常无效的。我们提出了新的稀疏卷积运算，被设计用于更加高效处理空间稀疏数据，并使用它们开发了空间稀疏卷积网络。我们在两个涉及的三维点云语义分割任务中表明了结果模型强大性能，这些模型称为子流稀疏卷积网络（SSCNs）。特别地，在近年的语义分割竞赛测试数据集中，我们的模型优于所有的先进方法。

![](https://github.com/leaguecn/leenotes/raw/master/img/3D-Semantic-Segmentation-with-Submanifold-Sparse-Convolutional-Networks.png)



---------------------
**2019-02-17**

#### 2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning

*by Diogo C. Luvizon, David Picard, Hedi Tabia*    
**Abstract**    
Action recognition and human pose estimation are closely related but both problems are generally handled as distinct tasks in the literature. In this work, we propose a multitask framework for jointly 2D and 3D pose estimation from still images and human action recognition from video sequences. We show that a single architecture can be used to solve the two problems in an efficient way and still achieves state-of-the-art results. Additionally, we demonstrate that optimization from end-to-end leads to significantly higher accuracy than separated learning. The proposed architecture can be trained with data from different categories simultaneously in a seamlessly way. The reported results on four datasets (MPII, Human3.6M, Penn Action and NTU) demonstrate the effectiveness of our method on the targeted tasks.    
动作识别和人体姿态估计十分相似但在研究中两者通常是作为不同任务来处理。在本文中，我们提出了一种多任务式框架，可以同时进行静态图像的2D和3D姿态估计以及视频系列的人体动作识别。在文中，我们展示了一个单一的设计被用来有效地解决这两个问题且取得了达到了国际前沿水平的结果。此外，我们解释了从端对端的优化会得到相对于分别单独学习更高的精度。所提出的框架设计可以无缝地由不同类别数据同时训练而成。在4种数据集（MPII, Human3.6M, Penn Action and NTU）实验的结果显示了本文所提方法的有效性。

![](https://github.com/leaguecn/leenotes/raw/master/img/Pose-Estimation-and-Action-Recognition-using-Multitask-Deep-Learning.png)


---

#### 3D Human Pose Estimation in the Wild by Adversarial Learning

*by Wei Yang, Wanli Ouyang, Xiaolong Wang, Jimmy Ren, Hongsheng Li, Xiaogang Wang*

**Abstract**    
Recently, remarkable advances have been achieved in 3D human pose estimation from monocular images because of the powerful Deep Convolutional Neural Networks (DCNNs). Despite their success on large-scale datasets collected in the constrained lab environment, it is difficult to obtain the 3D pose annotations for in-the-wild images. Therefore, 3D human pose estimation in the wild is still a challenge. In this paper, we propose an adversarial learning framework, which distills the 3D human pose structures learned from the fully annotated dataset to in-the-wild images with only 2D pose annotations. Instead of defining hard-coded rules to constrain the pose estimation results, we design a novel multi-source discriminator to distinguish the predicted 3D poses from the ground-truth, which helps to enforce the pose estimator to generate anthropometrically valid poses even with images in the wild. We also observe that a carefully designed information source for the discriminator is essential to boost the performance. Thus, we design a geometric descriptor, which computes the pairwise relative locations and distances between body joints, as a new information source for the discriminator. The efficacy of our adversarial learning framework with the new geometric descriptor has been demonstrated through extensive experiments on widely used public benchmarks. Our approach significantly improves the performance compared with previous state-of-the-art approaches.    
近年来，由于强大的深度卷积神经网络（DCNNs）,单目图像的3D人体姿态估计取得了令人瞩目的成就。尽管他们在一些实验环境下采集的大尺度数据集中获得了成功，但是在实际环境中获得3D姿态参数仍然存在难点。因此，在实际环境中的人体姿态估计仍是一个挑战。在本文中，我们提出了一种对抗学习框架，能够在完全的标注数据集以及实际环境中简单标注2D姿态的图像学习情况下提取3D人体姿态结构。而不是定义硬码规则来约束姿态估计结果，我们设计了一种新颖的多源识别器来分辨从地表真实预测的3D姿态，这种识别器能够协助加强姿态估计算子生成人体有效的姿态，即使是在实际环境中的图像。我们同时观测到精密地设计分辨器的信息源可以真实提升其性能。为此，我们设计了一个计算身体节点关于位置和距离的匹配对的几何描述子，作为分辨器的新信息源。附有新几何描述子的对抗学习框架性能已经通过广泛利用公共测试数据进行拓展实验的结果阐明。相比于先前的世界前沿水平方法，我们的方法有效地提升了其中的性能。

![](https://github.com/leaguecn/leenotes/raw/master/img/3D-Human-Pose-Estimation-in-the-Wild-by-Adversarial-Learning.png)


---
#### 3D Human Sensing, Action and Emotion Recognition in Robot Assisted Therapy of Children with Autism

*by Elisabeta Marinoiu, Mihai Zanfir, Vlad Olaru, Cristian Sminchisescu*


**Abstract**    
We introduce new, fine-grained action and emotion recognition tasks defined on non-staged videos, recorded during robot-assisted therapy sessions of children with autism. The tasks present several challenges: a large dataset with long videos, a large number of highly variable actions, children that are only partially visible, have different ages and may show unpredictable behaviour, as well as non-standard camera viewpoints. We investigate how state-of-the-art 3d human pose reconstruction methods perform on the newly introduced tasks and propose extensions to adapt them to deal with these challenges. We also analyze multiple approaches in action and emotion recognition from 3d human pose data, establish several baselines, and discuss results and their implications in the context of child-robot interaction.    

我们介绍了一种新、细粒的动作和情绪识别任务，这种任务广发存在于未上演的视频，视频记录了自闭症儿童在机器协助治疗期间的情景。这种任务存在几种挑战：1）大数据集中有长时段视频，大量的有效动作，儿童部分可视，有不同年龄段和可能存在不可预测行为，同时也是在非标准的相机视角。我们研究了当前世界前沿水平3D人体姿态重建方法在这种新式环境下的性能如何，并基于此方法进行改善，使之适应并能处理这些挑战。同时，我们分析了多种从3D人体姿态数据中识别动作和情绪方法，建立了几种基线，并讨论了结果和他们在儿童-机器人交互环境相的含意。


![](https://github.com/leaguecn/leenotes/raw/master/img/3D-Human-Sensing-Action-and-Emotion-Recognition-in-Robot-Assisted-Therapy-of-Children-with-Autism.png)

-----

#### VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection

*by Yin Zhou & Oncel Tuzel*


**Abstract**      
Accurate detection of objects in 3D point clouds is a central problem in many applications, such as autonomous navigation, housekeeping robots, and augmented/virtual reality. To interface a highly sparse LiDAR point cloud with a region proposal network (RPN), most existing efforts have focused on hand-crafted feature representations, for example, a bird’s eye view projection. In this work, we remove the need of manual feature engineering for 3D point clouds and propose VoxelNet, a generic 3D detection network that unifies feature extraction and bounding box prediction into a single stage, end-to-end trainable deep network. Specifically, VoxelNet divides a point cloud into equally spaced 3D voxels and transforms a group of points within each voxel into a unified feature representation through the newly introduced voxel feature encoding (VFE) layer. In this way, the point cloud is encoded as a descriptive volumetric representation, which is then connected to a RPN to generate detections. Experiments on the KITTI car detection benchmark show that VoxelNet outperforms the state-of-the-art LiDAR based 3D detection methods by a large margin. Furthermore, our network learns an effective discriminative representation of objects with various geometries, leading to encouraging results in 3D detection of pedestrians and cyclists, based on only LiDAR.

在三维点云中准确的目标检测是许多应用的主要问题，如自动导航，家用机器人和增强/虚拟现实。为了使用区域提案网络(RPN)连接高度稀疏的LiDAR点云，许多现有的研究都集中于人工特征再现，如：鹰眼投影。在本文中，我们摒弃人工的3D点云特征工程需求，提出了体素网络(VoxelNet)，一种普通的能够统一特征提取和边界框预测到单阶段的3D检测网络，端对端可训练深度网络。特别之处是体素网络将点云分摊到均匀空间的3D体素并通过新引入的体素特征编码层(VFE)将一组带有体素的点转变为单个统一特征代表。使用这种方法，点云被编码为可描述的体积代表，体积代表被连接到一个区域提案网络(RPN)来生成检测子。在KITTI汽车检测测试数据实验中显示体素网络（VoxelNet）性能超过当前世界前沿的基于LiDAR 3D检测方法一大截。此外，我们的网络使用不同几何学习了一种有效的分辨目标代表，在3D行人、自行车检测中取得了振奋人心的结果，仅仅基于LiDAR数据。

![](https://github.com/leaguecn/leenotes/raw/master/img/VoxelNet-End-to-End-Learning-for-Point-Cloud-Based-3D-Object-Detection.png)








