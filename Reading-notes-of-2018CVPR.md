

## Reading notes of 2018CVPR

<!-- MarkdownTOC levels="1,2,3,4" autolink="true" style="ordered" -->

1. [3D pose or shape reconstruction](#3d-pose-or-shape-reconstruction)
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
*by Qianyi Wu1, Juyong Zhang, Yu-Kun Lai, Jianmin Zheng and Jianfei Cai*

**Abstract**
Caricature is an art form that expresses subjects in abstract, simple and exaggerated views. While many caricatures are 2D images, this paper presents an algorithm for creating expressive 3D caricatures from 2D caricature images with minimum user interaction. The key idea of our approach is to introduce an intrinsic deformation representation that has the capability of extrapolation, enabling us to create a deformation space from standard face datasets, which maintains face constraints and meanwhile is sufficiently large for producing exaggerated face models. Built upon the proposed deformation representation, an optimization model is formulated to find the 3D caricature that captures the style of the 2D caricature image automatically. The experiments show that our approach has better capability in expressing caricatures than those fitting approaches directly using classical parametric face models such as 3DMM and FaceWareHouse. Moreover, our approach is based on standard face datasets and avoids constructing complicated 3D caricature training sets, which provides great flexibility in real applications.
//漫画是一种艺术的抽象表达方式，简单但视角夸张。然而，许多漫画是二维的图像，本文提出了一种从带有最小用户交互的二维漫画中制作表达丰富的三维漫画算法。所提方法的主要观点是：引入一种内部形变代表物，该代表具有外推能力，激发我们从标准面部数据集中创造一个变形空间。这种变形空间可维持面部约束，同时足够大到可以产夸张的面部模型。建立所提的变形代表物后，一种寻找三维漫画的优化模型被形成，可以自动地捕捉二维漫画图像。实验表明所提方法相对于那些直接利用传统参数化面部模型，如：3DMM和FaceWareHouse的拟合方法，拥有更好漫画表达能力。而且，我们的方法基于标准的面部数据集且避免了构造复杂三维漫画训练集，可以在实际应用中提供很好的灵活性。

![](https://github.com/leaguecn/leenotes/raw/master/img/Alive-Caricature-from-2D-to-3D.png)

------------
#### A Papier-Mache Approach to Learning 3D Surface Generation

*by Thibault Groueix, Matthew Fisher, Vladimir G. Kim, Bryan C. Russell, Mathieu Aubry*

**Abstract**
We introduce a method for learning to generate the surface of 3D shapes. Our approach represents a 3D shape as a collection of parametric surface elements and, in contrast to methods generating voxel grids or point clouds, naturally infers a surface representation of the shape. Beyond its novelty, our new shape generation framework, AtlasNet, comes with significant advantages, such as improved precision and generalization capabilities, and the possibility to generate a shape of arbitrary resolution without memory issues. We demonstrate these benefits and compare to strong baselines on the ShapeNet benchmark for two applications: (i) autoencoding shapes, and (ii) single-view reconstruction from a still image. We also provide results showing its potential for other applications, such as morphing, parametrization, super-resolution, matching, and co-segmentation.

![](https://github.com/leaguecn/leenotes/raw/master/img/A-Papier-Mache-Approach-to-Learning-3D-Surface-Generation.png)

-------

#### 3D Semantic Trajectory Reconstruction from 3D Pixel Continuum

*by Jae Shin Yoon, Ziwei Li & Hyun Soo Park*


**Abstract**
This paper presents a method to assign a semantic label to a 3D reconstructed trajectory from multiview image streams. The key challenge of the semantic labeling lies in the self-occlusion and photometric inconsistency caused by object and social interactions, resulting in highly fragmented trajectory reconstruction with noisy semantic labels. We address this challenge by introducing a new representation called 3D semantic map—a probability distribution over labels per 3D trajectory constructed by a set of semantic recognition across multiple views. Our conjecture is that among many views, there exist a set of views that are more informative than the others. We build the 3D semantic map based on a likelihood of visibility and 2D recognition confidence and identify the view that best represents the semantics of the trajectory. We use this 3D semantic map and trajectory affinity computed by local rigid transformation to precisely infer labels as a whole. This global inference quantitatively outperforms the baseline approaches in terms of predictive validity, representation robustness, and affinity effectiveness. We demonstrate that our algorithm can robustly compute the semantic labels of a large scale trajectory set (e.g., millions of trajectories) involving real-world human interactions with object, scenes, and people.


![](https://github.com/leaguecn/leenotes/raw/master/img/3D-Semantic-Trajectory-Reconstruction-from-3D-Pixel-Continuum.png)


---------

#### 3D-RCNN: Instance-level 3D Object Reconstruction via Render-and-Compare

*by Abhijit Kundu, Yin Li, James M. Rehg*

**Abstract**

We present a fast inverse-graphics framework for instance-level 3D scene understanding. We train a deep convolutional network that learns to map image regions to the full 3D shape and pose of all object instances in the image. Our method produces a compact 3D representation of the scene, which can be readily used for applications like autonomous driving. Many traditional 2D vision outputs, like instance segmentations and depth-maps, can be obtained by simply rendering our output 3D scene model. We exploit class-specific shape priors by learning a low dimensional shape-space from collections of CAD models. We present novel representations of shape and pose, that strive towards better 3D equivariance and generalization. In order to exploit rich supervisory signals in the form of 2D annotations like segmentation, we propose a differentiable Render-and-Compare loss that allows 3D shape and pose to be learned with 2D supervision. We evaluate our method on the challenging real-world datasets of Pascal3D+ and KITTI, where we achieve state-of-the-art results.



![](https://github.com/leaguecn/leenotes/raw/master/img/3D-RCNN-Instance-level-3D-Object-Reconstruction-via-Render-and-Compare.png)


-------------------
**2019-02-18**


#### 3D Pose Estimation and 3D Model Retrieval for Objects in the Wild
 
*by Alexander Grabner, Peter M. Roth, Vincent Lepetit*

**Abstract**

We propose a scalable, efficient and accurate approach  to retrieve 3D models for objects in the wild. Our contribution is twofold. We first present a 3D pose estimation approach for object categories which significantly outperforms the state-of-the-art on Pascal3D+. Second, we use the estimated pose as a prior to retrieve 3D models which accurately represent the geometry of objects in RGB images. For this purpose, we render depth images from 3D models under our predicted pose and match learned image descriptors of RGB images against those of rendered depth images using a CNN-based multi-view metric learning approach. In this way, we are the first to report quantitative results for 3D model retrieval on Pascal3D+, where our method chooses the same models as human annotators for 50% of the validation images on average. In addition, we show that our method, which was trained purely on Pascal3D+, retrieves rich and accurate 3D models from ShapeNet given RGB images of objects in the wild.
 
我们提出了

![](https://github.com/leaguecn/leenotes/raw/master/img/3D-Pose-Estimation-and-3D-Model-Retrieval-for-Objects-in-the-Wild.png)


-------------------

#### 3D Object Detection with Latent Support Surfaces

*by Zhile Ren & Erik B. Sudderth*

**Abstract**
We develop a 3D object detection algorithm that uses latent support surfaces to capture contextual relationships in indoor scenes. Existing 3D representations for RGB-D images capture the local shape and appearance of object categories, but have limited power to represent objects with different visual styles. The detection of small objects is also challenging because the search space is very large in 3D scenes. However, we observe that much of the shape variation within 3D object categories can be explained by the location of a latent support surface, and smaller objects are often supported by larger objects. Therefore, we explicitly use latent support surfaces to better represent the 3D appearance of large objects, and provide contextual cues to improve the detection of small objects. We evaluate our model with 19 object categories from the SUN RGB-D database, and demonstrate state-of-the-art performance.


![](https://github.com/leaguecn/leenotes/raw/master/img/3D-Object-Detection-with-Latent-Support-Surfaces.png)


----

#### 3D Registration of Curves and Surfaces using Local Differential Information


*by Carolina Raposo and Joao P. Barreto*


**Abstract**

This article presents for the first time a global method for registering 3D curves with 3D surfaces without requiring an initialization. The algorithm works with 2-tuples point+vector that consist in pairs of points augmented with the information of their tangents or normals. A closed-form solution for determining the alignment transformation from a pair of matching 2-tuples is proposed. In addition, the set of necessary conditions for two 2-tuples to match is derived. This allows fast search of correspondences that are used in an hypothesise-and-test framework for accomplishing global registration. Comparative experiments demonstrate that the proposed algorithm is the first effective solution for curve vs surface registration, with the method achieving accurate alignment in situations of small overlap and large percentage of outliers in a fraction of a second. The proposed framework is extended to the cases of curve vs curve and surface vs surface registration, with the former being particularly relevant since it is also a largely unsolved problem.




![](https://github.com/leaguecn/leenotes/raw/master/img/3D-Registration-of-Curves-and-Surfaces-using-Local-Differential-Information.png)


--------------


#### 3D Semantic Segmentation with Submanifold Sparse Convolutional Networks

*by Benjamin Graham, Martin Engelcke and Laurens van der Maaten*

**Abstract**

Convolutional networks are the de-facto standard for analyzing spatio-temporal data such as images, videos, and 3D shapes. Whilst some of this data is naturally dense (e.g., photos), many other data sources are inherently sparse. Examples include 3D point clouds that were obtained using a LiDAR scanner or RGB-D camera. Standard “dense” implementations of convolutional networks are very inefficient when applied on such sparse data. We introduce new sparse convolutional operations that are designed to process spatially-sparse data more efficiently, and use them to develop spatially-sparse convolutional networks. We demonstrate the strong performance of the resulting models, called submanifold sparse convolutional networks (SSCNs), on two tasks involving semantic segmentation of 3D point clouds. In particular, our models outperform all prior state-of-the-art on the test set of a recent semantic segmentation competition.


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








