

## Reading notes of 2018CVPR

<!-- MarkdownTOC levels="1,2,3,4" autolink="true" style="ordered" -->

1. [3D pose or shape reconstruction](#3d-pose-or-shape-reconstruction)
    1. [2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning](#2d3d-pose-estimation-and-action-recognition-using-multitask-deep-learning)
    1. [3D Human Pose Estimation in the Wild by Adversarial Learning](#3d-human-pose-estimation-in-the-wild-by-adversarial-learning)
    1. [3D Human Sensing, Action and Emotion Recognition in Robot Assisted Therapy of Children with Autism](#3d-human-sensing-action-and-emotion-recognition-in-robot-assisted-therapy-of-children-with-autism)
    1. [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](#voxelnet-end-to-end-learning-for-point-cloud-based-3d-object-detection)

<!-- /MarkdownTOC -->

### 3D pose or shape reconstruction

*Reading notes of 2018CVPR, Computer Vision & Pattern Recognition 2018,  processings selected in 3D reconstruction aspect.*

-----

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


-----

#### VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection

*by Yin Zhou & Oncel Tuzel*


**Abstract** 
Accurate detection of objects in 3D point clouds is a central problem in many applications, such as autonomous navigation, housekeeping robots, and augmented/virtual reality. To interface a highly sparse LiDAR point cloud with a region proposal network (RPN), most existing efforts have focused on hand-crafted feature representations, for example, a bird’s eye view projection. In this work, we remove the need of manual feature engineering for 3D point clouds and propose VoxelNet, a generic 3D detection network that unifies feature extraction and bounding box prediction into a single stage, end-to-end trainable deep network. Specifically, VoxelNet divides a point cloud into equally spaced 3D voxels and transforms a group of points within each voxel into a unified feature representation through the newly introduced voxel feature encoding (VFE) layer. In this way, the point cloud is encoded as a descriptive volumetric representation, which is then connected to a RPN to generate detections. Experiments on the KITTI car detection benchmark show that VoxelNet outperforms the state-of-the-art LiDAR based 3D detection methods by a large margin. Furthermore, our network learns an effective discriminative representation of objects with various geometries, leading to encouraging results in 3D detection of pedestrians and cyclists, based on only LiDAR.

在三维点云中准确的目标检测是许多应用的主要问题，如自动导航，家用机器人和增强/虚拟现实。为了使用区域提案网络(RPN)连接高度稀疏的LiDAR点云，许多现有的研究都集中于人工特征再现，如：鹰眼投影。在本文中，我们摒弃人工的3D点云特征工程需求，提出了体素网络(VoxelNet)，一种普通的能够统一特征提取和边界框预测到单阶段的3D检测网络，端对端可训练深度网络。特别之处是体素网络将点云分摊到均匀空间的3D体素并通过新引入的体素特征编码层(VFE)将一组带有体素的点转变为单个统一特征代表。使用这种方法，点云被编码为可描述的体积代表，体积代表被连接到一个区域提案网络(RPN)来生成检测子。在KITTI汽车检测测试数据实验中显示体素网络（VoxelNet）性能超过当前世界前沿的基于LiDAR 3D检测方法一大截。此外，我们的网络使用不同几何学习了一种有效的分辨目标代表，在3D行人、自行车检测中取得了振奋人心的结果，仅仅基于LiDAR数据。

![](https://github.com/leaguecn/leenotes/raw/master/img/VoxelNet-End-to-End-Learning-for-Point-Cloud-Based-3D-Object-Detection.png)