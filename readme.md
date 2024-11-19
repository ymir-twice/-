# 项目介绍
此系2024第六届全球校园人工智能算法精英大赛国一作品。
    - 竞赛名称: 全球校园人工智能算法精英大赛.算法挑战赛.钢材表面缺陷检测与分割
    - 主办方: 江苏省人工智能协会
    - 级别: 国家级
    - 作品奖项: 国一
    - 作者联系方式: QQ, 1164139715。欢迎大家前来交流竞赛相关事项。

# 目录介绍
    - .vscode/: 存放vscode任务配置文件
    - data/: 存放数据 
    - model-src/: 存放模型源代码。竞赛期间，一共查找到11种值得尝试的网络，
    - models/: 存放模型权重。其中networks.py中存放当前待测试模型的源码。
    - stat/s: 存放图像数据的统计表(csv格式)
    - utils/: 存放损失函数、可视化工具、学习率调度、损失函数、训练过程中的数据增强
    - activate_cpu.sh: 用于唤醒沉睡的CPU（GPU租赁平台上租了44核，默认有32都是睡着的，需要自行唤醒）
    - create_gt.py: 用于为data/images/test中存放的测试集数据生成 Ground Truth。
    - dataset.py: pytorch训练时所需的数据加载器
    - learn.ipynb: 搭建项目时为了学习新东西临时用的ipynb
    - make_dataset.ipynb: 同上，搭建项目时为了学习新东西临时用的ipynb
    - make_dataset.py: 对data/images/training_ori中的图片随机进行对角拼接、上下、左右拼接、添加随机强度的椒盐噪声、复制原图等操作，在data/images/training/中存放生成的数据（运行时不会先清除training中图片，因此，多运行几次该脚本可以生成很大的数据集）
    - predict.py: 读取指定的模型权重，加载到models/networds.py中定义的模型里，使用此权重和模型对data/images/test/中图片进行分割。
    - testworker.py: 测试并行使用几个核心载入数据最快。
    - train_GSSNet.py: 模型训练代码。里面的实现逻辑很好理解，便于替换为不同的网络进行训练。
    - train.ipynb: train_GSSNet.py的 notebook 版本。
    - vis_result.ipynb: 实现网络的过程中，拿来可视化结果的临时文件。

# 使用方法
    - 运行train_GSSNet.py可以训练模型。
    - 在vscode中，运行任务 predict 可以生成测试集的预测分割结果
    - 在vscode中，运行任务 score 可以获得某次预测结果的得分
    - 本项目的目录非常易懂，其他诸如修改损失函数、学习器调度、数据增广、各种超参数调节等，可以自行参考学习源码

# 设计思路
对各种模型进行尝试后，最终选定使用ULite模型作为基准模型。将ULite模型针对竞赛任务做了小改动。中间也尝试过使用参数量比较小的通道注意力机制。但是最终放弃。只在ULite网络的上采样中添加了一个双线性插值，用于对齐跳跃连接前后的图像尺寸。

提高性能的核心点在数据增强上。核心的技术是: CLAHE + 高斯滤波。 加上这两个机制后，在省赛中性能（mIoU）直接提高两个点。此外，可视化结果中也很明显，CLAHE会使得钢材图片明显变得清晰。此外，拉普拉斯金字塔变换、中值滤波、椒盐噪声检测算法，在国赛中也大放异彩。马赛克、mixup技术后来被证明也能有效提高成绩。

最终方案止步顶榜第8，输在模型不够深，以及算力不足，最终模型没有收敛。

# 欢迎联系
我们是来自同济大学的一支队伍。欢迎大家同我们交流，交个朋友也可以哦:
QQ: 1164139715

# 参考文献

1. **CGNet**  
   Zhenbo Zhu et al. *CGNet: A Light-weight Context Guided Network for Semantic Segmentation*. arXiv preprint arXiv:1811.08201, 2018.

2. **DeepLabV3+**  
   Liang-Chieh Chen et al. *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*. Proceedings of the European Conference on Computer Vision (ECCV), 2018.

3. **E-Net**  
   Adam Paszke et al. *ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation*. arXiv preprint arXiv:1606.02147, 2016.

4. **PSPNet**  
   Hengshuang Zhao et al. *Pyramid Scene Parsing Network*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

5. **ULite**
    [官方github](https://github.com/duong-db/U-Lite)



