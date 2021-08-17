# 工作:

## 模型搭建与训练:
模型训练和搭建文件:trail.py
### 简述:
1. 数据输入部分采用imagegenerator,在读入数据后默认将图片resize为256*256,尚未进行亮度数据增强工作,根据已有的模型推测结果来看会出现高光部分辨认为前景的问题,数据增强后应该可以提升部分正确率
2. 模型优化器采用Adam,loss fuction采用binary cross entropy,此外设置了reduce LR on Plateau, Min_LR设为1e-6
3. 训练部分目前只训练了100个epoch,根据在github上类似项目推测仍需继续,暂定为300.由于之前未设置保存检查点权重导致一定程度的时间损失.
## 测试
1. 测试情况见test_result_1.png,测试文件:load_model.py
前五张为sythetic中的图片,由于该类型图片(合成图,和训练集有光照,肤色和姿势区别)未放入训练集中,因此预测表现不如后五张和训练集同类型的照片.仅测试图片
2. 当前的模型:trained_best_weight.h5
## 其余
1. 阅读background_matting_v2论文和源代码(模型太大而且需要背景图来实现所需功能,但是分割能力极好)https://github.com/PeterL1n/BackgroundMattingV2
2. 测试了portrait segmentation中的部分模型,阅读源代码和说明文档.https://github.com/anilsathyan7/Portrait-Segmentation
3. 阅读了Modnet论文,但是在colab测试中由于gpu限额未测试成功 https://github.com/ZHKKKe/MODNet
# 计划:

1. 增加独立的python文件用来向训练文件传输以下参数:epoch,batchsize,imgshape,训练集文件夹
2. 增加和其他模型的对比(Bisenet,mobilenetv2,mobilenetv3,MODnet,SInet,slimnet),图片分割metrics采用MIOU和accuracy.
3. 读SINet,以及更新的文章(轻量级语义分割 相关方向)
4. 补充计算机图形学的知识(may take long)
# 更改:
2021/8/10: 增加camera在模型上的demo 测试了已有模型和其他几个模型
2021/8/16: 增加对比的tool和结果
2021/8/17: 增加亮度变化
# 过往更改(倒序):
1. 将数据输入改为imagegenerator,之前采用load.py进行导入,后来使用transform.py将各个文件夹中的图片放在同一个文件夹下方便Flow_from_directory读取.未改进前每次训练需要两个小时读入训练集,经常出现花了两个小时读完训练集后显存已经不足分配导致无法训练
2. 增加reduce_LR_on_Plateau
3. 增加保存检查点功能
4. 从cpu运算转为gpu,使用tensorflow版本从2.5.0转化为2.3.0,进行适配性更改
5. 数据集更改
# 问题:

1. 现有训练集输出的是grayscale的matte图,测试结果只有遮罩效果,如何实现背景虚化,是否需要更改数据集?还是说在生成遮罩基础上将白色部分保留为原图,黑色部分根据需求进行不同运算?(已解决)

