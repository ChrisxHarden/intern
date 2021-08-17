# BiseNet 阅读笔记
## 灵感来源:
需要同时注意 High accuracy & efficient inference speed
1. High accuracy: 
   two backbones
   1. Dilatiaon Backbone 
   2. Enconder decoder 
   问题:不关注计算速度和计算花销
2. efficient inference speed
   Two solutions:
   1. Input restriciting:限制输入图形的尺寸
   2. Channel pruning:
   问题:丢失细节
### what is BiseNEt 
1. 在语义分割中,low-level details 和 high level semantics都很重要
2. 在常见的模型里同时处理,在这个模型里分开处理
3. *detail branch*:捕捉空间细节 运用wide channel 和 shallow layers
4. *semantic branch*语义提取 运用narrow channel 和 deep layers
5. 细节由detail branch 补充,semantic branch 只需要大感受野来读取全图信息,因此可以做的很轻量,downsampling 可以很快(无需担心细节,快速收缩)
6. 使用guided aggregation layer 来融合两组特征
7. 使用booster 训练计划


## Related work
### generic
1. dilation backbone 去除下采样操作 upsample convolution filter 保留高精度细节
2. 引入attention mechanism来保留long-range context
3. encoder-decoder backbone加入例外的top-down & lateral连接来在decoder部分保留细节 
   e.g.:FCN,Segnet,RefineNet,LRR,GCN(global convlution networks)
4. 采用过大的网络来同时进行对底层细节和上层语义的编码,速度太慢
   
### real-time
1. Segnet 小的网络结构,快速
2. E-Net
3. ICNET 采用image cascade来加速
4. DLC cascade network structure 来加速高置信区域的推理速度
5. DFAnet 重用特征来增强特征表示和减少计算复杂度
6. 通病:损失精度
### light weight 
1. depth-wise convolution separable convolution 理论基础
2. Xception mobileNet, ShuffleNEt


## Core concept
### Detail Branch
1. resposible for spatial details,i.e. low-level information
2. require rich channel capacity to encode affluent spatial detailed information
3. design shallow structure with small stride
4. whole point : shallow layers and wide channels
5. due to large spatial size and wide channels , not adopt residual connections ,which will incerases the memory access cost and reduce the speed
   
### Semantic Branch
1. low channel capacity , can be any light weight convolution model 
2. fast-sown-sampling strategy ,promote feature representation ,enlarge rteceptive field
3. employ global average pooling
### Aggregation Layer
1. Both branch are complementary, need each other
2. saptial dimension of semantic branch is smaller than spatial branch, upsample it to match
3. Use bidirectional arrgregation method to Fuse information


## Details and the whole model
### detial branch
和 VGGnet 有相似之处
### Semantic  Branch
key features:
1. __Stem Block__ :semantic branch 的第一层,用两种不同的下采样方法,挤压特征表示,最终两个branch的output feature会concatenate在一起做输出,有很好的计算花销和特征表示能力.(为什么是连接, 换成别的会怎么样)
2. __Context Embedding Block__:Semantic Branch 需要大的感受野,采用residual connection和global average pooling来embed global textual information 
3. __Gather and Expansion Layer__:
   1. 3*3的conv 收集 local feature ,expand to higher-dimensional space
   2. 3*3的Depthwise conv:对先前一层的输出的每个channel独立运算
   3. 1*1的conc调整channel
当depthwise的stride为2时,有两个depthwise conv在主通道上,还有一个在shortcut上配备一个1*1的CONV调节channel
### Bilateral Guided Aggregation
简单的merge会因为两个branch上信息的不同层级导致表现不佳
employ contextual information of Semantic branch to guide the feature response of detail branch
不同层级的guidance可以捕获不同scale的feature representation,而且能使两个branch之间有更有效的沟通 
### Booster Training Strategy
提升segmentation acc,enhance feature representation in the training phase, discard in inference phase.


## Experimental Result

   
