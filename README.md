目前进度：

从零实现了lenet5，为了提升ACC略微修改了原版网络的结构，池化层由平均池化改为最大池化。训练过程中，激活函数选择使用带权重衰减的adam。----2021.07.06

从零实现了AlexNet，我的显存太小啦，batch_size调到2也训练不起来，也可能不是显卡的锅，是我太蠢了吧.....网上找了个容器训练，也挺慢的(4核心8G的GPU)，先放一放研究下代码，不行就先凑合着训练。----2021.07.16

调整了lenet训练时的batch_size，选用尽量大的值(目前使用64)，lr相应也选择较大的值(目前使用2e-3，学习到一个trick：大batch_size使用较大的lr，效果极好)，训练35轮后acc=99.1%。----2021.07.17

从零实现了vgg_net，受制于硬件性能限制，暂时不做训练。----2021.07.18

学习了其他大佬实现的SSD&YOLOv4。----2021.07.17
