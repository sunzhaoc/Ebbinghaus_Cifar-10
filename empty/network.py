'''
@Description: 
@Version: 1.0
@Autor: Vicro
@Date: 2020-07-24 21:28:34
@LastEditors: Vicro
@LastEditTime: 2020-07-24 21:29:55
https://blog.csdn.net/AugustMe/article/details/93914203?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase
'''

#from torch import nn
import torch.nn as nn
import torchvision.models as models # 导入预训练模型（训练好的）

class feature_net(nn.Module):
    def __init__(self,model,dim,n_classes):
        super(feature_net,self).__init__()
        if model == 'vgg19':
            vgg19 = models.vgg19(pretrained=True)
            self.feature = nn.Sequential(*list(vgg19.children())[:-1])
            self.feature.add_module('gloabl average',nn.AvgPool2d(9))
        elif model == 'inception_v3':
            inception_v3 = models.inception_v3(pretrained=True)
            self.feature = nn.Sequential(*list(inception_v3.children())[:-1])
            self.feature._modules.pop('13')
            self.feature.add_module('global average',nn.AvgPool2d(35))
        elif model == 'resnet152':
            resnet152 = models.resnet152(pretrained=True)
            self.feature = nn.Sequential(*list(resnet152.children())[:-1])
    
        self.classifier = nn.Sequential(nn.Linear(dim,4096),
                                        nn.ReLU(inplace=True),
                                    nn.Dropout(0.5),
                                    nn.Linear(4096,4096),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.5),
                                    nn.Linear(4096,n_classes)
                                    )
        
    def forward(self,x):
            x = self.feature(x)
            x = x.view(x.size(0),-1)
            x = self.classifier(x)
            return x

# 查看其中一个模型结构，训练好的模型需要下载      
#model = feature_net('vgg19',10,2)
#print(model) 
