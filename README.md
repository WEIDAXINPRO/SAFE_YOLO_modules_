| 文件 | 主接口 | 本地内部组件 |
|---|---|---|
| c3k2_sf.py | C3k2_SF(c1,c2,n=1,...) | C3k_SF、Bottleneck、AKConv、FCA、Conv/BN/SiLU |
| sica.py | SICA(channels,...) | GAP、通道卷积、空间平均/最大池化、7×7卷积、交叉调制、重编码、残差 |
| paffn.py | PAFFN(in_channels,out_channels=None,...) | ScaleAlign、ASFF-2/3、CRDR、RFAConv、EMA、ConvEMA |

兼容别名：C3k2SF、C3K2_SF、MultiSpectralFCA、Conv_EMA。


```



## 已检索并读取的上游代码

| 算子 | 来源 | Git blob SHA |
|---|---|---|
| AKConv/LDConv | https://github.com/CV-ZhangXin/LDConv/blob/main/model.py | 98d10afa7a6fee627802f8c9960229c0686ac1f1 |
| FcaNet DCT分组 | https://github.com/cfzd/FcaNet/blob/master/model/layer.py | ef7bde4078d042e8bf84733947fc2ca7a162e8d2 |
| RFAConv group-conv | https://github.com/Liuchen1997/RFAConv/blob/main/model.py | 68d70c0f633bf0e9391ab9d6ad3d78d787c80529 |
| EMA | https://github.com/YOLOonMe/EMA-attention-module/blob/main/EMA_attention_module | 57414d738660a2c1c38147a83709dfbd2563728b |
| ASFF | https://github.com/ruinmessi/ASFF/blob/master/models/network_blocks.py | b3d495f87a2e70584cccf968f1f3bd28c6aeb41b |
| C3k2连接参考 | https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py | a71f77cd324a75201682c48959b13dba9ce3911c |

相关论文：FcaNet https://doi.org/10.1109/ICCV48922.2021.00082；LDConv https://doi.org/10.1016/j.imavis.2024.105190；EMA https://doi.org/10.1109/ICASSP49357.2023.10096516。RFAConv、ASFF论文信息可从各原作者仓库README追溯。

已核查结构后本地重写、移除自定义外部层及einops。代码公开不等于没有许可约束；正式公开或商业使用前，请核查来源项目和Ultralytics适用许可、保留必要作者致谢。本文件夹不擅自授予上游代码新许可。
