# SAFE-YOLO 三个独立模块

依据用户提供的论文图1–5及公式(1)–(18)实现。仅含模块，没有完整检测器、检测头、训练流程或数据集。原DOCX和上级目录已有的旧代码均未修改。

三个文件均独立、仅依赖PyTorch；内部组件的类及前向计算全部展开，没有从Ultralytics或其他仓库导入自定义层。代码内部调用本文件定义的类属于普通PyTorch组合，不是外部黑盒调用。

| 文件 | 主接口 | 本地内部组件 |
|---|---|---|
| c3k2_sf.py | C3k2_SF(c1,c2,n=1,...) | C3k_SF、Bottleneck、AKConv、FCA、Conv/BN/SiLU |
| sica.py | SICA(channels,...) | GAP、通道卷积、空间平均/最大池化、7×7卷积、交叉调制、重编码、残差 |
| paffn.py | PAFFN(in_channels,out_channels=None,...) | ScaleAlign、ASFF-2/3、CRDR、RFAConv、EMA、ConvEMA |

兼容别名：C3k2SF、C3K2_SF、MultiSpectralFCA、Conv_EMA。

## 论文对应

### C3k2_SF

外层按图2：Conv→Split，保留两路初始张量；第二路顺序经过n个C3k_SF；两路初始张量和每个C3k_SF输出一起Concat→Conv。不是只拼接最终两支路。

内层C3k_SF：Conv→Split；第一支路经过Bottleneck堆叠，第二支路AKConv→FCA；两支路Concat→Conv，对应公式(1)–(3)。

- AKConv包含任意N点采样、学习偏移、双线性插值、高维重排和(N,1)卷积聚合。默认N=5来自图2的C,5H,W标签，不固定成普通3×3或9点卷积。
- FCA八频率严格按公式(12)：`[(0,0),(0,1),(1,0),(2,2),(3,3),(6,0),(0,6),(6,6)]`。使用归一化DCT-II基函数，不使用任意频带近似或上游top8代替。
- 按图2的C×1×1描述子和FcaNet分组方法，每组分配一个指定频率，拼接得C维描述子，经两层MLP和Sigmoid重标定。
- 默认在实际H×W上计算DCT；可显式设置`dct_size=(7,7)`选择先池化变体。默认频率含6，因此DCT网格H、W均至少7；尺寸不够明确报错，不静默截断频率。
- 隐藏通道向上对齐8的倍数，以支持八组频率。

### SICA

```text
F1,F2 = Split(Fin)                         # 每路C/2
Mc = Sigmoid(Conv1x1(GAP(F1)))              # B,C/2,1,1，公式14
Ms = Sigmoid(Conv7x7(Cat(Avg(F2),Max(F2)))) # B,1,H,W，公式15
Ffuse = F1 * Ms + F2 * Mc                   # 公式16，相加不是拼接
F3 = Conv1x1(DWConv3x3_BN_SiLU(Ffuse))      # C/2 -> C
Fout = Fin + F3                            # 公式18，没有额外输出激活
```

通道支路没有擅自换成两层SE-MLP。默认depthwise步骤综合正文描述与1×1投影；可切换为仅公式/图示的1×1投影。

### PAFFN




## 使用

推荐Python 3.8+和PyTorch 1.12+。实际验证环境为Python 3.8.5、PyTorch 2.4.1+cpu。无需einops。请从本文件夹导入，避免上级目录已有旧同名文件干扰。

```python
import torch
from c3k2_sf import C3k2_SF
from sica import SICA
from paffn import PAFFN

# 单独的模块示例，不组成完整检测器
sf = C3k2_SF(64, 128, e=0.25)
y = sf(torch.randn(1, 64, 160, 160))
attn = SICA(512)
z = attn(torch.randn(1, 512, 20, 20))

# 输入通道以你的backbone实际输出为准
neck = PAFFN((256, 256, 512), out_channels=(128, 256, 512))
xs = [torch.randn(1, 256, 80, 80),
      torch.randn(1, 256, 40, 40),
      torch.randn(1, 512, 20, 20)]
outputs = neck(xs)
print([tuple(t.shape) for t in outputs])
# [(1,128,80,80),(1,256,40,40),(1,512,20,20)]
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
