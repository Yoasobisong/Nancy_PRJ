# Nancy 机器人 — 强化学习路线图

> 目标：4 周内掌握所有必要知识，能读懂 Columbia Emo 论文并独立实现  
> 前提：已有嵌入式基础、Python 基础、吴恩达课程前半部分  
> 强度：每天 4-6 小时学习

---

## 知识全景

```
Nancy 项目 = 4 大模块

① 深度学习基础        看懂论文、理解模型、会训练
② FACS + 面部感知     项目的"语言"，贯穿软硬件
③ 机械结构设计        连杆、运动学、CAD、硅胶皮肤
④ 系统整合            Jetson + 舵机 + ROS + 实时控制
```

---

## 第 1 周：深度学习 + PyTorch（地基）course4

### Day 1-2：吴恩达补完

| 时间 | 内容                                       | 资源                                                                                              |
| ---- | ------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| 上午 | CNN 卷积神经网络：卷积层、池化层、特征图   | [YouTube Playlist: CNN](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF) |
| 下午 | 经典 CNN 架构：LeNet → VGG → ResNet        | 同上                                                                                              |
| 晚上 | 动手：用 PyTorch 跑一个 MNIST/CIFAR10 分类 | 代码跟着敲                                                                                        |

**免费配套资源：**

- [ ] 课程笔记 (中文): [GitHub - fengdu78](https://github.com/fengdu78/deeplearning_ai_books)
- [ ] 课后作业 (代码): [GitHub - amanchadha](https://github.com/amanchadha/coursera-deep-learning-specialization)

**关键概念清单：**

- [ ] 卷积核 (kernel/filter) 是什么
- [ ] 特征图 (feature map) 怎么产生
- [ ] 池化 (pooling) 的作用
- [ ] ResNet 残差连接为什么有效

### Day 3-4：序列模型

| 时间 | 内容                              | 资源                                                                                                          |
| ---- | --------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 上午 | RNN → LSTM → GRU 原理             | [YouTube Playlist: Sequence Models](https://www.youtube.com/playlist?list=PLkDaE6sCZn6F6wUI9tvS_Gw1vaFAx6rd6) |
| 下午 | Encoder-Decoder 架构 (论文核心!)  | 同上 (选看相关视频)                                                                                           |
| 晚上 | Attention 机制 + Transformer 概念 | 3Blue1Brown Transformer 可视化视频                                                                            |

**关键概念清单：**

- [ ] RNN 为什么能处理时间序列
- [ ] LSTM 的遗忘门/输入门/输出门
- [ ] Encoder-Decoder：输入序列 → 压缩表示 → 输出序列
- [ ] Self-Attention 的直觉理解

### Day 5-6：PyTorch 实战

| 时间 | 内容                                         | 资源                                                                              |
| ---- | -------------------------------------------- | --------------------------------------------------------------------------------- |
| 上午 | PyTorch 基础：Tensor、autograd、nn.Module    | [小土堆 PyTorch 教程 (B站)](https://www.bilibili.com/video/BV1hE411t7RN) 前 15 集 |
| 下午 | Dataset/DataLoader、训练循环、保存加载模型   | 同上 15-25 集                                                                     |
| 晚上 | 动手：训练一个简单的回归模型 (输入→输出映射) | 自己写                                                                            |

**必须会写的代码骨架：**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. Custom dataset
class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 2. Model definition
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# 3. Training loop
model = Model(136, 24)  # 68 landmarks * 2 → 24 servos
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(100):
    for batch_x, batch_y in dataloader:
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 4. Inference
model.eval()
with torch.no_grad():
    result = model(test_input)
```

### Day 7：复习 + 测验

- 回顾 CNN、LSTM、Encoder-Decoder、PyTorch
- 自测：能否不看资料写出上面的训练代码骨架？
- 能否画出 Encoder-Decoder 的数据流图？

---

## 第 2 周：FACS + 面部感知（核心）

### Day 8-9：FACS 面部动作编码系统

| 时间 | 内容                                 | 资源                                                                        |
| ---- | ------------------------------------ | --------------------------------------------------------------------------- |
| 上午 | FACS 基础：什么是 AU，AU 强度 (0-5)  | YouTube: "FACS Action Units Tutorial"                                       |
| 下午 | 核心 AU 学习 (下方表格)              | [FACS Wikipedia](https://en.wikipedia.org/wiki/Facial_Action_Coding_System) |
| 晚上 | 对着镜子练习：做出每个 AU 对应的表情 | 实践！                                                                      |

**Nancy 必须掌握的 AU（按面部区域）：**

| AU   | 名称               | 面部位置      | Nancy 对应机构 | 肌肉         |
| ---- | ------------------ | ------------- | -------------- | ------------ |
| AU1  | Inner Brow Raise   | 内眉上提      | 眉毛内侧舵机   | 额肌内侧     |
| AU2  | Outer Brow Raise   | 外眉上提      | 眉毛外侧舵机   | 额肌外侧     |
| AU4  | Brow Lowerer       | 皱眉          | 眉毛下拉舵机   | 皱眉肌       |
| AU5  | Upper Lid Raise    | 睁大眼        | 上眼睑舵机     | 提上睑肌     |
| AU6  | Cheek Raise        | 颧骨提升      | 脸颊舵机       | 颧大肌       |
| AU7  | Lid Tighten        | 眯眼          | 下眼睑舵机     | 眼轮匝肌     |
| AU9  | Nose Wrinkle       | 皱鼻          | (可选)         | 提上唇鼻翼肌 |
| AU10 | Upper Lip Raise    | 上唇上提      | 上唇舵机       | 提上唇肌     |
| AU12 | Lip Corner Pull    | 嘴角上扬=微笑 | 嘴角舵机       | 颧大肌       |
| AU15 | Lip Corner Depress | 嘴角下拉=悲伤 | 嘴角舵机(反向) | 降口角肌     |
| AU20 | Lip Stretch        | 唇展开        | 唇部舵机       | 颈阔肌       |
| AU25 | Lips Part          | 嘴唇分开      | 唇部舵机       | 降下唇肌     |
| AU26 | Jaw Drop           | 张嘴          | 下颌舵机       | 翼外肌       |
| AU45 | Blink              | 眨眼          | 眼睑舵机       | 眼轮匝肌     |

**表情 = AU 组合：**

| 表情    | AU 组合                      | 描述                       |
| ------- | ---------------------------- | -------------------------- |
| 😊 微笑 | AU6 + AU12                   | 颧骨提升 + 嘴角上扬        |
| 😢 悲伤 | AU1 + AU4 + AU15             | 内眉上提 + 皱眉 + 嘴角下拉 |
| 😮 惊讶 | AU1 + AU2 + AU5 + AU26       | 眉毛全提 + 睁大眼 + 张嘴   |
| 😠 愤怒 | AU4 + AU5 + AU7 + AU23       | 皱眉 + 睁大 + 眯眼 + 抿嘴  |
| 😨 恐惧 | AU1 + AU2 + AU4 + AU5 + AU20 | 眉上提+皱眉+睁大+唇展      |
| 🤢 厌恶 | AU9 + AU15 + AU25            | 皱鼻 + 嘴角下拉 + 嘴唇分开 |

### Day 10-11：面部检测实战

| 时间 | 内容                                      | 资源                                                                                                                                         |
| ---- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 上午 | MediaPipe Face Mesh 安装 + 跑起来         | `pip install mediapipe`                                                                                                                      |
| 下午 | 理解 468 个关键点的分布                   | [关键点图谱](https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png) |
| 晚上 | 写代码：提取关键点 → 计算简单的 AU 近似值 | 自己写                                                                                                                                       |

**关键点 → AU 近似计算示例：**

```python
import mediapipe as mp
import numpy as np

def landmarks_to_au_approx(landmarks):
    """
    Approximate AU values from MediaPipe 468 landmarks.
    Each landmark is (x, y, z) normalized to [0,1].
    """
    lm = np.array([(l.x, l.y, l.z) for l in landmarks])

    # AU26 - Jaw Drop: distance between upper and lower lip
    upper_lip = lm[13]  # upper lip center
    lower_lip = lm[14]  # lower lip center
    jaw_open = np.linalg.norm(upper_lip - lower_lip)
    au26 = min(jaw_open * 30, 5.0)  # scale to 0-5

    # AU12 - Smile: mouth width relative to rest
    mouth_left = lm[61]
    mouth_right = lm[291]
    mouth_width = np.linalg.norm(mouth_left - mouth_right)
    au12 = min(mouth_width * 15, 5.0)

    # AU45 - Blink: eye aspect ratio
    # left eye: top=159, bottom=145, left=33, right=133
    eye_top = lm[159]
    eye_bot = lm[145]
    eye_height = np.linalg.norm(eye_top - eye_bot)
    au45 = max(0, (0.02 - eye_height) * 200)  # closed = high value

    return {'AU12': au12, 'AU26': au26, 'AU45': au45}
```

### Day 12：OpenFace 安装 + 对比

| 时间 | 内容                                             |
| ---- | ------------------------------------------------ |
| 上午 | 安装 OpenFace 2.0 (Windows 有编译好的版本)       |
| 下午 | 对比 MediaPipe 自己算的 AU 和 OpenFace 输出的 AU |
| 晚上 | 决定 V1 用哪个方案 (推荐: MediaPipe 更轻量)      |

> **OpenFace 下载：** https://github.com/TadasBaltrusaitis/OpenFace/releases

### Day 13-14：精读论文

| 时间        | 内容                                               |
| ----------- | -------------------------------------------------- |
| Day 13 上午 | 通读论文全文，标注不懂的地方                       |
| Day 13 下午 | 重点读 Section 3 (Method)：网络结构、输入输出      |
| Day 14 上午 | 重点读 Section 4 (Hardware)：舵机布局、AU→舵机映射 |
| Day 14 下午 | **画出完整的系统框图**（从摄像头到舵机每一步）     |

> **论文地址：** https://www.creativemachineslab.com/emo.html

**读论文时重点关注：**

- [ ] 输入是什么？(对方面部 AU 的时间序列)
- [ ] 输出是什么？(机器人 AU 的目标值)
- [ ] 网络结构是什么？(Encoder-Decoder + Attention?)
- [ ] 训练数据从哪来？(人类对话视频)
- [ ] 损失函数是什么？(AU 预测误差)

---

## 第 3 周：机械 + 电子（你的主场）

### Day 15-16：机构学速成

| 时间 | 内容                          | 资源                    |
| ---- | ----------------------------- | ----------------------- |
| 上午 | 四连杆机构原理 + 类型         | B 站搜"机械原理 四连杆" |
| 下午 | 曲柄滑块、万向节、差动机构    | 同上                    |
| 晚上 | 用纸板 + 铁丝做一个四连杆模型 | 动手！                  |

**重点理解：**

- [ ] 曲柄摇杆 → 连续旋转变往复摆动 (唇部机构)
- [ ] 平行四连杆 → 保持角度 (眼球机构)
- [ ] 差动机构 → 两输入合成两输出 (脖子)
- [ ] 万向节 → 两轴旋转自由度 (脖子顶部)

### Day 17-18：CAD 建模（Fusion360）

| 时间 | 内容                                       | 资源                                                                        |
| ---- | ------------------------------------------ | --------------------------------------------------------------------------- |
| 上午 | Fusion360 安装 + 基础操作 (草图/拉伸/旋转) | [Fusion360 官方教程](https://help.autodesk.com/view/fusion360/ENU/courses/) |
| 下午 | 画一个舵机支架 + 连杆                      | 跟教程做                                                                    |
| 晚上 | 装配体：把舵机+连杆+摇臂装起来             | 实操                                                                        |

### Day 19-20：电子系统搭建

| 时间 | 内容                                |
| ---- | ----------------------------------- |
| 上午 | PCA9685 + Jetson I2C 接线 + 测试    |
| 下午 | 控制 2-3 个舵机，写缓入缓出插值代码 |
| 晚上 | 4S 电池 + BEC 分电方案焊接          |

**测试代码：**

```python
from adafruit_servokit import ServoKit
import math, time

kit = ServoKit(channels=16, address=0x40)

def ease_move(ch, target, duration=0.5):
    """Smooth servo movement with ease-in-out."""
    current = kit.servo[ch].angle or 90
    steps = int(duration * 50)  # 50Hz update
    for i in range(1, steps + 1):
        t = i / steps
        ease = 0.5 - 0.5 * math.cos(math.pi * t)
        kit.servo[ch].angle = current + (target - current) * ease
        time.sleep(duration / steps)

# Test
ease_move(0, 120, 0.5)
time.sleep(0.5)
ease_move(0, 60, 0.5)
```

### Day 21：V1 完整管线

把前面所有东西串起来：

```
摄像头 → MediaPipe → 关键点 → AU 近似 → 查表 → PCA9685 → 舵机

一个 Python 文件搞定！
```

---

## 第 4 周：整合 + 论文复现（冲刺）

### Day 22-23：V1 整合调试

| 时间 | 内容                                    |
| ---- | --------------------------------------- |
| 上午 | 整合完整 pipeline：摄像头到舵机         |
| 下午 | 调整 AU → 舵机映射表（手动标定每个 AU） |
| 晚上 | 测试各种表情：微笑、惊讶、悲伤、愤怒    |

### Day 24-25：论文模型复现

| 时间        | 内容                                      |
| ----------- | ----------------------------------------- |
| Day 24 上午 | 用 PyTorch 搭建论文的 Encoder-Decoder     |
| Day 24 下午 | 准备训练数据：录制对话视频 → 提取 AU 序列 |
| Day 25 上午 | 训练模型 (先用小数据集验证 pipeline)      |
| Day 25 下午 | 模型推理测试：输入 AU 序列 → 预测下一帧   |

### Day 26-27：部署 + 优化

| 时间 | 内容                                     |
| ---- | ---------------------------------------- |
| 上午 | PyTorch → ONNX → TensorRT (你熟悉的流程) |
| 下午 | 在 Jetson 上跑完整管线，测帧率           |
| 晚上 | 延迟优化、舵机响应调整                   |

### Day 28：总结 + V2 规划

- V1 演示录像
- 记录所有问题和改进方向
- 规划 V2：唇同步、更好的训练数据、更精细的 AU 标定

---

## 资源汇总

### 视频课程

| 资源            | 内容               | 时长 | 链接                                                                                         |
| --------------- | ------------------ | ---- | -------------------------------------------------------------------------------------------- |
| 吴恩达 CNN      | 卷积神经网络       | ~10h | [YouTube Playlist](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF) |
| 吴恩达 序列模型 | RNN/LSTM/Attention | ~10h | [YouTube Playlist](https://www.youtube.com/playlist?list=PLkDaE6sCZn6F6wUI9tvS_Gw1vaFAx6rd6) |
| 小土堆 PyTorch  | PyTorch 入门       | ~5h  | [B 站](https://www.bilibili.com/video/BV1hE411t7RN)                                          |
| 3Blue1Brown     | 神经网络可视化     | ~1h  | [YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)          |
| FACS 教程       | 面部动作编码       | ~2h  | YouTube 搜 "FACS tutorial"                                                                   |

### 工具 & 库

| 工具              | 用途             | 安装                                                                      |
| ----------------- | ---------------- | ------------------------------------------------------------------------- |
| PyTorch           | 深度学习框架     | `pip install torch torchvision`                                           |
| MediaPipe         | 面部关键点检测   | `pip install mediapipe`                                                   |
| OpenFace          | AU 检测 (备选)   | [GitHub Releases](https://github.com/TadasBaltrusaitis/OpenFace/releases) |
| OpenCV            | 图像处理         | `pip install opencv-python`                                               |
| adafruit-servokit | PCA9685 舵机控制 | `pip install adafruit-circuitpython-servokit`                             |
| Fusion360         | CAD 建模         | [免费下载](https://www.autodesk.com/products/fusion-360/personal)         |

### 论文 & 参考

| 资源                                                                                 | 说明            |
| ------------------------------------------------------------------------------------ | --------------- |
| [Columbia Emo Robot](https://www.creativemachineslab.com/emo.html)                   | 核心论文 + 视频 |
| [FACS Manual (Wikipedia)](https://en.wikipedia.org/wiki/Facial_Action_Coding_System) | AU 编码参考     |
| [OpenFace 2.0 论文](https://ieeexplore.ieee.org/document/8373812)                    | AU 检测算法     |
| [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)   | 468 关键点文档  |

---

## 每日检查清单

```
□ 今天学了什么新概念？能用自己的话解释吗？
□ 今天写了代码吗？能跑起来吗？
□ 这个知识和 Nancy 项目的哪个环节有关？
□ 还有什么不懂的？记录下来明天解决。
```

---

## 里程碑

| 时间点   | 里程碑                        | 验证方法                 |
| -------- | ----------------------------- | ------------------------ |
| 第 7 天  | ✅ 能写 PyTorch 训练循环      | 不看资料写出完整训练代码 |
| 第 11 天 | ✅ 对着摄像头实时输出 AU 值   | MediaPipe 代码跑起来     |
| 第 14 天 | ✅ 能画出论文完整系统框图     | 手绘/Excalidraw          |
| 第 18 天 | ✅ Fusion360 画出一个舵机支架 | 导出 STL 打印            |
| 第 21 天 | ✅ 摄像头→AU→舵机 跑通        | 对着摄像头，舵机动起来   |
| 第 28 天 | ✅ 完整 V1 演示               | 录一个演示视频           |
