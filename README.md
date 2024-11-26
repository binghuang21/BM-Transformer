# 面向丰富曲调要素的影视配乐生成模型

本项目旨在通过模型生成适合影视作品的配乐，以丰富视听体验。模型效果可以在以下Hugging face网站上的结果对比页面中查看，并了解如何为视频创作背景音乐。

- **模型效果展示**：[点击查看](https://huggingface.co/spaces/bingshuang21/BM-Transformer)

如遇到huggingface无法正常显示，可能是网络原因，请稍等一段时间重新访问或更换浏览器访问

- **模型下载链接**：[Google Drive](https://drive.google.com/file/d/1HdGrn1PsusQZxS69zOX6w-8_WHFtSXQ6/view?usp=drive_link)

## 数据集准备

### 使用预制数据集
可从以下链接下载准备好的数据集：
[Google Drive数据集下载](https://drive.google.com/drive/folders/1DkPQa4rACMLiJIJHW5XFLRjvstZpjPzp?usp=drive_link)

### 从头准备数据集
如果需要从头开始准备数据集，请参阅 *dataset* 文件夹下的 `README.md` 文件。

## 模型训练

使用预处理后的数据集进行训练，您需要在 `main_cp.py` 中更改以下超参数：

- `mode`：使用 `train` 进行训练阶段，使用 `inference` 进行推断阶段。
- `gid`：设置所用GPU的序号。
- `data_parallel`：训练阶段设置为 `1`，推断阶段设置为 `0`。
- `data_root`：设置训练数据目录。
- `videos`：推断阶段待配乐视频目录。
- `num_songs`：为每个视频创作背景音乐的数量。
- `out_dir`：创作背景音乐的输出目录。

设置好各参数后，执行以下命令进行训练：
```bash
python main_cp.py
```

## 模型推断

在推断阶段，首先需要提取视频的节奏信息，并转换成模型可以识别的数据格式。按照以下步骤操作：

1. 提取视频节奏信息：

```bash
python video2metadata.py --video 输入视频地址 --meta_data 输出节奏信息文件地址 --is_tempo 1 --my_tempo 若使用用户指定节奏，在此设定 --is_path 1
```

2. 将节奏信息转化为模型可用的数据格式：

```bash
python metadata2numpy_mix.py --video 输入视频地址 --meta_data 输入节奏信息文件地址 --is_path 1 --out_dir 视频节奏输出目录
```

3. 完成上述步骤后，调整 main_cp.py 中的推断相关超参数。接着执行下面的命令进行推断：：

```bash
python main_cp.py
```
## 双视角评估平台

访问以下存储库了解更多关于双视角评估平台的信息：[双视角评估平台GitHub存储库](https://github.com/binghuang21/bgm-evaluation-platform.git)

## 用户友好型背景音乐配乐平台

此平台基于BMT开发，简化了为视频创作背景音乐的流程。更多信息和操作指南，请参阅下方链接：[用户友好型背景音乐配乐平台GitHub存储库](https://github.com/binghuang21/soundtrack-platform.git)

## 使用授权

此项目仅供学习使用，已申请软件著作权，商用需联系作者授权。

<!-- # **面向丰富曲调要素的影视配乐生成模型**

为展示模型效果，本文在 Hugging face 上构建了结果对比网页，可见：https://huggingface.co/spaces/bingshuang21/BM-Transformer
用户可以查看上方链接观看模型效果，以及了解如何操作平台为视频创作需要的背景音乐。

模型下载：https://drive.google.com/file/d/1HdGrn1PsusQZxS69zOX6w-8_WHFtSXQ6/view?usp=drive_link

### 数据集准备
**使用准备好的数据集：**

从https://drive.google.com/drive/folders/1DkPQa4rACMLiJIJHW5XFLRjvstZpjPzp?usp=drive_link处下载。

**从头开始准备数据集：**详见*dataset*文件夹下README.md。

### 模型训练
使用预处理后的数据，训练只需更改 main_cp.py 中的超参数，参数说明如下：
• mode：训练阶段使用“train”，推断阶段使用“inference”。
• gid：设置所用 GPU 的序号。
• data_parallel：训练阶段为 1，推断阶段为 0。
• data_root：训练数据目录。
• videos：推断阶段待配乐视频目录。
• num_songs：推断阶段为每个视频创作背景音乐的数量。
• out_dir：推断阶段创作背景音乐的输出目录。
清楚各参数含义并设定后，运行 $python main_cp.py$ 即可进行训练。

### 模型推断
推理阶段首先要提取视频的节奏信息，转换为原始推理数据，请遵循以下步骤：
(1) $python video2metadata.py$ 提取视频节奏信息。
• video：输入视频地址。
• meta_data：输出节奏信息文件的地址。
• is_tempo：1 为用户指定节奏，0 为模型自动提取节奏。
• my_tempo：若选择用户指定节奏，则在此处设定节奏。
• is_path：1 代表输入视频目录，0 代表输入单个视频文件。
(2) $python metadata2numpy_mix.py$ 将视频节奏信息转化为原始推断数据。
• video：输入视频地址。
• meta_data：输入节奏信息文件的地址。
• is_path：1 代表输入视频目录，0 代表输入单个视频文件。
• out_dir：视频节奏输出目录。
提取得到视频节奏信息后，调整 main_cp.py 中相应推断超参数，运行 $python main_cp.py$ 即可进行推断。可从谷歌云盘下载本文训练好的模型进行推断。

### 双视角评估平台
详见存储库：https://github.com/binghuang21/bgm-evaluation-platform.git

### 用户友好型背景音乐配乐平台
此平台是基于BMT开发的用户友好型背景音乐配乐平台，用户可以查看上方链接观看模型效果，以及了解如何操作平台为视频创作需要的背景音乐。详见存储库：https://github.com/binghuang21/soundtrack-platform.git -->