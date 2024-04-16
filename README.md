# **面向丰富曲调要素的影视配乐生成模型**

为展示模型效果，本文在 Hugging face 上构建了结果对比网页，可见：https://huggingface.co/spaces/bingshuang21/BM-Transformer
用户可以查看上方链接观看模型效果，以及了解如何操作平台为视频创作需要的背景音乐。

模型下载：https://drive.google.com/file/d/1HdGrn1PsusQZxS69zOX6w-8_WHFtSXQ6/view?usp=drive_link

### 数据集准备
**使用准备好的数据集：**从https://drive.google.com/drive/folders/1DkPQa4rACMLiJIJHW5XFLRjvstZpjPzp?usp=drive_link处下载。
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
此平台是基于BMT开发的用户友好型背景音乐配乐平台，用户可以查看上方链接观看模型效果，以及了解如何操作平台为视频创作需要的背景音乐。详见存储库：https://github.com/binghuang21/soundtrack-platform.git