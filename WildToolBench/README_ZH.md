# Benchmarking LLM Tool-Use in the Wild


<p align="center">
    📖 <a href="README.md">English</a> •
    <a>中文</a> •
    📚 <a href="https://openreview.net/forum?id=yz7fL5vfpn">ICLR 2026 Paper</a> •
    🌐 <a href="https://yupeijei1997.github.io/WildToolBench/">Project Page</a>
</p>


![Example](./picture/benchmark_comparison.png)


## 📖 Overview

通过大型语言模型（LLM）的多轮、多步骤工具调用满足用户需求，往往并非简单直接的过程。真实的用户交互本质上具有“随机性”——不可预测、复杂混乱且灵活多变。我们从用户行为中识别出三大核心挑战：
1. **组合式任务**：这类任务要求对工具调用的拓扑结构进行协调编排；
2. **隐含意图**：用户意图分散在多轮对话中，需要模型进行上下文推理才能捕捉；
3. **指令切换**：用户输入中可能混合任务查询、澄清提问与日常对话，迫使LLM实时调整自身策略。

现有基准测试忽略了这些真实行为，导致在工具调用场景下观察到的 LLM 性能提升存在“虚假性”。为解决这一问题，我们提出了**WildToolBench**——一个基于真实世界用户行为模式构建的 LLM 工具调用基准测试集。

对57个LLM的全面评估显示，没有任何一个模型的准确率超过15%，这表明 LLM Agent 能力的鲁棒性仍存在巨大差距。对照实验与深度分析进一步表明，LLM 工具调用的真正挑战并非来自人为设计的复杂任务，而是源于用户行为的“随机性”本质。这一发现强调，我们有必要重新审视LLM、用户与工具三者之间的交互关系。

## 😊 Key Materials

- 测试数据地址：wild-tool-bench/data/WildToolBench.jsonl
- 更多关于 WildToolBench 的详细信息可以在下文中获取

## ⚡️ Quickstart

### Basic Installation
```bash
# Create a new Conda environment with Python 3.10
conda create -n WildToolBench python=3.10
conda activate WildToolBench

# Install the package
pip install -r requirements.txt
```

## ⏳ Inference

### 🤖 API Models
本项目支持 OpenAI 格式调用的 API 模型

以 deepseek-chat 为例，参考.env.example，创建一个.env文件，并设置以下key

```bash
DEEPSEEK_API_KEY=sk-XXXXXX
```

之后，使用以下代码请求模型结果
```bash
cd wild-tool-bench/

python3 -u -m wtb.openfunctions_evaluation --model=deepseek-chat
```

## 💫 Evaluation
使用以下代码对模型预测结果进行评估

```bash
cd wild-tool-bench

python3 -u -m wtb.eval_runner --model=deepseek-chat
```

## 🧠 Controllable Multi Agent Data Generation Framework

### ⚡️ Quickstart

以所有智能体均使用 deepseek-chat 作为基座模型，并且生成中文的数据为例。首先，参考.env.example，创建一个.env文件，并设置以下key

```bash
DEEPSEEK_API_KEY=sk-XXXXXX
LANGUAGE=zh
```

本论文的核心创新点在于，我们提出的 WildToolBench 能够覆盖任意数量任务的所有可能的动作空间，并且除第一轮以外全部为真多轮任务，因此我们的框架能够支持生成任意数量任务的数据，此处我们以生成四个任务为例，参考如下代码：
```bash
cd multi-agent-framework

python3 generate.py \
    --layer-num-total 4 \
    --user-model "deepseek-chat" \
    --planner-model "deepseek-chat" \
    --tool-model "deepseek-chat" \
    --agent-model "deepseek-chat" \
    --checker-model "deepseek-chat"
```
特别的，我们保存了三种格式的数据，包括：原始生成数据、精简后的数据(删除checker message和角色切换 message)、各个智能体的抓取数据，以供需要的研究人员使用。

最后，展示一下我们生成的一个实际case的部分结果，可以发现，Continue Question User Agent 生成了一个指代理解类型的真多轮任务 "第三张壁纸的上传者是谁？"，这意味着在解决该任务时，需要模型参考上文内容，才能获取第三张壁纸的ID 778899，从而继续调用获取壁纸详细信息的工具getWaifuDetails。

```
用户: 我需要最新的5张动漫壁纸。

Planner：getWaifuImages(limit=5, tags=["动漫壁纸"])

Tool：...

Planner：prepare_to_answer(answer_type=tool)

Agent助手:

以下是最新的5张动漫壁纸:
1. 壁纸ID: 112233
 - [查看壁纸](https://wallhaven.cc/w/112233)
 - ![缩略图](https://wallhaven.cc/thumb/small/112233.jpg)
 - 分辨率: 1920x1080
2. 壁纸ID: 445566
 - [查看壁纸](https://wallhaven.cc/w/445566)
 - ![缩略图](https://wallhaven.cc/thumb/small/445566.jpg)
 - 分辨率: 2560x1440
3. 壁纸ID: 778899
 - [查看壁纸](https://wallhaven.cc/w/778899)
 - ![缩略图](https://wallhaven.cc/thumb/small/778899.jpg)
 - 分辨率: 3840x2160
4. 壁纸ID: 101112
 - [查看壁纸](https://wallhaven.cc/w/101112)
 - ![缩略图](https://wallhaven.cc/thumb/small/101112.jpg)
 - 分辨率: 1920x1080
5. 壁纸ID: 131415
 - [查看壁纸](https://wallhaven.cc/w/131415)
 - ![缩略图](https://wallhaven.cc/thumb/small/131415.jpg)
 - 分辨率: 2560x1440

你可以点击链接查看完整壁纸。希望你喜欢这些壁纸！如果需要更多帮助，请告诉我。

Continue Question User Agent Generate:
用户: 第三张壁纸的上传者是谁? 

Planner：getWaifuDetails(image_id=778899)
```

## 📝 引用

如果您觉得 WildToolBench 对您的研究有帮助，请考虑引用我们的论文：

```bibtex
@inproceedings{
  yu2026wildtoolbench,
  title={Benchmarking {LLM} Tool-Use in the Wild},
  author={Peijie Yu and Wei Liu and Yifan Yang and Jinjian Li and Zelong Zhang and Xiao Feng and Feng Zhang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=yz7fL5vfpn}
}
```
