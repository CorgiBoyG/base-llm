# Base LLM 学习笔记

> 本仓库是我学习 [Datawhale Base LLM 教程](https://github.com/datawhalechina/base-llm) 的个人笔记与实践记录

## 📚 关于原教程

**Base LLM** 是 Datawhale 开源的一套从传统自然语言处理（NLP）到大语言模型（LLM）的全栈式学习教程。

- **在线阅读地址**: https://datawhalechina.github.io/base-llm/
- **GitHub 仓库**: https://github.com/datawhalechina/base-llm

### 教程核心特点

- ✅ 体系完整：从 NLP 基础到 LLM 前沿技术
- ✅ 理论与实践结合：不仅讲原理，更重视代码实现
- ✅ 工程化导向：覆盖模型训练、微调、量化到部署全流程
- ✅ 手写核心代码：深入理解 Transformer、Llama2 等架构

## 📖 学习大纲

### 第一部分：理论篇

- 第 1 章：NLP 简介
- 第 2 章：文本表示与词向量（Word2Vec、Gensim 实战）
- 第 3 章：循环神经网络（RNN、LSTM、GRU）
- 第 4 章：注意力机制与 Transformer
- 第 5 章：预训练模型（BERT、GPT、T5、Hugging Face）
- 第 6 章：深入大模型架构（手搓大模型、MOE、生成策略）

### 第二部分：实战篇

- 第 1 章：文本分类（LSTM、BERT 微调）
- 第 2 章：命名实体识别（NER 全流程）

### 第三部分：微调量化篇

- 第 1 章：参数高效微调（PEFT、LoRA、Qwen2.5 微调）
- 第 2 章：高级微调技术（RLHF、DPO）
- 第 3 章：大模型训练与量化（模型量化、Deepspeed）

### 第四部分：应用部署篇

- 第 1 章：模型服务部署（FastAPI、Docker Compose）
- 第 2 章：自动化与性能优化（Git/GitHub、Jenkins CI/CD）

### 第五部分：大模型安全

- 第 1 章：安全全景与威胁建模
- 第 2 章：安全工程（行为对齐、架构设计）

### 第六部分：多模态前沿

## 📝 笔记结构

```
├── notes/           # 学习笔记
│   ├── 01-nlp-basics/
│   ├── 02-word-vectors/
│   ├── 03-rnn/
│   ├── 04-transformer/
│   ├── 05-pretrained-models/
│   └── ...
├── code/            # 代码实践
│   ├── text-classification/
│   ├── ner/
│   ├── fine-tuning/
│   └── ...
└── resources/       # 学习资源
```

## 🎯 学习目标

-  掌握 NLP 核心理论（词向量、RNN、Transformer）
-  理解预训练模型的原理（BERT、GPT）
-  实现文本分类和命名实体识别项目
-  掌握大模型微调技术（LoRA、RLHF）
-  学会模型量化与部署
-  了解大模型安全与多模态技术

## 🛠️ 环境配置

```bash
# Python 版本
Python 3.8+

# 主要依赖
PyTorch
Transformers (Hugging Face)
PEFT
DeepSpeed
FastAPI
Docker
```

## 📌 学习进度

- [ ]  第一部分：理论篇
- [ ]  第二部分：实战篇
- [ ]  第三部分：微调量化篇
- [ ]  第四部分：应用部署篇
- [ ]  第五部分：大模型安全
- [ ]  第六部分：多模态前沿

## 🤝 致谢

感谢 Datawhale 团队开源这套优质教程，为 AI 学习者提供了宝贵的学习资源。

## 📄 许可证

本学习笔记仅供个人学习使用。原教程采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可协议。

------

**开始时间**: [2026.1.14]
**最后更新**: [2026.1.18]