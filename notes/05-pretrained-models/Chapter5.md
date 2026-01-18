# 第五章：预训练模型

## 前言

这段时间系统学习了Transformer家族的三大分支：BERT、GPT和T5。说实话，之前只是知道这些模型很火，但一直没搞清楚它们之间到底有什么区别。这次终于把这些概念理顺了，顺便也把Hugging Face这套工具链摸透了。

简单来说，这三个模型其实代表了三种不同的思路：

- BERT选择了Encoder，专注理解文本
- GPT选择了Decoder，专注生成文本
- T5把完整的Encoder-Decoder都用上了，想做个大一统

## BERT：双向理解

### 为什么需要BERT

在BERT之前，Word2Vec这类静态词向量有个致命问题：一词多义没法处理。比如"破防"这个词，在"我出了一件破防装备"和"这题目把我学破防了"里意思完全不同，但Word2Vec给的向量却是一样的。

BERT的核心创新就是做**动态词向量**——同一个词在不同上下文里会有不同的表示。这是怎么做到的？关键在于Transformer的自注意力机制，它能让每个词"看到"整个句子的其他所有词。

### MLM：完形填空式的预训练

BERT的预训练任务叫Masked Language Model（MLM），本质上就是做完形填空。随机盖住15%的词，让模型根据上下文猜被盖住的是什么。

这里有个很巧妙的设计。这15%的词并不是全部替换成[MASK]，而是：

- 80%替换成[MASK]：`My [MASK] is a good student`
- 10%替换成随机词：`My apple is a good student`
- 10%保持不变：`My son is a good student`

刚开始我也没理解为什么要这么复杂。后来想明白了：如果训练时全是[MASK]，实际使用时却没有[MASK]，这就存在train-test mismatch的问题。加入随机词和原词，能让模型更robust，不会过度依赖[MASK]这个特殊标记。

### 输入表示的三个组成部分

BERT的输入表示由三部分相加得到：

Input=Token Embedding+Position Embedding+Segment Embedding\text{Input} = \text{Token Embedding} + \text{Position Embedding} + \text{Segment Embedding}Input=Token Embedding+Position Embedding+Segment Embedding

- **Token Embedding**：就是词本身的向量表示，使用WordPiece分词
- **Position Embedding**：位置信息，这里BERT用的是可学习的，而不是Transformer原文的正余弦函数
- **Segment Embedding**：用来区分句子A和句子B，主要服务于NSP任务

这里的Position Embedding是个`[512, 768]`的矩阵，这也就解释了为什么BERT的最大长度是512——本质上是这个矩阵的第一维大小限制的。

### 关于NSP的争议

BERT还有个预训练任务叫Next Sentence Prediction（NSP），就是判断两个句子是不是连续的。原论文说这个任务很重要，但后来RoBERTa的实验发现去掉NSP效果反而更好。

我的理解是：NSP在BERT那个规模（110M参数）和数据量下确实有帮助，但放到更大规模时就不一定了。这也提醒我们，很多结论都是有适用条件的。

## GPT：自回归生成的力量

### 单向注意力的设计

GPT和BERT最大的区别在于注意力机制的方向性。GPT使用的是**Masked Self-Attention**（也叫Causal Attention），保证预测第ttt个词时只能看到前t−1t-1t−1个词：

P(x1,...,xT)=∏t=1TP(xt∣x1,...,xt−1)P(x_1, ..., x_T) = \prod_{t=1}^{T} P(x_t | x_1, ..., x_{t-1})P(x1,...,xT)=∏t=1TP(xt∣x1,...,xt−1)

这个设计看起来限制很大，但正是这种"只能看过去"的约束，让GPT天然适合文本生成任务。

### 从微调到提示的范式转变

GPT系列的演进很有意思：

**GPT-1时期**还在做有监督微调，为不同任务设计不同的输入格式。比如做文本蕴含时，把前提和假设用分隔符拼起来：`前提 $ 假设`。

**GPT-2时期**发现模型规模扩大后（1.5B），零样本能力开始显现。直接给模型指令就能工作，比如：

```
Translate English to French: cheese =>
```

模型会自动续写出`fromage`。

**GPT-3时期**（175B参数）In-context Learning的能力爆发了。只需要在prompt里给几个示例，模型就能理解任务：

```
中文翻译成英文：
苹果 -> apple
香蕉 -> banana  
你好 ->
```

这种范式转变意味着我们不再需要为每个任务训练一个模型，而是通过设计prompt来"编程"一个通用模型。

### 自回归生成的实现细节

手写一个简单的生成循环能帮助理解自回归的本质：

```python
generated_ids = input_ids
for i in range(max_new_tokens):
    outputs = model(generated_ids)
    next_token_logits = outputs.logits[:, -1, :]  # 只取最后一个位置
    next_token_id = torch.argmax(next_token_logits, dim=-1)
    generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(-1)], dim=1)
```

关键点在于每次只预测下一个token（`outputs.logits[:, -1, :]`），然后把新预测的token拼回输入，再预测下一个。这就是"自回归"的含义——用自己的输出作为下一步的输入。

### BPE分词的优劣

GPT使用Byte-level BPE分词，理论上能处理任何字符（因为最坏情况下可以拆成字节）。但这也带来一个问题：对于非英语语言效率很低。

我自己试了一下，同样长度的英文和中文：

- 英文："I like eating fried chicken" -> 5个token
- 中文："我喜欢吃炸鸡" -> 15个token

中文被拆成了3倍的token！这是因为`gpt2`的词表是针对英文设计的，中文字符只能被拆成多个字节级别的token。这也解释了为什么原版GPT-2生成中文会是乱码——模型把中文当成一堆无意义的字节序列来预测。

## T5：Text-to-Text的大一统

### 万物皆可Text-to-Text

T5的核心思想特别简洁：把所有NLP任务都转成"文本输入->文本输出"的形式。

- 翻译：`translate English to German: The house is wonderful.` -> `Das Haus ist wunderbar.`
- 分类：`sentiment: This movie is great!` -> `positive`
- 相似度：`stsb sentence1: ... sentence2: ...` -> `4.5`

注意最后这个例子——连回归任务都能转成文本生成（直接生成数字字符串）。这种统一框架的好处是不需要为不同任务设计不同的输出层，所有任务都用同一个Decoder生成文本。

### Span Corruption预训练

T5的预训练任务叫Span Corruption，相比BERT的单字mask更有挑战性：

1. 随机选择一些**连续片段**（平均长度3个token）
2. 用哨兵token（`<extra_id_0>`, `<extra_id_1>`...）替换
3. 让模型重建这些片段

举个例子：

- 原文：`黑神话悟空是一款以中国神话为背景的动作角色扮演游戏。`
- 输入：`黑神话悟空是一款<extra_id_0>的动作<extra_id_1>游戏。`
- 输出：`<extra_id_0>以中国神话为背景<extra_id_1>角色扮演<extra_id_2>`

这个设计很巧妙：

- 遮盖连续片段迫使模型理解更长的上下文
- 哨兵token让模型知道要恢复哪几段内容
- 输出也是序列形式，完美契合Encoder-Decoder架构

### 相对位置编码的精妙设计

T5的相对位置编码是个亮点。不同于BERT的绝对位置编码，T5认为重要的是词与词之间的**相对距离**，而不是绝对位置。

具体实现用了分桶策略：

- 近距离（<8）：精确区分每个距离
- 远距离（≥8）：用对数映射压缩到少数几个桶里

这符合语言学直觉：相邻词的关系（比如主谓）需要精确建模，但距离很远的词只需要知道"大概很远"就够了。

数学上，这些位置偏置不是加在输入embedding上，而是直接作为bias加在注意力分数矩阵上：

Attention(Q,K,V)=softmax(QKTdk+B)V\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right)VAttention(Q,K,V)=softmax(dkQKT+B)V

其中BBB就是相对位置偏置矩阵，而且这个BBB在所有层之间共享参数，大幅减少了参数量。

## Hugging Face实践

### 生态全景

Hugging Face现在已经不只是个代码库了，更像是AI领域的GitHub。核心组件包括：

- **Hub**：托管模型、数据集和演示应用
- **Transformers**：模型库的核心
- **Tokenizers**：Rust实现的高速分词器
- **Datasets**：数据处理工具

### Pipeline的便捷性

对于快速验证想法，Pipeline简直是神器：

```python
# 一行代码搞定情感分析
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")
```

它会自动：

1. 下载任务对应的默认模型
2. 进行分词预处理
3. 模型推理
4. 后处理得到结果

当然，实际项目中还是要手动控制每个环节的。

### AutoClass的智能加载

`AutoModel`这个设计很聪明。它能根据checkpoint名称自动识别模型架构：

```python
model = AutoModel.from_pretrained("bert-base-chinese")  # 自动加载BertModel
model = AutoModel.from_pretrained("gpt2")  # 自动加载GPT2Model
```

这避免了手动判断模型类型的麻烦。背后的原理是每个checkpoint都有一个`config.json`文件，里面记录了模型架构信息。

### Datasets的内存映射

`datasets`库有个杀手级特性：基于Apache Arrow的内存映射。这意味着可以处理比内存大得多的数据集。

```python
dataset = load_dataset("rotten_tomatoes")
tokenized = dataset.map(tokenize_function, batched=True, num_proc=4)
```

`map`函数支持：

- `batched=True`：批处理，加速分词
- `num_proc=4`：多进程并行

我在处理大数据集时真的体会到了这个设计的优雅——完全不用担心OOM。

### Trainer的封装层次

`Trainer`把训练流程封装得很好，但也保留了足够的灵活性：

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)
trainer.train()
```

它自动处理了：

- 混合精度训练
- 梯度累积
- 分布式训练（通过Accelerate）
- Checkpoint保存和恢复

对于标准任务，用Trainer能省很多工程化的代码。

## 一些思考

### 三种架构的适用场景

经过这次学习，我对这三种架构的理解：

- **BERT**：如果任务是分类、实体识别这种需要理解全文的，首选BERT。它的双向注意力能充分捕捉上下文。
- **GPT**：如果是生成任务（对话、续写、创作），GPT的自回归机制更自然。而且现在大模型时代，GPT这条路线基本统治了。
- **T5**：想用一个模型同时做多种任务时，T5的Text-to-Text框架很合适。但代价是需要Encoder和Decoder两部分，参数量更大。

### 预训练任务的设计哲学

回顾这三个模型的预训练任务：

- BERT的MLM：遮盖部分词，预测被遮盖的内容
- GPT的CLM：预测下一个词
- T5的Span Corruption：遮盖连续片段，生成完整序列

它们本质上都是在让模型"补全"文本，只是形式不同。这个过程迫使模型学习语言的内在规律。

有意思的是，越往后的模型，预训练任务越简单。GPT-3基本就是纯粹的next token prediction，没有什么花哨的设计，但配合巨大的规模和数据，效果反而最好。这或许说明：**简单的目标+足够的规模，可能比精巧的设计更有效**。

### 从微调到提示的转变

BERT时代，我们为每个任务训练一个专门的模型（虽然共享预训练权重）。GPT-3时代，我们通过设计prompt来"编程"一个通用模型。

这个转变背后的原因是什么？我觉得主要是：

1. 模型规模的爆炸式增长（110M -> 175B）
2. 训练数据的质量和多样性提升
3. 涌现能力（emergent abilities）的出现

但这不意味着微调就过时了。实际上，现在的大模型还是会做指令微调（Instruction Tuning）或RLHF。只是微调的目的变了：不再是适配特定任务，而是对齐人类偏好。

### 工程实践的启发

Hugging Face这套工具链给我最大的启发是：**好的抽象能大大降低使用门槛**。

- `AutoModel`让用户不用关心具体的模型类
- `pipeline`让用户不用关心预处理细节
- `Trainer`让用户不用写训练循环

但这些封装都是有层次的，你可以选择使用高层API快速开发，也可以深入底层精细控制。这种设计哲学值得学习。

## 参考文献

[1] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.

[2] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI blog.

[3] Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language Models are Few-Shot Learners. Advances in neural information processing systems, 33, 1877-1901.

[4] Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal of Machine Learning Research, 21(140), 1-67.

[5] Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692.

[6] Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (pp. 38-45).