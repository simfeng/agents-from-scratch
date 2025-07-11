# AI智能体从零开始实战教程

这是一个手把手教你构建AI智能体的完整教程。从最基础的概念开始，一步步带你搭建出真正能用的智能体。

项目灵感来自 [Agents From Scratch](https://github.com/langchain-ai/agents-from-scratch)，但我把内容重新整理了一遍，让它更好理解、更实用。

## 为什么做这个项目？

说实话，我自己学习智能体的时候踩了不少坑。网上的教程要么太理论化，要么跳跃性太大，新手很难跟上。所以我决定把自己的学习过程整理出来，用大白话把这些概念讲清楚。

**这个项目的所有代码和文档都是我在AI的帮助下完成的。**

技术栈主要用的是 LangChain 和 LangGraph。如果你之前没接触过也没关系，跟着教程走就行了。


## 教程内容

整个教程分为这几个部分：

- **AI智能体入门** - 什么是智能体，为什么要用它
- **LangGraph框架** - 构建智能体的核心工具
- **搭建智能体：邮件助手** - 从简单到复杂的实战项目
- **智能体评估** - 怎么知道你的智能体好不好用
- **人类干预机制** - 让人类可以介入智能体的决策过程
- **记忆系统** - 让智能体记住之前的对话
- **外部工具调用** - 让智能体能够使用各种API和服务

## 怎么使用这个教程？

每个章节的文件夹里都有：
- **README.md** - 详细的理论讲解和概念介绍
- **notebook.ipynb** - 可以直接运行的代码示例
- **其他项目文件** - 完整的代码实现（如果有的话）

建议的学习方式：
1. 先看README了解概念
2. 再跟着notebook动手写代码
3. 遇到问题就Google或者问AI

## 快速开始

1. **克隆项目**
   ```bash
   git clone https://github.com/simfeng/agents-from-scratch.git
   cd agents-from-scratch
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple/
   ```

3. **配置环境变量**
   - 复制 `.env.example` 文件为 `.env`
   - 填入你的API密钥（OpenAI、Claude等）

4. **开始学习**
   - 进入任意章节文件夹
   - 按照 README 和 notebook 的顺序学习

## 更新计划

这个项目我会持续更新，但是没有具体计划，每完成一个章节就发布。

## 进度追踪

**已完成：**
- ✅ 第1章：智能体入门（[博客](./01-agents-intro/README.md) + [Notebook](./01-agents-intro/notebook.ipynb) + [Colab](https://colab.research.google.com/github/simfeng/agents-from-scratch/blob/main/01-agents-intro/notebook.ipynb)）
- ✅ 第2章：LangGraph框架（[博客](./02-langgraph-intro/README.md) + [Notebook](./02-langgraph-intro/notebook.ipynb) + [Colab](https://colab.research.google.com/github/simfeng/agents-from-scratch/blob/main/02-langgraph-intro/notebook.ipynb)）
- ✅ 第3章：搭建智能体：邮件助手（[博客](./03-building-agents/README.md) + [Notebook](./03-building-agents/notebook.ipynb) + [Colab](https://colab.research.google.com/github/simfeng/agents-from-scratch/blob/main/03-building-agents/notebook.ipynb)）
- ✅ 第4章：智能体评估（[博客](./04-evaluation/README.md) + [Notebook](./04-evaluation/notebook.ipynb) + [Colab](https://colab.research.google.com/github/simfeng/agents-from-scratch/blob/main/04-evaluation/notebook.ipynb)）
- ✅ 第5章：人类干预机制（[博客](./05-human-in-the-loop/README.md) + [Notebook](./05-human-in-the-loop/notebook.ipynb) + [Colab](https://colab.research.google.com/github/simfeng/agents-from-scratch/blob/main/05-human-in-the-loop/notebook.ipynb)）
- ✅ 第8章：多智能体（Multi-Agent）介绍（[博客](https://schemax.tech/blog/anthropic-multi-agent-system-architecture)）

**进行中：**
- 🚧 第6章：Memory

**计划中：**
- ⏳ 第7章：外部工具调用

---

如果这个项目对你有帮助，欢迎给个⭐️支持一下！有问题可以提Issue，我会尽量回复。
