# %% [markdown]
# # 智能体（AI Agents）
#
# Agent 这个词表示一个可以执行任务的实体，它可以是人类，也可以是一个程序。宽泛的讲，我们使用的任何软件都可以看作是一个 Agent，比如：
#
# - 浏览器：你通过浏览器上网，浏览器就是一个 Agent，它可以帮助你找到你想要的信息；
# - 打车软件：你通过打车软件打车，打车软件就是一个 Agent，它可以帮助你找到你想要的车；
#
# 这些既然都是 Agent，那之前为什么他们没有让 Agent 这个概念流行起来呢？因为他们没有“大脑”，或者说，他们没有“思考”的能力，他们更多的是根据预设的规则来执行任务，叫他们“工具”更合适。
#
# 比如：当你在浏览器输入一个网址然后点击跳转，他会直接跳转到目标网站，不会思考为什么这个网站会是你想要的，需不需要再找几个类似的网站给你；
#
# 直到大语言模型（Large Language Model, LLM）的出现，我们终于有了一个可以“思考”的工具，终于可以构建一个可以“思考”的 Agent 了，大家给他起了个名字，叫 AI Agent，翻译成中文就是“智能体”。以后，我们使用 Agent 来指代 AI Agent。
#
# 让 LLM 代替人的角色来思考，把那些没有“大脑”的工具用起来，就是构建 AI Agent 的思路。
#
# 构建智能体的第一步，需要先明确这个智能体要完成的任务是什么（注意，不是解决什么问题，而是完成什么任务）。
#
#
#

# %% [markdown]
# 接下来，我们将逐步探索基于大模型的应用，是如何一步步从简单的Prompt应用，一步步进化到智能的Agents。中间涉及到代码的部分，我们会使用 [LangChain](https://www.langchain.com/) 和 [LangGraph](https://langchain-ai.github.io/langgraph/) 来实现。

# %% [markdown]
# ## Chat models
#
# [Chat models](https://python.langchain.com/docs/concepts/chat_models/) 是大语言模型应用的基础，通常来说，我们通过API来调用它们，他们接受一组消息作为输入，并返回一个消息作为输出。
#
# LangChain 提供了标准化的接口来使用 Chat Models，可以让我们很轻松的使用和调整不同供应商所提供的模型。
#
# 运行下面的代码，需要申请 OPENAI_API_KEY，并设置环境变量，可以通过 https://github.com/chatanywhere/GPT_API_free 免费申请。

import os
from dotenv import load_dotenv
load_dotenv("../.env", override=True)
model_name = os.getenv("OPENAI_MODEL")
model_provider = os.getenv("MODEL_PROVIDER")
print(f"我们使用的模型是： {model_name}")

from langchain.chat_models import init_chat_model
from langchain.tools import tool
llm = init_chat_model(model_name, model_provider=model_provider, temperature=0)

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """撰写并发送邮件"""
    return f"邮件已发送给: {to}，主题为: {subject}，内容为:\n {content}"

model_with_tools = llm.bind_tools([write_email], tool_choice="any", parallel_tool_calls=False)
