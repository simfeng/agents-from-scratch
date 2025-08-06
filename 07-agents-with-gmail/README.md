# AI Agents 实战项目第7篇：Gmail 集成（Gmail Integration）

将你的邮件助手与 Gmail 和 Google Calendar API 集成，实现真正的生产级邮件处理功能。

在前面的章节中，我们构建了具备记忆功能的邮件助手，它能够学习用户偏好并持续改进。但是，我们一直在使用模拟的邮件和日历工具。现在，让我们将这个助手连接到真实的 Gmail 和 Google Calendar API，让它能够处理你的实际邮件！

本节的 notebook 地址为：[https://github.com/simfeng/agents-from-scratch/blob/main/07-agents-with-gmail/notebook.ipynb](https://github.com/simfeng/agents-from-scratch/blob/main/07-agents-with-gmail/notebook.ipynb)

## 目录

- [Graph 配置](#graph-配置)
- [设置凭据](#设置凭据)
  - [1. 设置 Google Cloud 项目并启用必需的 API](#1-设置-google-cloud-项目并启用必需的-api)
  - [2. 设置身份验证文件](#2-设置身份验证文件)
- [本地部署使用](#本地部署使用)
- [托管部署使用](#托管部署使用)
- [Gmail 抓取原理](#gmail-抓取原理)
- [重要的 Gmail API 限制](#重要的-gmail-api-限制)

## Graph 配置

`src/email_assistant_hitl_memory_gmail.py` graph 已配置为使用 Gmail 工具。

你只需要运行下面的设置来获得使用你自己邮箱运行 graph 所需的凭据。

## 设置凭据

### 1. 设置 Google Cloud 项目并启用必需的 API

#### 启用 Gmail 和 Calendar API

1. 前往 [Google APIs Library 并启用 Gmail API](https://developers.google.com/workspace/gmail/api/quickstart/python#enable_the_api)
2. 前往 [Google APIs Library 并启用 Google Calendar API](https://developers.google.com/workspace/calendar/api/quickstart/python#enable_the_api)

#### 创建 OAuth 凭据

1. 在[这里](https://developers.google.com/workspace/gmail/api/quickstart/python#authorize_credentials_for_a_desktop_application)为桌面应用程序授权凭据
2. 前往 凭据 → 创建凭据 → OAuth 客户端 ID
3. 将应用程序类型设置为"桌面应用"
4. 点击"创建"

> 注意：如果使用个人邮箱（非 Google Workspace），请在"受众"下选择"外部"

<img width="1496" alt="Screenshot 2025-04-26 at 7 43 57 AM" src="https://github.com/user-attachments/assets/718da39e-9b10-4a2a-905c-eda87c1c1126" />

> 然后，将您自己添加为测试用户
 
5. 保存下载的 JSON 文件（下一步中您将需要它）

### 2. 设置身份验证文件

1. 将下载的客户端密钥 JSON 文件移动到 `.secrets` 目录

```bash
# 创建 secrets 目录
mkdir -p src/tools/gmail/.secrets

# 将下载的客户端密钥移动到 secrets 目录
mv /path/to/downloaded/client_secret.json src/tools/gmail/.secrets/secrets.json
```

2. 运行 Gmail 设置脚本

```bash
# 运行 Gmail 设置脚本
python src/tools/gmail/setup_gmail.py
```

- 这将打开一个浏览器窗口让您使用 Google 账户进行身份验证
- 这将在 `.secrets` 目录中创建一个 `token.json` 文件
- 该令牌将用于 Gmail API 访问

## 本地部署使用

### 1. 使用本地运行的 LangGraph 服务器运行 Gmail 抓取脚本

1. 设置好身份验证后，在本地运行 LangGraph 服务器：

```bash
langgraph dev
```

2. 在另一个终端中使用所需参数运行抓取脚本：

```bash
python src/tools/gmail/run_ingest.py --email <your-email@gmail.com> --minutes-since 1000
```

- 默认情况下，这将使用本地部署 URL（http://127.0.0.1:2024）并获取过去 1000 分钟内的邮件
- 它将使用 LangGraph SDK 将每封邮件传递给本地运行的邮件助手
- 它将使用配置为使用 Gmail 工具的 `email_assistant_hitl_memory_gmail` graph

#### 参数说明：

- `--graph-name`：要使用的 LangGraph 名称（默认："email_assistant_hitl_memory_gmail"）
- `--email`：要获取邮件的邮箱地址（可替代设置 EMAIL_ADDRESS）
- `--minutes-since`：仅处理比此分钟数更新的邮件（默认：60）
- `--url`：LangGraph 部署的 URL（默认：http://127.0.0.1:2024）
- `--rerun`：处理已经处理过的邮件（默认：false）
- `--early`：处理一封邮件后停止（默认：false）
- `--include-read`：包括已读邮件（默认仅处理未读邮件）
- `--skip-filters`：处理所有邮件而不过滤（默认仅处理您不是发送者的线程中的最新消息）

#### 故障排除：

- **邮件丢失？** Gmail API 默认应用过滤器只显示重要/主要邮件。您可以：
  - 将 `--minutes-since` 参数增加到更大的值（例如 1000）以获取更长时间段的邮件
  - 使用 `--include-read` 标志处理标记为"已读"的邮件（默认仅处理未读邮件）
  - 使用 `--skip-filters` 标志包括所有消息（不仅仅是线程中的最新消息，包括您发送的消息）
  - 尝试使用所有选项运行以处理所有内容：`--include-read --skip-filters --minutes-since 1000`
  - 使用 `--mock` 标志测试具有模拟邮件的系统

### 2. 连接到 Agent Inbox

抓取后，您可以在 Agent Inbox (https://dev.agentinbox.ai/) 中访问所有中断的线程：
* 部署 URL：http://127.0.0.1:2024
* 助手/Graph ID：`email_assistant_hitl_memory_gmail`
* 名称：`Graph Name`

## 托管部署使用

### 1. 部署到 LangGraph 平台

1. 导航到 LangSmith 中的部署页面
2. 点击新建部署
3. 将其连接到您的 [此仓库](https://github.com/langchain-ai/agents-from-scratch) 的 fork 和所需分支
4. 给它一个名称，如 `Yourname-Email-Assistant`
5. 添加以下环境变量：
   * `OPENAI_API_KEY`
   * `GMAIL_SECRET` - 这是 `.secrets/secrets.json` 中的完整字典
   * `GMAIL_TOKEN` - 这是 `.secrets/token.json` 中的完整字典
6. 点击提交
7. 从部署页面获取 `API URL`（https://your-email-assistant-xxx.us.langgraph.app）

### 2. 使用托管部署运行抓取

一旦您的 LangGraph 部署启动并运行，您可以使用以下命令测试邮件抓取：

```bash
python src/email_assistant/tools/gmail/run_ingest.py --email lance@langchain.dev --minutes-since 2440 --include-read --url https://your-email-assistant-xxx.us.langgraph.app
```

### 3. 连接到 Agent Inbox

抓取后，您可以在 Agent Inbox (https://dev.agentinbox.ai/) 中访问所有中断的线程：
* 部署 URL：https://your-email-assistant-xxx.us.langgraph.app
* 助手/Graph ID：`email_assistant_hitl_memory_gmail`
* 名称：`Graph Name`
* LangSmith API Key：`LANGSMITH_API_KEY`

### 4. 设置 Cron 作业

使用托管部署，您可以设置 cron 作业以指定间隔运行抓取脚本。

要自动化邮件抓取，请使用包含的设置脚本设置计划的 cron 作业：

```bash
python src/email_assistant/tools/gmail/setup_cron.py --email lance@langchain.dev --url https://lance-email-assistant-4681ae9646335abe9f39acebbde8680b.us.langgraph.app 
```

#### 参数说明：

- `--email`：要获取消息的邮箱地址（必需）
- `--url`：LangGraph 部署 URL（必需）
- `--minutes-since`：仅获取比此分钟数更新的邮件（默认：60）
- `--schedule`：Cron 计划表达式（默认："*/10 * * * *" = 每10分钟）
- `--graph-name`：要使用的 graph 名称（默认："email_assistant_hitl_memory_gmail"）
- `--include-read`：包括标记为已读的邮件（默认仅处理未读邮件）（默认：false）

#### Cron 如何工作

cron 由两个主要组件组成：

1. **`src/email_assistant/cron.py`**：定义一个简单的 LangGraph graph：
   - 调用 `run_ingest.py` 使用的相同 `fetch_and_process_emails` 函数
   - 将其包装在一个简单的 graph 中，以便可以使用 LangGraph Platform 作为托管 cron 运行

2. **`src/email_assistant/tools/gmail/setup_cron.py`**：创建计划的 cron 作业：
   - 使用 LangGraph SDK `client.crons.create` 为托管的 `cron.py` graph 创建 cron 作业

#### 管理 Cron 作业

要查看、更新或删除现有的 cron 作业，您可以使用 LangGraph SDK：

```python
from langgraph_sdk import get_client

# 连接到部署
client = get_client(url="https://your-deployment-url.us.langgraph.app")

# 列出所有 cron 作业
cron_jobs = await client.crons.list()
print(cron_jobs)

# 删除 cron 作业
await client.crons.delete(cron_job_id)
```

## Gmail 抓取原理

Gmail 抓取过程分为三个主要阶段：

### 1. CLI 参数 → Gmail 搜索查询

CLI 参数被转换为 Gmail 搜索查询：

- `--minutes-since 1440` → `after:TIMESTAMP`（过去24小时内的邮件）
- `--email you@example.com` → `to:you@example.com OR from:you@example.com`（您是发送者或接收者的邮件）
- `--include-read` → 移除 `is:unread` 过滤器（包括已读消息）

例如，运行：
```
python run_ingest.py --email you@example.com --minutes-since 1440 --include-read
```

创建的 Gmail API 搜索查询如下：
```
(to:you@example.com OR from:you@example.com) after:1745432245
```

### 2. 搜索结果 → 线程处理

对于搜索返回的每条消息：

1. 脚本获取线程 ID
2. 使用此线程 ID，获取包含所有消息的**完整线程**
3. 线程中的消息按日期排序以识别最新消息
4. 根据过滤选项，它会处理以下之一：
   - 搜索中找到的特定消息（默认行为）
   - 线程中的最新消息（使用 `--skip-filters` 时）

### 3. 默认过滤器和 `--skip-filters` 行为

#### 应用的默认过滤器

没有 `--skip-filters` 时，系统按顺序应用这三个过滤器：

1. **未读过滤器**（由 `--include-read` 控制）：
   - 默认行为：仅处理未读消息
   - 使用 `--include-read`：处理已读和未读消息
   - 实现：将 `is:unread` 添加到 Gmail 搜索查询
   - 此过滤器在检索任何消息之前在搜索级别发生

2. **发送者过滤器**：
   - 默认行为：跳过您自己邮箱地址发送的消息
   - 实现：检查您的邮箱是否出现在"From"标头中
   - 逻辑：`is_from_user = email_address in from_header`
   - 这防止助手回复您自己的邮件

3. **线程位置过滤器**：
   - 默认行为：仅处理每个线程中的最新消息
   - 实现：将消息 ID 与线程中的最后一条消息进行比较
   - 逻辑：`is_latest_in_thread = message["id"] == last_message["id"]`
   - 防止在存在更新回复时处理较旧的消息

这些过滤器的组合意味着只有每个线程中不是您发送且未读的最新消息（除非指定了 `--include-read`）才会被处理。

#### `--skip-filters` 标志的效果

启用 `--skip-filters` 时：

1. **绕过发送者和线程位置过滤器**：
   - 您发送的消息将被处理
   - 不是线程中最新的消息将被处理
   - 逻辑：`should_process = skip_filters or (not is_from_user and is_latest_in_thread)`

2. **改变处理的消息**：
   - 没有 `--skip-filters`：使用搜索中找到的特定消息
   - 使用 `--skip-filters`：始终使用线程中的最新消息
   - 即使最新消息不在搜索结果中

3. **未读过滤器仍然适用（除非被覆盖）**：
   - `--skip-filters` 不会绕过未读过滤器
   - 要处理已读消息，您仍必须使用 `--include-read`
   - 这是因为未读过滤器发生在搜索级别

总结：
- 默认：仅处理您不是发送者且是其线程中最新的未读消息
- `--skip-filters`：处理搜索找到的所有消息，使用每个线程中的最新消息
- `--include-read`：在搜索中包括已读消息
- `--include-read --skip-filters`：最全面，处理搜索找到的所有线程中的最新消息

## 重要的 Gmail API 限制

Gmail API 有几个影响邮件抓取的限制：

1. **基于搜索的 API**：Gmail 不提供直接的"从时间范围获取所有邮件"端点
   - 所有邮件检索都依赖于 Gmail 的搜索功能
   - 对于非常近期的消息，搜索结果可能会延迟（索引延迟）
   - 搜索结果可能不包括技术上符合条件的所有消息

2. **两阶段检索过程**：
   - 初始搜索以查找相关消息 ID
   - 二级线程检索以获取完整对话
   - 这种两阶段过程是必要的，因为搜索不能保证完整的线程信息

## 总结

通过本章的学习，我们成功地将邮件助手与真实的 Gmail 和 Google Calendar API 集成，实现了从原型到生产级应用的完整转变。这标志着我们的 AI 邮件助手正式具备了处理真实邮件的能力。

### 关键成就

通过 Gmail 集成，我们的邮件助手现在能够：

- **真实邮件处理**：直接从您的 Gmail 账户读取和处理邮件
- **智能过滤**：使用多层过滤机制确保只处理相关邮件
- **自动化工作流**：通过 cron 作业实现定时邮件处理
- **生产级部署**：支持本地开发和云端托管两种部署方式
- **人工监督**：结合 Agent Inbox 实现人机协作

### 系统架构优势

Gmail 集成版本具备以下重要特性：

1. **灵活的过滤系统**：
   - 默认只处理未读邮件，避免重复处理
   - 智能识别线程中的最新消息
   - 自动排除用户自己发送的邮件

2. **强大的参数控制**：
   - 时间窗口控制（`--minutes-since`）
   - 读取状态控制（`--include-read`）
   - 过滤策略控制（`--skip-filters`）

3. **可扩展的部署选项**：
   - 本地开发环境：适合测试和调试
   - 云端托管环境：适合生产使用
   - 自动化 cron 作业：实现无人值守运行

### 技术深度

本章涉及的技术要点包括：

- **OAuth 2.0 认证流程**：安全的 Google API 访问
- **Gmail API 复杂查询**：理解 Gmail 搜索语法和限制
- **线程处理逻辑**：正确识别和处理邮件对话
- **异步处理模式**：高效的邮件批处理
- **错误处理和重试机制**：提高系统稳定性

### 实际应用价值

Gmail 集成使我们的邮件助手从学习项目转变为实用工具：

- **日常工作助手**：自动处理例行邮件，释放人工时间
- **智能分类系统**：准确识别需要关注的重要邮件
- **个性化服务**：结合记忆功能，提供个性化的邮件处理
- **企业级应用**：支持大规模部署和自动化运维

这个完整的邮件助手系统展示了现代 AI Agent 开发的最佳实践：从基础功能到高级特性，从本地原型到生产部署，从单一功能到完整生态系统。它不仅是一个技术演示，更是一个可以直接投入使用的实用工具。

通过七个章节的递进学习，我们构建了一个功能完整、技术先进的 AI 邮件助手。它集成了工具调用、人机交互、记忆学习和真实 API 集成等多个重要概念，为理解和开发复杂 AI 系统打下了坚实基础。