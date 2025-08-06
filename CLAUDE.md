# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive tutorial for building AI agents from scratch using LangChain and LangGraph. The project progressively teaches agent development concepts through practical examples, culminating in a production-ready email assistant with Gmail integration.

## Development Commands

### Core Dependencies
- Install dependencies: `pip install -r requirements.txt`
- The project uses Python 3.12 as specified in `langgraph.json`

### LangGraph Development
- Start local development server: `langgraph dev`
  - Default URL: http://127.0.0.1:2024
  - Available graphs defined in `langgraph.json`:
    - `email_assistant` - Basic email assistant
    - `email_assistant_hitl` - With human-in-the-loop capability
    - `email_assistant_hitl_memory` - With memory system
    - `email_assistant_hitl_memory_gmail` - Full Gmail integration

### Testing
- Run tests: `pytest`
- Run tests for specific agent: `pytest --agent-module=email_assistant_hitl_memory`
- Test configuration is in `tests/conftest.py`

### Gmail Integration
- Set up Gmail credentials: `python src/tools/gmail/setup_gmail.py`
- Run email ingestion: `python src/tools/gmail/run_ingest.py --email your@email.com --minutes-since 1000`
- Set up automated cron job: `python src/tools/gmail/setup_cron.py --email your@email.com --url https://your-deployment-url`

## Architecture

### Core Agent Structure
All agents follow a consistent StateGraph pattern with these key components:
- **Triage Router** (`triage_router`): Classifies emails as ignore/notify/respond
- **Response Agent** (`response_agent`): Handles actual email processing and tool calls
- **State Management**: Uses LangGraph's MessagesState with custom email and classification fields

### Agent Evolution Progression
1. **Basic Agent** (`src/email_assistant.py`): Simple tool-calling agent
2. **HITL Agent** (`src/email_assistant_hitl.py`): Adds human interruption capability
3. **Memory Agent** (`src/email_assistant_hitl_memory.py`): Includes user preference learning
4. **Gmail Agent** (`src/email_assistant_hitl_memory_gmail.py`): Full Gmail API integration

### Key Components

#### State Schema (`src/schemas.py`)
- `State`: Core state with messages, email input, and classification decision
- `RouterSchema`: Structured output for email triage decisions
- `EmailData`: Gmail message structure
- `UserPreferences`: Memory system for learning user preferences

#### Tools Architecture (`src/tools/`)
- **Base Tools** (`src/tools/base.py`): Tool loading and management utilities
- **Default Tools** (`src/tools/default/`): Mock email and calendar tools for development
- **Gmail Tools** (`src/tools/gmail/`): Production Gmail and Google Calendar APIs
- Tools are dynamically loaded based on `include_gmail` parameter

#### Prompt System (`src/prompts.py`)
- Modular prompt templates with placeholders for customization
- Separate prompts for triage, basic agent, HITL, and memory variants
- Default configurations for background, preferences, and triage rules
- Memory update instructions for preference learning

### Evaluation Framework (`src/eval/`)
- Email dataset generation and management
- Automated triage evaluation with comparison metrics
- Results visualization and analysis tools

## Environment Configuration

### Required Environment Variables
- `OPENAI_API_KEY`: OpenAI API access
- `MODEL_PROVIDER`: Set to "openai" 
- `OPENAI_MODEL`: Model name (e.g., "gpt-4")
- `LANGCHAIN_API_KEY`: For LangSmith tracking (optional)
- `LANGCHAIN_PROJECT`: Project name for LangSmith (optional)

### Gmail Integration Setup
1. Create `.secrets` directory in `src/tools/gmail/`
2. Place Google OAuth credentials as `secrets.json`
3. Run setup script to generate `token.json`
4. For hosted deployments, set `GMAIL_SECRET` and `GMAIL_TOKEN` environment variables

## Key Development Patterns

### Agent Node Pattern
All agents use this consistent node structure:
- `llm_call`: LLM makes tool decisions
- `tool_node`/`environment`: Executes selected tools
- `should_continue`: Routes between tool execution and completion

### Tool Integration
- Tools are loaded via `get_tools()` function with optional Gmail inclusion
- Tools use consistent schema with `name`, `description`, and `args_schema`
- Tool execution results are formatted as messages in conversation history

### Human-in-the-Loop
- Interrupts occur before tool execution via `interrupt="before"`
- Agent Inbox integration for reviewing and editing responses
- Memory system learns from user feedback and edits

### Memory System
- User preferences stored per namespace (email address)
- Selective updates preserve existing information
- Integration with conversation history for context-aware responses

## Deployment Options

### Local Development
- Use `langgraph dev` for local testing with hot reload
- Agent Inbox available at https://dev.agentinbox.ai/ for HITL workflows

### LangGraph Platform
- Deploy via LangSmith deployments page
- Connect to GitHub repository
- Set required environment variables
- Support for scheduled cron jobs via LangGraph SDK

## Gmail API Integration Details

### Email Processing Pipeline
1. **Search**: Gmail API search with time/read status filters
2. **Thread Retrieval**: Fetch complete conversation threads
3. **Processing**: Apply sender and thread-position filters
4. **Agent Invocation**: Send to appropriate LangGraph endpoint

### Key Filtering Options
- `--include-read`: Process read messages (default: unread only)
- `--skip-filters`: Process all messages including self-sent
- `--minutes-since`: Time window for email retrieval
- `--rerun`: Reprocess already-handled emails

## Tutorial Structure

Each chapter (01-07) contains:
- `README.md`: Theoretical concepts and explanations
- `notebook.ipynb`: Hands-on code examples
- Supporting images in `img/` directories
- Complete working code examples where applicable

The progression builds from basic agent concepts to production-ready Gmail integration with memory and human oversight capabilities.