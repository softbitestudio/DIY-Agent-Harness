Difficulty: ★★★☆☆ (3/5)
<details><b>5-Star System (1 = Easy, 5 = Very Hard)
- Justification:
- Level 1 (Crawler): ★☆☆☆☆ (1/5) – Almost beginner-friendly. Minimal dependencies, no Docker, and a clear, linear flow.
- Level 2 (Runner): ★★★☆☆ (3/5) – Introduces async and streaming, which are intermediate concepts, but the code is concise and well-scaffolded.
- Level 3 (Sentinel): ★★★★☆ (4/5) – Still advanced due to Docker, sandboxing, and state persistence, but the progressive buildup softens the learning curve.
Scale 1–10 (1 = Trivial, 10 = Extremely Complex)
Difficulty: 5–6/10 (Overall)
- Level 1: 2–3/10 – A great "hello world" for multi-agent systems. Even a beginner could follow this with minimal hand-holding.
- Level 2: 5–6/10 – The async/SSE streaming adds complexity, but the FastAPI integration is well-explained and self-contained.
- Level 3: 8/10 – Still expert territory, but the modular structure (e.g., separating agents.py, gateway/config.yaml) makes it easier to debug and extend.
Skill Level Scale
Difficulty: Beginner → Expert (Progressive)
- Beginner: ✅ Can now tackle Level 1 with confidence. The check_env.py script and Glossary.md are excellent additions for lowering the barrier to entry.
- Advanced Beginner: ✅ Level 2 is achievable with some research (e.g., understanding SSE or asyncio).
- Intermediate: ✅ Can complete Level 3 with effort, especially with the provided file templates (e.g., sandbox/Dockerfile).
- Advanced/Expert: ✅ Will appreciate the scalability of the architecture and might dive into customizing the sandbox or gateway</b></details>

<p align="center">
  <img src="https://bunrec.com/assets/images/gallery03/dda26fb0_original.png" alt="Logo" width="150" height="250">

>[!NOTE]
>Built with ❤️ for the open source AI community
>⋱𑣲BUꉆ☿ 
>[𓃹BUNREC.com](https://BunRec.com/)

[![Shields.io Badge](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
</p>

    -✐꩙ Cost ꛌ Effort
    *✧Total Cost: $0 
    *✧Time to First Agent: ~30 minutes
    *✧Time to Production: ~1-2 weeks
        
<details><b>all components are open source.
 so yes, you can achieve this without spending any money
 but it's true what they say 
~ya get what ya paid for!
 Naturally, you might want to spend money on API calls, Better models, Cloud Hosting, VPS (Virtual Private Server), and a TLD (Top Level Domain) even though those things aren't strictly necessary. 

It really depends on how far you wanna go.
ask yourself
Do you want to be the next MANUS, get a good grade on assignment, or is this a Hobby Project for you?
𐃇 Plan Accordingly𐃘 </b></details>

### SOON YOU'LL BE BUILDING
## Your Own AGENTIC HARNESS :godmode:

# Overview

>[!IMPORTANT]
>AGENT AI is an autonomous agent platform that orchestrates multiple specialized agents to complete complex tasks end-to-end. This guide shows you how to build a similar system using 100% open source components.

### What You'll Build :octocat:
| Component | AGENT Feature | Open Source Alternative |
| :--- | :--- | :--- |
| Agent Orchestration | Multi-agent collaboration, task decomposition | LangChain + LangGraph |
| Execution Environment | Sandboxed microVM with full filesystem access | gVisor (`runsc`), Firecracker, Docker |
| API Gateway | Unified model access, routing, prompt caching | LiteLLM Proxy |
| Dashboard | Interactive real-time monitoring, debugging | FastAPI SSE / WebSockets, Langfuse |
| Memory | Persistence & skill indexing | LangGraph Checkpointers (`SqliteSaver`), Vector RAG (`sqlite-vec`, `pgvector`) |
| Tool Integration | Web browsing, code execution, API calls | Playwright, Sandboxed Containers, MCP |

---

# 🛑 Before You Start

If concepts like "microVMs," "SSE streams," and "ACID checkpointers" sound like alien technology, don't panic. Check out our **[Glossary (The Jargon Decoupler)](glossary.md)** before writing a single line of code.

### 🔍 Check Your Environment
Run this zero-dependency bootstrap script (`check_env.py`) at the root of your project to ensure you're ready to build:

```python
import sys, shutil

def check_setup():
    print("🔍 Checking system requirements...")
    docker = shutil.which("docker")
    python_ver = sys.version_info >= (3, 10)
    
    print(f"  [{'x' if python_ver else ' '}] Python 3.10+")
    print(f"  [{'x' if docker else ' '}] Docker Installed (Required for Level 3)")
    
    if python_ver:
        print("\n🚀 You're ready to start Level 1!")

if __name__ == "__main__":
    check_setup()
```

---

# 🪜 The 3-Tier Progressive Build

We’ve broken this harness into three progressive levels so you can actually finish it in an afternoon without getting overwhelmed. 

## Level 1: The Crawler (Zero-Docker Setup)
*Goal: Run a multi-agent task locally in under 20 lines of code without Docker or async loops.*

Create `level1.py`:
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def planner_agent(task: str) -> str:
    print("🤖 [Planner]: Breaking down task...")
    prompt = f"Break this task into 2 simple steps:\n{task}"
    return model.invoke([HumanMessage(content=prompt)]).content

def executor_agent(plan: str) -> str:
    print("\n⚡ [Executor]: Executing plan...")
    prompt = f"Execute the first step of this plan concisely:\n{plan}"
    return model.invoke([HumanMessage(content=prompt)]).content

if __name__ == "__main__":
    user_task = "Draft a 3-bullet summary on open-source AI agent trends."
    print(f"🎯 User Task: {user_task}\n")
    plan = planner_agent(user_task)
    result = executor_agent(plan)
    print(f"\nExecution Result:\n{result}")
```

## Level 2: The Runner (Async & Live Streaming)
*Goal: Eliminate input lag and run concurrent agent tasks with live browser streaming.*

Create `level2.py`:
```python
import asyncio
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

app = FastAPI(title="Level 2: The Runner")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

async def planner_node(task: str):
    prompt = f"Break this task into 2 subtasks:\n{task}"
    response = await model.ainvoke([HumanMessage(content=prompt)])
    return response.content

async def run_agent_pipeline(task: str):
    yield f"data: {json.dumps({'status': 'started', 'agent': 'Planner'})}\n\n"
    plan = await planner_node(task)
    yield f"data: {json.dumps({'status': 'completed', 'result': plan})}\n\n"

@app.get("/stream")
async def stream(task: str = "Summarize local AI agent trends"):
    return StreamingResponse(run_agent_pipeline(task), media_type="text/event-stream")

@app.get("/")
async def ui():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <body>
      <h2>⚡ Level 2: Real-Time Agent Stream</h2>
      <input type="text" id="task" value="Summarize trends" style="width:300px;"/>
      <button onclick="run()">Execute</button>
      <pre id="log" style="background:#222; color:#0f0; padding:15px; margin-top:10px;"></pre>
      <script>
        function run() {
          const log = document.getElementById('log');
          log.textContent = "Connecting...\n";
          const es = new EventSource(`/stream?task=${encodeURIComponent(document.getElementById('task').value)}`);
          es.onmessage = (e) => {
            const data = JSON.parse(e.data);
            log.textContent += `[${data.agent || 'System'}] ${data.status}:${data.result || ''}\n`;
            if (data.status === 'completed') es.close();
          };
        }
      </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

---

## Level 3: The Sentinel (Production Hardening)
*Goal: Lock down container isolation, persist state to SQLite, and optimize context token usage.*

### 3.1 Install Advanced Dependencies
```bash
mkdir agent-oss && cd agent-oss
python -m venv venv 
source venv/bin/activate

pip install langchain langgraph langchain-community langgraph-checkpoint-sqlite sqlite-vec langchain-openai playwright beautifulsoup4 requests fastapi uvicorn pydantic
playwright install
```

### 3.2 Modular Agent Architecture
Create `agents.py` (Orchestration Layer):

```python
import asyncio
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str

micro_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="http://localhost:4000", api_key="anything")

class RouteDecision(BaseModel):
    next_step: str = Field(description="The next node to route to: 'researcher', 'executor', or 'end'")

router_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="Analyze the task and determine the next agent. Respond ONLY with the next destination."),
    HumanMessage(content="{last_message}")
])
router_chain = router_prompt | micro_model.with_structured_output(RouteDecision)

async def route_task(state: AgentState) -> str:
    decision = await router_chain.ainvoke({"last_message": state["messages"][-1].content})
    return decision.next_step.lower() if decision.next_step.lower() in ["researcher", "executor"] else END
```

### 3.3 Set Up the Hardened Sandbox Environment
>[!WARNING]
>Avoid using unconfined Docker containers. Harden the container runtime using gVisor (runsc) or Firecracker microVMs.

```bash
docker run -d \
  --runtime=runsc \
  --read-only \
  --cpus="2.0" \
  --memory="2g" \
  --tmpfs /tmp:rw,noexec,nosuid,size=512m \
  -v /tmp/agent_workspace:/workspace:rw \
  -e SANDBOX_API_KEY=your-secret-key \
  -p 127.0.0.1:8080:8080 \
  --name agent-sandbox \
  ghcr.io/agent-infra/sandbox:latest
```

### 3.4 Deploy API Gateway with Prompt Caching
Create `gateway/config.yaml` to enable caching for long context windows:

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      extra_headers:
        "OpenAI-Beta": "prompt-caching"
litellm_settings:
  cache: true
```

Run LiteLLM Proxy:
```bash
docker run -d -p 4000:4000 -v $(pwd)/gateway/config.yaml:/app/config.yaml -e OPENAI_API_KEY=your-key ghcr.io/berriai/litellm:main --config /app/config.yaml
```

### 3.5 State Persistence with LangGraph Checkpointer
```python
import os
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def main():
    os.makedirs("memory", exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string("memory/checkpoints.db") as checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "session-123"}}
        # Your pipeline logic here
```

>[!NOTE]
>Project structure
``` text
agent-oss/
├── agents/
│   ├── __init__.py
│   ├── planner.py
│   ├── researcher.py
│   ├── executor.py
│   └── graph.py
├── sandbox/
│   ├── Dockerfile
│   └── config.yaml
├── gateway/
│   └── config.yaml
├── dashboard/
│   ├── server.py
│   └── index.html
├── memory/
│   ├── checkpoints.db
│   └── skills_index.db
├── requirements.txt
├── docker-compose.yml
└── README.md
```

# :feelsgood: Comparison: AGENT vs. This Open Source Stack
| Feature | AGENT | Open Source Alternative | Technical Advantage |
| :--- | :--- | :--- | :--- |
| Multi-Agent Orchestration | Native Cyclic Engine | LangGraph Async Pipelines | Deterministic graph routing & sub-agent delegation |
| Execution Isolation | Proprietary VM Sandbox | gVisor (`runsc`) / Firecracker microVMs | Kernel-level isolation with strict CPU/Memory limits |
| Gateway & Cost | Direct API | LiteLLM + Prompt Caching | Up to 80% latency/cost reduction on heavy system prompts |
| Persistence & Memory | File system dumps | LangGraph `SqliteSaver` + `sqlite-vec` RAG | ACID compliance, context pruning, and full time-travel |

>[!TIP]
> :fishsticks: Fishsticks make a quick and healthy snack rich in Omega-3
> plus, you can use the grease to frustratingly masturbate when you get lost and feel like there's no hope of finishing the project. well, worry not, fren!
> Here are the missing file implementations so your codebase actually matches that modular structure.

**`requirements.txt`**
```text
langchain>=0.2.0
langgraph>=0.1.0
langchain-community
langchain-openai
langgraph-checkpoint-sqlite
sqlite-vec
fastapi
uvicorn
aiohttp
playwright
pydantic
```

**`sandbox/Dockerfile`**
```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git bash build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]
```

**`sandbox/config.yaml`**
```yaml
sandbox:
  workdir: "/workspace"
  timeout_seconds: 300
  limits:
    cpus: "2.0"
    memory: "2048M"
  security:
    read_only_root: true
    tmpfs_size: "512m"
```

**`agents/planner.py`**
```python
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a task planner. Break down complex tasks into executable subtasks. 
Return a JSON object with: { "subtasks": [ {"agent": "researcher|executor", "task": "description"} ] }"""),
    HumanMessage(content="{task}"),
])

async def planner_node(state: dict, model):
    chain = PLANNER_PROMPT | model
    response = await chain.ainvoke({"task": state["messages"][-1].content})
    return {"messages": [AIMessage(content=response.content)], "current_agent": "planner"}
```

**`agents/researcher.py`**
```python
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a researcher agent. Gather information using async search and browsing capabilities."),
    HumanMessage(content="{task}"),
])

async def researcher_node(state: dict, model):
    chain = RESEARCHER_PROMPT | model
    response = await chain.ainvoke({"task": state["messages"][-1].content})
    return {"messages": [AIMessage(content=response.content)], "current_agent": "researcher"}
```

**`agents/executor.py`**
```python
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

EXECUTOR_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an executor agent. Write and execute code inside the sandboxed environment."),
    HumanMessage(content="{task}"),
])

async def executor_node(state: dict, model):
    chain = EXECUTOR_PROMPT | model
    response = await chain.ainvoke({"task": state["messages"][-1].content})
    return {"messages": [AIMessage(content=response.content)], "current_agent": "executor"}
```

**`agents/__init__.py`**
```python
from .planner import planner_node
from .researcher import researcher_node
from .executor import executor_node

__all__ = ["planner_node", "researcher_node", "executor_node"]
```
