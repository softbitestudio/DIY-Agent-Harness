<p align="center">
  <img src="https://bunrec.com/assets/images/gallery03/dda26fb0_original.png" alt="Logo" width="150" height="250">

>[!NOTE]
>Built with ❤️ for the open source AI community
⋱𑣲BUꉆ☿ 
[𓃹BUNREC.com](https://BunRec.com/)

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

What You'll Build :octocat:
| Component | AGENT Feature | Open Source Alternative |
| :--- | :--- | :--- |
| Agent Orchestration | Multi-agent collaboration, task decomposition | LangChain + LangGraph |
| Execution Environment | Sandboxed microVM with full filesystem access | gVisor (`runsc`), Firecracker, Docker |
| API Gateway | Unified model access, routing, prompt caching | LiteLLM Proxy |
| Dashboard | Interactive real-time monitoring, debugging | FastAPI SSE / WebSockets, Langfuse |
| Memory | Persistence & skill indexing | LangGraph Checkpointers (`SqliteSaver`), Vector RAG (`sqlite-vec`, `pgvector`) |
| Tool Integration | Web browsing, code execution, API calls | Playwright, Sandboxed Containers, MCP |


# :gear: Step 1: Set Up the Agent Orchestration Layer
Option A: Async LangChain + LangGraph (Recommended for Production)
LangGraph provides precise control over stateful, cyclic multi-agent systems with native checkpointing, branching, and async execution support.

## 1.1 Install Dependencies


#### Create a new project directory

```
mkdir agent-oss && cd agent-oss
```
^replace `agent-oss` with w/e you're calling your agent^
^*i.e. Hermes, Odysseus, Bunnyclaw, etc.*^
# Create virtual environment
```bash
python -m venv venv 
source venv/bin/activate # On Windows: venv\Scripts\activate
```

# Install core packages and async drivers
```bash
pip install langchain langgraph langchain-community langgraph-checkpoint-sqlite
pip install sqlite-vec langchain-openai
pip install playwright beautifulsoup4 requests fastapi uvicorn
playwright install
```

1.2 Create Your Asynchronous Multi-Agent System
Create `agents.py`:

```python
import asyncio
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[list, "Messages in the conversation"]
    current_agent: str
    task_progress: dict

# Initialize models via LiteLLM Proxy Gateway
# Frontier model for complex synthesis & planning
frontier_model = ChatOpenAI(
    model="gpt-4o", 
    temperature=0, 
    base_url="http://localhost:4000", 
    api_key="anything"
)

# Micro-model dedicated to lightweight execution tasks
micro_model = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0, 
    base_url="http://localhost:4000", 
    api_key="anything"
)

# Prompts
PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a task planner. Break down complex tasks into executable subtasks. 
Return a JSON object with: { "subtasks": [ {"agent": "researcher|executor", "task": "description"} ] }"""),
    HumanMessage(content="{task}"),
])

RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a researcher agent. Gather information using async search and browsing capabilities."),
    HumanMessage(content="{task}"),
])

EXECUTOR_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an executor agent. Write and execute code inside the sandboxed environment."),
    HumanMessage(content="{task}"),
])

# Async Agent Nodes
async def planner_node(state: AgentState):
    chain = PLANNER_PROMPT | frontier_model
    response = await chain.ainvoke({"task": state["messages"][-1].content})
    return {"messages": [AIMessage(content=response.content)], "current_agent": "planner"}

async def researcher_node(state: AgentState):
    chain = RESEARCHER_PROMPT | micro_model
    response = await chain.ainvoke({"task": state["messages"][-1].content})
    return {"messages": [AIMessage(content=response.content)], "current_agent": "researcher"}

async def executor_node(state: AgentState):
    chain = EXECUTOR_PROMPT | frontier_model
    response = await chain.ainvoke({"task": state["messages"][-1].content})
    return {"messages": [AIMessage(content=response.content)], "current_agent": "executor"}

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("executor", executor_node)

workflow.set_entry_point("planner")
```

## 1.3 Add Micro-Model Task Delegation Routing
Route conditional edges using fast micro-models instead of wasting high-parameter frontier models on conditional structural checks.

```python
from pydantic import BaseModel, Field

class RouteDecision(BaseModel):
    next_step: str = Field(description="The next node to route to: 'researcher', 'executor', or 'end'")

# Dedicated routing chain with a micro-model
router_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="Analyze the task and determine the next agent. Respond ONLY with the next destination."),
    HumanMessage(content="{last_message}")
])
router_chain = router_prompt | micro_model.with_structured_output(RouteDecision)

async def route_task(state: AgentState) -> str:
    """Micro-Model Delegation for Fast Conditional Routing"""
    last_msg = state["messages"][-1].content
    decision = await router_chain.ainvoke({"last_message": last_msg})
    
    if decision.next_step.lower() == "researcher":
        return "researcher"
    elif decision.next_step.lower() == "executor":
        return "executor"
    return END

# Add conditional routing edges
workflow.add_conditional_edges(
    "planner", 
    route_task, 
    {"researcher": "researcher", "executor": "executor", END: END}
)
workflow.add_conditional_edges(
    "researcher", 
    route_task, 
    {"executor": "executor", END: END}
)
```

## Step 2: Set Up the Hardened Sandbox Environment

>[!WARNING]
>This is a warning in a box.

***Avoid using unconfined Docker containers. Harden the container runtime using gVisor (runsc) or Firecracker microVMs alongside read-only root filesystems and strict CPU/memory quotas.***

## 2.1 Run Hardened Sandbox with gVisor
 Pull and execute the sandbox with gVisor
```bash
 runtime and strict resource constraints
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


## 2.2 Integrate Sandbox Client in Async Pipelines

```python
import aiohttp

class HardenedSandbox:
    def __init__(self, base_url: str = 'http://localhost:8080', api_key: str = 'your-secret-key'):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_key}'}

    async def execute_command(self, command: str) -> str:
        """Executes shell script asynchronously inside the isolated gVisor container"""
        async with aiohttp.ClientSession() as session:
            payload = {"command": command, "workdir": "/workspace"}
            async with session.post(f"{self.base_url}/shell/exec", json=payload, headers=self.headers) as resp:
                res = await resp.json()
                return res.get("output", "")

# Usage inside an async executor node
sandbox = HardenedSandbox()

async def execute_code_node(state: AgentState):
    code = state["messages"][-1].content
    output = await sandbox.execute_command(f"python3 -c '{code}'")
    return {"messages": [AIMessage(content=output)], "current_agent": "executor"}
```

## Step 3: Deploy API Gateway with Prompt Caching

## 3.1 LiteLLM Proxy Configuration with Caching enabled
Create `config.yaml` to enable prompt caching headers for long context windows and agent system prompts:

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      extra_headers:
        "OpenAI-Beta": "prompt-caching"
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
      cache_control: {"type": "ephemeral"}

router_settings:
  routing_strategy: usage-based-routing

litellm_settings:
  cache: true
  cache_params:
    type: "redis"
    supported_call_types: ["acontext_embedding", "acompletion"]
    host: "localhost"
    port: 6379
```

## 3.2 Run LiteLLM Proxy

```bashdocker run -d \
  -p 4000:4000 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -e OPENAI_API_KEY=your-key \
  -e ANTHROPIC_API_KEY=your-key \
  ghcr.io/berriai/litellm:main --config /app/config.yaml
```

## Step 4: Server-Sent Events (SSE) Real-Time UI
Eliminate input lag and re-render loops by serving a lightweight Server-Sent Events (SSE) stream via FastAPI.

***4.1 FastAPI Real-Time Streaming Server***
Create `server.py`:

```python
import asyncio
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def agent_event_generator(task: str):
    """Streams task execution progress over SSE"""
    yield f"data: {json.dumps({'status': 'started', 'agent': 'planner'})}\n\n"
    await asyncio.sleep(1)
    
    yield f"data: {json.dumps({'status': 'executing', 'agent': 'researcher', 'chunk': 'Gathering live metrics...'})}\n\n"
    await asyncio.sleep(1.5)
    
    yield f"data: {json.dumps({'status': 'executing', 'agent': 'executor', 'chunk': 'Running sandboxed code...'})}\n\n"
    await asyncio.sleep(1)
    
    yield f"data: {json.dumps({'status': 'completed', 'result': 'Task completed successfully.'})}\n\n"

@app.get("/stream")
async def stream_agent_execution(task: str):
    return StreamingResponse(agent_event_generator(task), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 4.2 Lightweight HTML SSE Client
Create `index.html`:

>[!IMPORTANT]
> OBVI THIS IS A PLACEHOLDER
> SPICE IT UP WITH YOUR OWN STYLE :fishcake:

```html
<!DOCTYPE html>
<html>
<head><title>AGENT Dashboard</title></head>
<body>
  <h2>🤖 AGENT Stream Dashboard</h2>
  <input type="text" id="taskInput" placeholder="Enter task..." style="width: 300px;"/>
  <button onclick="startStream()">Execute</button>
  <div id="output" style="margin-top:20px; white-space:pre-wrap; font-family:monospace; background:#f4f4f4; padding:10px;"></div>

  <script>
    function startStream() {
      const task = document.getElementById('taskInput').value;
      const outputDiv = document.getElementById('output');
      outputDiv.innerHTML = "Connecting...\n";

      const eventSource = new EventSource(`http://localhost:8000/stream?task=${encodeURIComponent(task)}`);

      eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        outputDiv.innerHTML += `[${data.agent || 'system'}] ${data.chunk || data.status}\n`;
        if (data.status === 'completed') {
          eventSource.close();
        }
      };
    }
  </script>
</body>
</html>

```

>[!TIP]
>  :fishsticks:

## Step 5: State Persistence & Hybrid Memory Indexing
Replace manual file serialization (JSON dumps) and flat SKILL.md structures with ACID-compliant LangGraph state checkpointers and Vector RAG retrieval.

5.1 ACID State Persistence with LangGraph Checkpointer

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiofiles

async def main():
    # Native SQLite Async Checkpointer for thread management and state time-travel
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
        
        # Configuration thread ID for persistent context
        config = {"configurable": {"thread_id": "session-123"}}
        
        inputs = {"messages": [HumanMessage(content="Analyze latest tech news")], "current_agent": "", "task_progress": {}}
        
        async for event in app.astream(inputs, config=config):
            for k, v in event.items():
                print(f"State Update Node: {k}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 5.2 Hybrid Memory Indexing with Vector RAG
Embed skill definitions and historical output tasks into a lightweight vector database (==sqlite-vec== or Supabase ==pgvector==) to inject relevant skills into the context window only when needed.

```python
from langchain_community.vectorstores import SQLiteVec
from langchain_openai import OpenAIEmbeddings

# Initialize Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Vector Store Initialization for Skills & Context Retrieval
vector_db = SQLiteVec(
    table="agent_skills",
    db_file="skills_index.db",
    embedding=embeddings
)

async def index_skill(skill_name: str, content: str):
    """Embeds skill markdown text into sqlite-vec"""
    await vector_db.aadd_texts(texts=[content], metadatas=[{"skill_name": skill_name}])

async def retrieve_relevant_skills(query: str) -> str:
    """Fetch relevant skill context dynamically based on task similarity"""
    docs = await vector_db.asimilarity_search(query, k=2)
    return "\n\n".join([doc.page_content for doc in docs])
'''

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

## Step 6: :shipit: Deployment Options
Docker Compose Stack with Hardened Runtime

```yaml
version: '3.8'

services:
  gateway:
    image: ghcr.io/berriai/litellm:main
    ports:
      - "4000:4000"
    volumes:
      - ./gateway/config.yaml:/app/config.yaml
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    restart: unless-stopped

  sandbox:
    image: ghcr.io/agent-infra/sandbox:latest
    runtime: runsc
    read_only: true
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2048M
    tmpfs:
      - /tmp:rw,noexec,nosuid,size=512m
    ports:
      - "8080:8080"
    environment:
      - SANDBOX_API_KEY=${SANDBOX_API_KEY}
    restart: unless-stopped

  agents-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LITELLM_BASE_URL=http://gateway:4000
      - SANDBOX_BASE_URL=http://sandbox:8080
    depends_on:
      - gateway
      - sandbox
    restart: unless-stopped
```

# :feelsgood: Comparison: AGENT vs. This Open Source Stack
| Feature | AGENT | Open Source Alternative | Technical Advantage |
| :--- | :--- | :--- | :--- |
| Multi-Agent Orchestration | Native Cyclic Engine | LangGraph Async Pipelines | Deterministic graph routing & sub-agent delegation |
| Execution Isolation | Proprietary VM Sandbox | gVisor (`runsc`) / Firecracker microVMs | Kernel-level isolation with strict CPU/Memory limits |
| Gateway & Cost | Direct API | LiteLLM + Prompt Caching | Up to 80% latency/cost reduction on heavy system prompts |
| Persistence & Memory | File system dumps | LangGraph `SqliteSaver` + `sqlite-vec` RAG | ACID compliance, context pruning, and full time-travel |
| Client Streaming | Proprietary UI | FastAPI SSE / WebSockets | Low-latency real-time token and state emission |

