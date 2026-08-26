# DIY-Agent-Harness
how to build your own Ai harness like Manus

🚀 Build Your Own Manus: Open Source AI Agent Platform Guide

Step-by-step guide to creating an autonomous multi-agent system with interactive dashboard and API gateway

📋 Overview

Manus AI is an autonomous agent platform that orchestrates multiple specialized agents to complete complex tasks end-to-end. This guide shows you how to build a similar system using 100% open source components.

What You'll Build

ComponentManus FeatureOpen Source AlternativeAgent OrchestrationMulti-agent collaboration, task decompositionLangChain + LangGraphExecution EnvironmentSandboxed VM with full filesystem accessAIO Sandbox / OpenSandboxAPI GatewayUnified model access, routingLiteLLMDashboardInteractive monitoring, debuggingLangfuse / Custom StreamlitMemoryFile-based progress trackingLangGraph CheckpointersTool IntegrationWeb browsing, code execution, API callsPlaywright, Docker, MCP 

🏗️ Architecture Diagram

flowchart TB subgraph Client["📱 Client Applications"] A[Web UI] B[API Clients] C[CLI] end subgraph Gateway["🚪 API Gateway"] D[LiteLLM Proxy] end subgraph Orchestration["🤖 Agent Orchestration"] E[LangGraph] F[LangChain] G[Agent Definitions] end subgraph Execution["🔧 Execution Environment"] H[AIO Sandbox] I[Docker Containers] end subgraph Observability["📊 Dashboard & Observability"] J[Langfuse] K[Custom UI] end subgraph Models["🧠 AI Models"] L[OpenAI] M[Anthropic] N[Local LLMs] end Client -->|HTTP/HTTPS| Gateway Gateway -->|Unified API| Orchestration Orchestration -->|Tasks| Execution Execution -->|Results| Orchestration Orchestration -->|Metrics| Observability Orchestration -->|LLM Calls| Models Observability -->|Dashboard| Client 

⚙️ Step 1: Set Up the Agent Orchestration Layer

Option A: LangChain + LangGraph (Recommended for Production)

Why? LangGraph is the most mature open source framework for stateful, cyclic multi-agent systems with precise control over execution order, branching, and error recovery.

1.1 Install Dependencies

# Create a new project directory mkdir manus-oss && cd manus-oss # Create virtual environment python -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate # Install core packages pip install langchain langgraph langchain-community # Install additional tools pip install playwright beautifulsoup4 requests playwright install 

1.2 Create Your First Multi-Agent System

Create agents.py:

from typing import TypedDict, Annotated from langchain_core.messages import HumanMessage, AIMessage, SystemMessage from langchain_core.prompts import ChatPromptTemplate from langchain_openai import ChatOpenAI from langgraph.graph import StateGraph, END from langgraph.prebuilt import ToolNode # Define agent state class AgentState(TypedDict): messages: Annotated[list, "Messages in the conversation"] current_agent: str task_progress: dict # Initialize models (replace with your API keys) planner_model = ChatOpenAI(model="gpt-4o-mini", temperature=0) executor_model = ChatOpenAI(model="gpt-4o-mini", temperature=0) researcher_model = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Define agent prompts PLANNER_PROMPT = ChatPromptTemplate.from_messages([ SystemMessage(content="""You are a task planner. Break down complex tasks into executable subtasks. Assign each subtask to the appropriate specialized agent. Available agents: - researcher: For web research, data gathering - executor: For code execution, file operations - analyst: For data analysis, report generation Return a JSON object with: { "subtasks": [ {"agent": "agent_name", "task": "task description", "dependencies": []} ] }"""), HumanMessage(content="{task}"), ]) RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([ SystemMessage(content="""You are a researcher agent. Gather information from the web and other sources to answer questions and collect data. You have access to: - Web browsing capabilities - API calls - File system access (read-only) Always cite your sources."""), HumanMessage(content="{task}"), ]) EXECUTOR_PROMPT = ChatPromptTemplate.from_messages([ SystemMessage(content="""You are an executor agent. You write and run code, create files, and perform technical tasks. You have access to: - Python code execution - Shell commands - File system (read/write) Be careful and deliberate. Always explain what you're doing."""), HumanMessage(content="{task}"), ]) # Create agent nodes def planner_node(state: AgentState): chain = PLANNER_PROMPT | planner_model response = chain.invoke({"task": state["messages"][-1].content}) # Parse and return subtasks return {"messages": [AIMessage(content=response.content)], "current_agent": "planner"} def researcher_node(state: AgentState): chain = RESEARCHER_PROMPT | researcher_model response = chain.invoke({"task": state["messages"][-1].content}) return {"messages": [AIMessage(content=response.content)], "current_agent": "researcher"} def executor_node(state: AgentState): chain = EXECUTOR_PROMPT | executor_model response = chain.invoke({"task": state["messages"][-1].content}) return {"messages": [AIMessage(content=response.content)], "current_agent": "executor"} # Build the graph workflow = StateGraph(AgentState) # Add nodes workflow.add_node("planner", planner_node) workflow.add_node("researcher", researcher_node) workflow.add_node("executor", executor_node) # Define edges (simplified - you'll add conditional logic) workflow.add_edge("planner", "researcher") workflow.add_edge("planner", "executor") workflow.add_edge("researcher", END) workflow.add_edge("executor", END) # Set entry point workflow.set_entry_point("planner") # Compile graph = workflow.compile() # Example usage if __name__ == "__main__": result = graph.invoke({ "messages": [HumanMessage(content="Create a competitive analysis for Tesla")], "current_agent": "", "task_progress": {} }) print(result) 

1.3 Add Conditional Routing

Update agents.py to include proper routing logic:

from langgraph.prebuilt import cond_map def should_research(state: AgentState) -> str: """Route to researcher if task requires information gathering""" last_message = state["messages"][-1].content.lower() if any(word in last_message for word in ["research", "find", "gather", "analyze", "compare"]): return "researcher" return "executor" def should_execute(state: AgentState) -> str: """Route to executor if task requires code or file operations""" last_message = state["messages"][-1].content.lower() if any(word in last_message for word in ["create", "build", "write", "code", "script", "file"]): return "executor" return END # Update the graph with conditional edges workflow.add_conditional_edges( "planner", lambda s: should_research(s) if "research" in s["messages"][-1].content.lower() else should_execute(s), {"researcher": "researcher", "executor": "executor"} ) workflow.add_conditional_edges( "researcher", lambda s: should_execute(s), {"executor": "executor", END: END} ) 

Option B: CrewAI (Simpler Alternative)

Why? CrewAI offers an intuitive role-based mental model that's easier to set up for business workflow automation.

1.1 Install CrewAI

pip install crewai 

1.2 Create a Crew

Create crewai_demo.py:

from crewai import Agent, Task, Crew, Process # Define agents planner = Agent( role='Task Planner', goal='Break down complex tasks into executable subtasks', backstory='You are an expert at decomposing work into manageable pieces', verbose=True ) researcher = Agent( role='Researcher', goal='Gather accurate information from web and data sources', backstory='You have access to web browsing and API tools', verbose=True ) executor = Agent( role='Executor', goal='Write and run code to complete technical tasks', backstory='You are a skilled developer who can execute code safely', verbose=True ) # Define tasks task = Task( description="Create a comprehensive competitive analysis for Tesla, including financials, market position, and technology stack", expected_output="A detailed report in markdown format with sections for each analysis area", agent=planner, async_execution=True ) # Create and run crew crew = Crew( agents=[planner, researcher, executor], tasks=[task], process=Process.sequential, verbose=2 ) result = crew.kickoff() print(result) 

🔌 Step 2: Set Up the Sandbox Environment

Option A: AIO Sandbox (All-in-One)

Why? Combines Browser, Shell, File, MCP operations, and VSCode Server in a single Docker container.

2.1 Run AIO Sandbox

# Pull and run the sandbox docker run --security-opt seccomp=unconfined --rm -it \ -e SANDBOX_API_KEY=your-secret-key \ -p 127.0.0.1:8080:8080 \ ghcr.io/agent-infra/sandbox:latest 

2.2 Use the Sandbox in Your Agents

from sandbox import Sandbox # Initialize client sandbox = Sandbox(baseURL='http://localhost:8080', apiKey='your-secret-key') # Execute shell commands result = sandbox.shell.exec(command='ls -la') print(result.output) # Read files content = sandbox.file.read(path='/home/user/data.csv') # Browser automation screenshot = sandbox.browser.screenshot() 

Option B: OpenSandbox (Enterprise-Grade)

# Install OpenSandbox git clone https://github.com/opensandbox-group/OpenSandbox.git cd OpenSandbox pip install -e . # Run a sandbox docker run -d --name my-sandbox opensandbox/sandbox:latest 

🚪 Step 3: Deploy the API Gateway (LiteLLM)

3.1 Install and Configure LiteLLM

# Install LiteLLM pip install litellm # Create config file (config.yaml) cat > config.yaml << 'EOF' model_list: - model_name: gpt-4o litellm_params: model: openai/gpt-4o api_key: os.environ/OPENAI_API_KEY - model_name: claude-3-5-sonnet litellm_params: model: anthropic/claude-3-5-sonnet-20241022 api_key: os.environ/ANTHROPIC_API_KEY - model_name: local-llama litellm_params: model: ollama/llama3.2 api_base: http://localhost:11434 # Environment variables litellm_settings: drop_params: true set_verbose: true EOF 

3.2 Run the LiteLLM Proxy

# Run the proxy server litellm --config config.yaml --port 4000 # Or with Docker docker run -d \ -p 4000:4000 \ -v /path/to/config.yaml:/app/config.yaml \ -e OPENAI_API_KEY=your-key \ -e ANTHROPIC_API_KEY=your-key \ ghcr.io/berriai/litellm:main 

3.3 Configure Your Agents to Use the Gateway

from langchain_openai import ChatOpenAI # Point to your LiteLLM proxy model = ChatOpenAI( model="gpt-4o", api_key="anything", # LiteLLM ignores this base_url="http://localhost:4000" ) 

📊 Step 4: Build the Interactive Dashboard

Option A: Langfuse (Production Observability)

Why? Open source alternative to LangSmith with full dashboard capabilities.

4.1 Deploy Langfuse

# Clone and run Langfuse git clone https://github.com/langfuse/langfuse.git cd langfuse # Use Docker Compose docker compose up -d # Access at http://localhost:3000 

4.2 Instrument Your Agents

from langfuse import Langfuse from langfuse.langchain import CallbackHandler # Initialize Langfuse langfuse = Langfuse( public_key="your-public-key", secret_key="your-secret-key", host="http://localhost:3000" ) # Add callback to your LLM handler = CallbackHandler(langfuse) model = ChatOpenAI( model="gpt-4o", callbacks=[handler] ) 

Option B: Custom Streamlit Dashboard

Why? More control over the UI and features.

4.1 Install Streamlit

pip install streamlit 

4.2 Create Dashboard App

Create dashboard.py:

import streamlit as st from langgraph.graph import StateGraph from datetime import datetime import json # Initialize session state if "agent_graph" not in st.session_state: st.session_state.agent_graph = None if "conversation_history" not in st.session_state: st.session_state.conversation_history = [] if "agent_states" not in st.session_state: st.session_state.agent_states = {} st.title("🤖 Manus-OSS Agent Dashboard") # Sidebar with st.sidebar: st.header("Configuration") # Model selection model_choice = st.selectbox( "Select Model", ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "local-llama"] ) # Agent selection st.subheader("Active Agents") active_agents = st.multiselect( "Enable Agents", ["Planner", "Researcher", "Executor", "Analyst"], default=["Planner", "Researcher", "Executor"] ) # Sandbox status st.subheader("Sandbox") sandbox_status = st.radio("Sandbox Mode", ["Enabled", "Disabled"]) # Main chat interface st.header("Agent Conversation") # Display conversation for i, msg in enumerate(st.session_state.conversation_history): with st.chat_message(msg["role"]): st.markdown(msg["content"]) if "agent" in msg: st.caption(f"Agent: {msg['agent']}") if "timestamp" in msg: st.caption(f"{msg['timestamp']}") # Input if prompt := st.chat_input("What would you like the agents to do?"): # Add user message st.session_state.conversation_history.append({ "role": "user", "content": prompt, "timestamp": datetime.now().strftime("%H:%M:%S") }) # Display user message with st.chat_message("user"): st.markdown(prompt) # Process with agents with st.spinner("Agents working..."): # Here you would call your agent graph # For demo, we'll simulate import time time.sleep(2) response = f"Agents have received your task: {prompt}. Processing with {len(active_agents)} active agents..." st.session_state.conversation_history.append({ "role": "assistant", "content": response, "agent": "Planner", "timestamp": datetime.now().strftime("%H:%M:%S") }) # Display response with st.chat_message("assistant"): st.markdown(response) st.caption("Agent: Planner") # Agent state visualization st.header("📈 Agent State") col1, col2, col3 = st.columns(3) with col1: st.metric("Tasks Completed", "42") with col2: st.metric("Active Agents", len(active_agents)) with col3: st.metric("Total Tokens", "12,345") # Recent activity st.subheader("Recent Activity") activity_df = st.dataframe({ "Time": ["10:30:00", "10:25:00", "10:20:00"], "Agent": ["Researcher", "Executor", "Planner"], "Action": ["Web search", "Code execution", "Task planning"], "Status": ["✅ Complete", "✅ Complete", "✅ Complete"] }) # Run the app if __name__ == "__main__": st.write("Run with: streamlit run dashboard.py") 

4.3 Run the Dashboard

streamlit run dashboard.py 

🔗 Step 5: Connect Everything Together

5.1 Final Architecture Integration

""" Complete Manus-OSS Integration Example """ from langchain_openai import ChatOpenAI from langgraph.graph import StateGraph from sandbox import Sandbox from langfuse import Langfuse from langfuse.langchain import CallbackHandler # Initialize all components # 1. LiteLLM Gateway Client llm = ChatOpenAI( model="gpt-4o", base_url="http://localhost:4000", # LiteLLM proxy api_key="anything" ) # 2. Sandbox sandbox = Sandbox( baseURL='http://localhost:8080', apiKey='your-sandbox-key' ) # 3. Observability langfuse = Langfuse( public_key="your-public-key", secret_key="your-secret-key", host="http://localhost:3000" ) callback = CallbackHandler(langfuse) # 4. Agent with all integrations def executor_node(state): # Use sandbox for safe execution code = state["messages"][-1].content # Execute in sandbox result = sandbox.shell.exec(command=f"python -c '{code}'") return { "messages": [AIMessage(content=result.output)], "current_agent": "executor" } # Build and run your graph # ... (use the patterns from Step 1) 

📁 Project Structure

manus-oss/ ├── agents/ │ ├── __init__.py │ ├── planner.py │ ├── researcher.py │ ├── executor.py │ └── graph.py ├── sandbox/ │ ├── config.yaml │ └── Dockerfile ├── gateway/ │ ├── config.yaml │ └── docker-compose.yml ├── dashboard/ │ ├── streamlit_app.py │ └── requirements.txt ├── memory/ │ └── checkpoint.db ├── config/ │ └── settings.yaml ├── requirements.txt ├── docker-compose.yml └── README.md 

🚀 Step 6: Deployment Options

Option A: Docker Compose (Recommended for Development)

Create docker-compose.yml:

version: '3.8' services: # LiteLLM Gateway gateway: image: ghcr.io/berriai/litellm:main ports: - "4000:4000" volumes: - ./gateway/config.yaml:/app/config.yaml environment: - OPENAI_API_KEY=${OPENAI_API_KEY} - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} restart: unless-stopped # AIO Sandbox sandbox: image: ghcr.io/agent-infra/sandbox:latest ports: - "8080:8080" environment: - SANDBOX_API_KEY=${SANDBOX_API_KEY} security_opt: - seccomp:unconfined restart: unless-stopped # Langfuse langfuse: image: langfuse/langfuse:latest ports: - "3000:3000" environment: - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/postgres - NEXTAUTH_SECRET=${NEXTAUTH_SECRET} - NEXTAUTH_URL=http://localhost:3000 depends_on: - postgres restart: unless-stopped postgres: image: postgres:15 environment: - POSTGRES_PASSWORD=postgres volumes: - postgres_data:/var/lib/postgresql/data ports: - "5432:5432" restart: unless-stopped # Agent Service agents: build: . ports: - "8000:8000" environment: - LITELLM_BASE_URL=http://gateway:4000 - SANDBOX_BASE_URL=http://sandbox:8080 - LANGFUSE_BASE_URL=http://langfuse:3000 depends_on: - gateway - sandbox - langfuse restart: unless-stopped # Dashboard dashboard: build: ./dashboard ports: - "8501:8501" environment: - AGENT_API_URL=http://agents:8000 depends_on: - agents restart: unless-stopped volumes: postgres_data: 

Option B: Kubernetes (Production)

# deployment.yaml apiVersion: apps/v1 kind: Deployment metadata: name: manus-agents spec: replicas: 3 selector: matchLabels: app: manus-agents template: metadata: labels: app: manus-agents spec: containers: - name: agents image: your-registry/manus-agents:latest ports: - containerPort: 8000 env: - name: LITELLM_BASE_URL value: "http://litellm-gateway:4000" - name: SANDBOX_BASE_URL value: "http://aio-sandbox:8080" resources: limits: memory: "2Gi" cpu: "1" --- apiVersion: v1 kind: Service metadata: name: manus-agents spec: selector: app: manus-agents ports: - port: 8000 targetPort: 8000 

🔧 Step 7: Advanced Features

7.1 Add File-Based Memory (Like Manus)

import json import os from pathlib import Path class FileMemory: def __init__(self, memory_dir="./memory"): self.memory_dir = Path(memory_dir) self.memory_dir.mkdir(exist_ok=True) def save_task_state(self, task_id: str, state: dict): """Save task state to file""" file_path = self.memory_dir / f"{task_id}.json" with open(file_path, 'w') as f: json.dump(state, f, indent=2) def load_task_state(self, task_id: str) -> dict: """Load task state from file""" file_path = self.memory_dir / f"{task_id}.json" if file_path.exists(): with open(file_path, 'r') as f: return json.load(f) return {} def save_skill(self, skill_name: str, skill_content: str): """Save a skill definition (like Manus SKILL.md)""" file_path = self.memory_dir / "skills" / f"{skill_name}.md" file_path.parent.mkdir(exist_ok=True) with open(file_path, 'w') as f: f.write(skill_content) def load_skill(self, skill_name: str) -> str: """Load a skill definition""" file_path = self.memory_dir / "skills" / f"{skill_name}.md" if file_path.exists(): return file_path.read_text() return "" 

7.2 Add MCP (Model Context Protocol) Support

from mcp import ClientSession import asyncio async def connect_to_mcp_server(server_url: str): """Connect to an MCP server""" session = ClientSession(server_url) await session.connect() return session # Example: Connect to a file system MCP server async def list_files_via_mcp(mcp_client): tools = await mcp_client.list_tools() for tool in tools: if tool.name == "list_files": result = await mcp_client.call_tool(tool.name, {"path": "."}) return result 

7.3 Multi-Model Dynamic Invocation

from langchain_openai import ChatOpenAI from langchain_anthropic import ChatAnthropic from langchain_community.chat_models import ChatOllama def get_model(model_name: str, temperature: float = 0.7): """Dynamically select model based on task requirements""" # Complex reasoning tasks if "analyze" in model_name.lower() or "reason" in model_name.lower(): return ChatAnthropic( model="claude-3-5-sonnet-20241022", temperature=temperature ) # Coding tasks elif "code" in model_name.lower() or "python" in model_name.lower(): return ChatOpenAI( model="gpt-4o", temperature=temperature ) # General tasks else: return ChatOllama( model="llama3.2", temperature=temperature ) 

📊 Comparison: Manus vs. This Open Source Stack

FeatureManusOpen Source AlternativeNotesMulti-Agent Orchestration✅LangGraph / CrewAIProduction-ready, statefulSandbox Environment✅AIO Sandbox / OpenSandboxFull isolation, multiple interfacesAPI Gateway✅LiteLLM100+ providers, unified APIDashboard✅Langfuse / StreamlitObservability + custom UIFile-Based Memory✅Custom implementationJSON/YAML filesMCP Support✅MCP Python SDKGrowing ecosystemMulti-Model Invocation✅Dynamic model selectionFlexible routingWeb Browsing✅PlaywrightFull browser automationCode Execution✅Sandbox + PythonSafe executionSKILL.md Files✅Custom implementationDefine agent capabilitiesCost Tracking✅LiteLLMPer-model cost trackingRate Limiting✅LiteLLMConfigurable limitsLoad Balancing✅LiteLLMMultiple provider support 

🎯 Quick Start: Minimal Viable Manus-OSS

If you want the simplest possible version that works:

1. Install Everything

# Clone this repo (or create your own) git clone https://github.com/your-org/manus-oss.git cd manus-oss # Install dependencies pip install langchain langgraph litellm streamlit playwright playwright install # Set API keys export OPENAI_API_KEY=your-key 

2. Create Minimal Files

minimal_agents.py:

from langchain_openai import ChatOpenAI from langgraph.graph import StateGraph from langgraph.prebuilt import ToolNode from typing import TypedDict class State(TypedDict): messages: list llm = ChatOpenAI(model="gpt-4o-mini", base_url="http://localhost:4000") def agent_node(state): return {"messages": [llm.invoke(state["messages"])]} workflow = StateGraph(State) workflow.add_node("agent", agent_node) workflow.set_entry_point("agent") graph = workflow.compile() # Test result = graph.invoke({"messages": ["Hello, analyze Tesla stock"]}) print(result) 

docker-compose-minimal.yml:

version: '3' services: gateway: image: ghcr.io/berriai/litellm:main ports: - "4000:4000" environment: - OPENAI_API_KEY=${OPENAI_API_KEY} 

3. Run It

# Start gateway docker compose -f docker-compose-minimal.yml up -d # Run agents python minimal_agents.py 

📚 Resources & Community

Key Repositories

LangChain: https://github.com/langchain-ai/langchain

LangGraph: https://github.com/langchain-ai/langgraph

LiteLLM: https://github.com/BerriAI/litellm

Langfuse: https://github.com/langfuse/langfuse

AIO Sandbox: https://github.com/agent-infra/sandbox

OpenSandbox: https://github.com/opensandbox-group/OpenSandbox

CrewAI: https://github.com/joaomdmoura/crewAI

Learning Resources

LangChain Documentation

LangGraph Tutorials

LiteLLM Docs

AI Agent Framework Comparison

Community

LangChain Discord: https://discord.gg/langchain

LiteLLM Discord: https://discord.gg/berriai

CrewAI Discord: https://discord.gg/crewai

🔮 Next Steps

Start Small: Begin with a single agent and LiteLLM gateway

Add Complexity: Introduce multi-agent orchestration with LangGraph

Enhance Safety: Deploy AIO Sandbox for secure execution

Monitor: Set up Langfuse for observability

Scale: Deploy to Kubernetes for production

Customize: Add your own tools, skills, and agent types

💡 Pro Tips

Start with CrewAI if you want the easiest path to multi-agent systems

Use LangGraph when you need precise control over agent workflows

LiteLLM is mandatory for multi-provider support and cost tracking

Always use a sandbox for code execution and file operations

Monitor everything - observability is key for debugging complex agent systems

Implement checkpoints - save agent state regularly for resilience

Rate limit your models - prevent API abuse and control costs

Use virtual keys in LiteLLM for multi-tenant scenarios

🎉 Conclusion

You now have a complete blueprint for building your own Manus-like autonomous agent platform using open source software. The combination of LangGraph + LiteLLM + AIO Sandbox + Langfuse gives you a production-ready system with all the key features of Manus:

✅ Multi-agent orchestration

✅ Safe sandboxed execution

✅ Unified API gateway

✅ Interactive dashboard

✅ File-based memory

✅ Multi-model support

✅ Extensible architecture

Total Cost: $0 (all components are open source)
Time to First Agent: ~30 minutes
Time to Production: ~1-2 weeks (depending on complexity)

Built with ❤️ for the open source AI community
~♪BUꉆ https://BunRec.com/
Last updated: July 2026

