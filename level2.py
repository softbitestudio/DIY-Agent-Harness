import asyncio
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="Level 2: The Runner")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

async def planner_node(task: str):
    prompt = f"Break this task into 2 subtasks:\n{task}"
    response = await model.ainvoke([HumanMessage(content=prompt)])
    return response.content

async def executor_node(plan: str):
    prompt = f"Execute the first step briefly:\n{plan}"
    response = await model.ainvoke([HumanMessage(content=prompt)])
    return response.content

async def run_agent_pipeline(task: str):
    yield f"data: {json.dumps({'status': 'started', 'agent': 'Planner'})}\n\n"
    plan = await planner_node(task)
    yield f"data: {json.dumps({'status': 'planning_complete', 'plan': plan})}\n\n"
    
    yield f"data: {json.dumps({'status': 'executing', 'agent': 'Executor'})}\n\n"
    result = await executor_node(plan)
    yield f"data: {json.dumps({'status': 'completed', 'result': result})}\n\n"

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
      <input type="text" id="task" value="Summarize local AI agent trends" style="width:300px;"/>
      <button onclick="run()">Execute</button>
      <pre id="log" style="background:#222; color:#0f0; padding:15px; margin-top:10px;"></pre>
      <script>
        function run() {
          const log = document.getElementById('log');
          log.textContent = "Connecting...\n";
          const es = new EventSource(`/stream?task=${encodeURIComponent(document.getElementById('task').value)}`);
          es.onmessage = (e) => {
            const data = JSON.parse(e.data);
            log.textContent += `[${data.agent || 'System'}] ${data.status}: ${data.plan || data.result || ''}\n`;
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
