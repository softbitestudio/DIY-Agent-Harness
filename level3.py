import asyncio
import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# FIX 1: Use add_messages reducer so agent history appends instead of overwriting
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

async def executor_node(state: AgentState):
    # Standard node execution hookable to gVisor API
    code = state["messages"][-1].content
    # Simulated execution response from gVisor container runtime
    output = f"Executed safely in gVisor sandbox: {code}"
    return {"messages": [AIMessage(content=output)]}

workflow = StateGraph(AgentState)
workflow.add_node("executor", executor_node)
workflow.set_entry_point("executor")
workflow.add_edge("executor", END)

async def main():
    # FIX 2: Ensure the directory exists before SQLite tries to write to it
    os.makedirs("memory", exist_ok=True)
    
    async with AsyncSqliteSaver.from_conn_string("memory/checkpoints.db") as checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "session-1"}}
        
        inputs = {"messages": [HumanMessage(content="print('Hello from isolated sandbox')")]}
        async for event in app.astream(inputs, config=config):
            print("Checkpoint State Update:", event)

if __name__ == "__main__":
    asyncio.run(main())
