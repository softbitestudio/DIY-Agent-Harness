from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Initialize lightweight model
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
    print(f"Plan:\n{plan}")
    
    result = executor_agent(plan)
    print(f"Execution Result:\n{result}")
