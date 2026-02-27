from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import SystemMessage
from langchain_ollama import ChatOllama

import re
from dotenv import load_dotenv

load_dotenv(".env.local")

@tool
def add_numbers(text: str) -> str:
    """Extract exactly two numbers from text and add them."""
    
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if len(nums) != 2:
        return f"Error: Please provide exactly two numbers (found {len(nums)})."

    a, b = float(nums[0]), float(nums[1])
    print(f"[tool] add_numbers(a={a}, b={b})")
    res = a + b
    return str(int(res)) if res.is_integer() else str(res)

# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
llm = ChatOllama(model="qwen2.5:3b", temperature=0)

systemPrompt = SystemMessage(
     content=(
        "You are a helpful CLI assistant.\n\n"
        "TOOL USAGE RULES:\n"
        "- For any addition or arithmetic request, you MUST call the add_numbers tool.\n"
        "- Never compute arithmetic yourself.\n"
        "- The add_numbers tool takes exactly ONE argument named 'text'.\n"
        "- Always pass the user's full message exactly as written into the 'text' parameter.\n"
        "- Do NOT extract numbers yourself.\n"
        "- Do NOT guess missing numbers.\n"
        "- If the user provides more or fewer than two numbers, the tool will return an error message. You should not handle this yourself.\n"
        "For non-math questions, respond normally without calling any tool."
    )
)

agent = create_agent(model=llm, tools=[add_numbers], system_prompt=systemPrompt)

def main():
    while True:
        user_input = input("You> ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the program. Goodbye!")
            break

        messages: list[dict] = [{"role": "user", "content": user_input}]

        result = agent.invoke({"messages": messages})
        
        print("AI>", result["messages"][-1].content)

if __name__ == "__main__":
    main()
