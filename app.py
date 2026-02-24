from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

@tool
def add_numbers(a: float, b: float) -> str:
    """Add two numbers and return the result as a string."""
    print(f"[tool] add_numbers(a={a}, b={b})")
    return str(a + b)

# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
llm = ChatOllama(model="qwen2.5:3b", temperature=0)

systemPrompt = SystemMessage(
    content="You are a helpful CLI assistant. "
    "If the user asks any calculation question, "
    "you MUST call the add_numbers tool. "
    "For non-math questions, answer normally."
)

agent = create_agent(
    model=llm,
    tools=[add_numbers],
    system_prompt=systemPrompt
)

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
