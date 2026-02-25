from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv(".env.local")

@tool
def add_numbers(numbers: list[float]) -> str:
    """Add at least two numbers from a list and return the result as a string."""
    
    if len(numbers) != 2:
        return "Error: Please provide exactly two numbers."

    print(f"[tool] add_numbers(a={numbers[0]}, b={numbers[1]})")

    result = sum(numbers)

    return str(int(result)) if float(result).is_integer() else str(result)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
# llm = ChatOllama(model="qwen2.5:3b", temperature=0)

systemPrompt = SystemMessage(
    content=(
        "You are a helpful CLI assistant.\n"
        "Tool rule: add_numbers takes ONE parameter numbers=[...].\n"
        "When calling the tool:\n"
        "- Extract ALL numbers from the user message.\n"
        "- If there are exactly 2 numbers, call add_numbers with both.\n"
        "- If there are more than 2 numbers, still include ALL numbers in the list "
        "(the tool will return an error).\n"
        "Never omit numbers.\n"
        "For non-math questions, answer normally."
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
