import os
import random

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage

# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(".env.local")

FAULT_RATE = float(os.getenv("FAULT_RATE", "0"))  # npr 0.2 = 20%

@tool
def add_numbers(numbers: list[float]) -> str:
    """Add at least two numbers from a list and return the result as a string."""

    print(f"[tool] add_numbers(a={numbers[0]}, b={numbers[1]})")

    result = sum(numbers)

    if FAULT_RATE > 0 and random.random() < FAULT_RATE:
        result += 1  # namjerno pokvari rezultat
        print("[fault] add_numbers injected +1 error")

    return str(int(result)) if float(result).is_integer() else str(result)


class ValidateInput(BaseModel):
    numbers: list[float] = Field(..., description="Numbers that were added")
    result: float = Field(..., description="Claimed sum to validate")

@tool
def validate_sum(data: ValidateInput) -> str:
    """Validate that result equals sum(numbers). Returns 'OK' or an error string."""
   
    if not data.numbers or len(data.numbers) != 2:
        return "Error: Please provide exactly two numbers."

    expected = sum(data.numbers)
    ok = abs(expected - data.result) < 1e-9

    print(f"[tool] validate_sum(numbers={data.numbers}, result={data.result})")

    if ok:
        return "OK"
    return f"Error: Validation failed. expected={expected}, got={data.result}"


llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
# llm = ChatOllama(model="qwen2.5:3b", temperature=0)

systemPrompt = SystemMessage(
    content=(
        "You are a CLI assistant with tools.\n\n"
        "IMPORTANT RULES (must follow exactly):\n"
        "1) If the user asks for addition / sum / arithmetic and provides numbers, you MUST call tools.\n"
        "2) ALWAYS do this sequence:\n"
        "   a) Call add_numbers(numbers=[...]) with ALL numbers you find in the user message.\n"
        "   b) Take the tool output (the sum) and THEN call validate_sum(data={numbers: [...], result: <sum_as_number>}).\n"
        "   c) Only if validate_sum returns 'OK', you may output the final answer to the user.\n"
        "   d) If validate_sum returns an Error, show that Error to the user and stop.\n"
        "3) Never compute sums in your head; only use tool outputs.\n"
        "4) For non-math messages, respond normally without calling tools.\n"
    )
)

agent = create_agent(
    model=llm, tools=[add_numbers, validate_sum], system_prompt=systemPrompt
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
