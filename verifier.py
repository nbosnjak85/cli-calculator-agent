from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv(".env.local")

verifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # može i jeftiniji model

VERIFIER_PROMPT = SystemMessage(
    content=(
        "You are a strict verifier.\n"
        "If the user doesn't ask an arithmetic question, skip verification and return YES.\n"
        "Given:\n"
        "- the original user input\n"
        "- all tool calls that were executed and their tool outputs\n"
        "- if it was an arithmetic request\n"
        "- tool used only two numbers from the user input and added them correctly\n"
        "- the assistant final answer\n\n"
        "Decide if the final answer is consistent with the tool outputs and the user request.\n"
        "Return ONLY one of these exact outputs:\n"
        "YES\n"
        "NO\n"
        "No extra text."
    )
)

def verify_final_answer(user_input: str, messages: list) -> bool:
    print("[verifier] Verifying the final answer...")
    # Collect tool outputs (what actually happened)
    tool_outputs = []
    for m in messages:
        if isinstance(m, ToolMessage):
            tool_outputs.append(str(m.content))

    # Find final assistant text
    final_ai = None
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            final_ai = m.content
            break

    # If we somehow don't have an AI answer, fail closed
    if final_ai is None:
        return False

    # Build a compact verification payload
    verification_text = (
        f"USER_INPUT:\n{user_input}\n\n"
        f"TOOL_OUTPUTS:\n{tool_outputs}\n\n"
        f"FINAL_ANSWER:\n{final_ai}\n"
    )

    verdict = verifier_llm.invoke([VERIFIER_PROMPT, HumanMessage(content=verification_text)]).content.strip()
    print(f"[verifier] Verdict: {verdict}")
    return verdict == "YES"