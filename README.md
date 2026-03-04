# Python Interactive CLI (LangChain tool-calling)

A minimal interactive CLI app that:
- Starts an interactive loop (`User>` prompt)
- Prints assistant responses as `AI>`
- Exits on `exit` or `quit`
- Uses LangChain tool-calling with a local Ollama model (`qwen2.5:3b`)
- Implements exactly one tool: `add_numbers(a, b)`
- prints a tool debug line ONLY when a tool is called (required by acceptance test)

## Requirements

- Python 3.11 or 3.12 recommended (LangChain/Pydantic can be noisy on very new Python versions)
- Ollama installed locally
- Model `qwen2.5:3b` pulled in Ollama

## Install

```bash
pip install -r requirements.txt