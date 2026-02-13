# Build Your Own Coding Agent: Complete Learning Roadmap

**Timeline:** 4–6 weeks (10–15 hrs/week)
**Hardware:** Desktop with 6GB VRAM
**Cost:** $0
**Prerequisites:** Python fundamentals, basic CLI comfort

---

## Phase 0: Environment Setup (Day 1)

**Time:** 2–3 hours
**Goal:** Get a local LLM running and talking to your code.

### Steps

1. Install [Ollama](https://ollama.com) for your OS.
2. Pull a model that fits your VRAM:
   - Start with: `ollama pull qwen2.5-coder:7b-instruct-q4_K_M`
   - Backup option: `ollama pull deepseek-coder-v2:lite`
   - If those feel too slow or dumb, try `ollama pull qwen2.5-coder:3b` for faster iteration while prototyping (worse quality, but snappier feedback loop)
3. Test it works: `ollama run qwen2.5-coder:7b-instruct-q4_K_M "Write a Python function that reverses a string"`
4. Test the API works from Python:

```python
import requests

response = requests.post("http://localhost:11434/api/chat", json={
    "model": "qwen2.5-coder:7b-instruct-q4_K_M",
    "messages": [{"role": "user", "content": "Write hello world in Python"}],
    "stream": False
})

print(response.json()["message"]["content"])
```

5. Set up your project folder:

```
coding-agent/
├── agent.py          # main agent loop
├── tools.py          # tool definitions (run code, read file, etc.)
├── prompts.py        # system prompts and prompt templates
├── sandbox/          # where the agent writes and runs code
└── README.md
```

### What you're learning
- How local LLMs work (model quantization, VRAM constraints, inference speed)
- The Ollama API (it mirrors the OpenAI chat completions format, which is industry standard)
- Why model size matters — you'll immediately feel the difference between 3B and 7B

### Resources
- [Ollama docs](https://github.com/ollama/ollama)
- [Ollama API reference](https://github.com/ollama/ollama/blob/main/docs/api.md)

---

## Phase 1: The Dumb Loop (Week 1)

**Time:** 8–10 hours
**Goal:** Build the simplest possible agent that can write and run code.

### What you're building

A loop that does exactly this:
1. You give it a task in plain English
2. It writes Python code
3. Your program runs that code in a subprocess
4. If there's an error, it sends the error back to the model
5. The model tries to fix it
6. Repeat until it works or you hit a retry limit

### Steps

**Step 1: Build the code execution tool (2 hrs)**

Write a function that takes a string of Python code, writes it to a file in `sandbox/`, runs it via `subprocess.run()`, and returns stdout + stderr. This is the agent's only "tool" for now.

Key decisions you'll make:
- Timeout (start with 10 seconds)
- How to capture output (subprocess.PIPE)
- How to handle the sandbox (just a folder — don't overthink security yet)

```python
import subprocess
import tempfile
import os

def run_code(code: str, timeout: int = 10) -> dict:
    """Run Python code and return stdout/stderr."""
    filepath = os.path.join("sandbox", "task.py")
    with open(filepath, "w") as f:
        f.write(code)
    try:
        result = subprocess.run(
            ["python3", filepath],
            capture_output=True, text=True, timeout=timeout
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "TIMEOUT", "returncode": -1}
```

**Step 2: Build the prompt template (2 hrs)**

This is where you learn prompt engineering for structured output. You need the model to return code in a parseable format. The simplest approach:

```
You are a coding assistant. When given a task, respond with ONLY a Python
code block. Do not explain. Do not add commentary.

If you receive an error, fix the code and respond with the corrected
version.

Task: {task}
```

You'll quickly discover this doesn't work perfectly with a 7B model. The model will add explanations, forget the format, or hallucinate. This is good — it forces you to iterate on your prompts and build parsing logic.

**Step 3: Build the agent loop (3 hrs)**

```python
def agent_loop(task: str, max_retries: int = 5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task}
    ]

    for attempt in range(max_retries):
        # Get model response
        response = call_ollama(messages)

        # Parse code from response
        code = extract_code(response)

        if code is None:
            # Model didn't return parseable code — ask it to try again
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Please respond with only a Python code block."})
            continue

        # Run the code
        result = run_code(code)

        if result["returncode"] == 0:
            print(f"SUCCESS on attempt {attempt + 1}")
            print(result["stdout"])
            return code

        # Failed — send error back
        error_msg = f"Error:\n{result['stderr']}\n\nFix the code."
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": error_msg})

    print("FAILED after max retries")
    return None
```

**Step 4: Test with simple tasks (2 hrs)**

Start with trivially easy things and increase difficulty:
1. "Print the numbers 1 to 10"
2. "Write a function that checks if a number is prime, test it on 17 and 20"
3. "Read a CSV file called data.csv and print the average of the 'price' column" (create a test CSV first)
4. "Fetch the top 5 Hacker News stories using the API" (this will fail without requests installed — good!)

Track what works, what fails, and why. Write it down.

### What you're learning
- The **core agent loop**: observe → decide → act → evaluate. Every agent ever built is a variation of this.
- **Prompt engineering for structured output**: getting a small model to reliably return parseable responses is genuinely hard and directly transferable.
- **Error recovery**: the retry-with-feedback pattern is universal across all agents.
- **Where small models break**: you'll develop intuition for what 7B can and can't do.

### Definition of done
Your agent can solve at least 3 out of the 4 test tasks above without your intervention (besides starting it).

---

## Phase 2: Tool Use (Week 2)

**Time:** 8–10 hours
**Goal:** Give the agent multiple tools and teach it to choose between them.

### Why this matters

Right now your agent can only "write and run Python." Real agents have multiple tools — read files, write files, search, execute shell commands, etc. The hard part isn't adding tools; it's getting the model to **choose the right tool** and **use it correctly**.

### Steps

**Step 1: Define a tool interface (2 hrs)**

Create a standard way to define tools so the model knows what's available:

```python
TOOLS = {
    "run_python": {
        "description": "Execute Python code and return output",
        "parameters": {"code": "string"},
        "function": run_code
    },
    "read_file": {
        "description": "Read the contents of a file",
        "parameters": {"filepath": "string"},
        "function": read_file
    },
    "write_file": {
        "description": "Write content to a file",
        "parameters": {"filepath": "string", "content": "string"},
        "function": write_file
    },
    "list_files": {
        "description": "List files in a directory",
        "parameters": {"directory": "string"},
        "function": list_files
    },
    "run_shell": {
        "description": "Run a shell command and return output",
        "parameters": {"command": "string"},
        "function": run_shell
    }
}
```

**Step 2: Update your prompt to present tools (2 hrs)**

Now the model needs to output structured "actions" instead of just code. This is the part where you learn **function calling / tool use** — the core pattern behind every agent framework.

You'll need the model to output something like:

```json
{"tool": "read_file", "parameters": {"filepath": "src/main.py"}}
```

Getting a 7B model to do this reliably is a real challenge. You'll experiment with:
- JSON output format vs XML vs custom format
- Few-shot examples in the prompt
- Parsing with fallbacks (regex, json.loads with error handling)

This is one of the most transferable skills you'll build.

**Step 3: Update the agent loop for multi-tool use (3 hrs)**

The loop now becomes:
1. Send task + available tools to model
2. Model chooses a tool and parameters
3. You execute the tool and return the result
4. Model decides: use another tool, or give a final answer
5. Repeat

The key addition is a **"think" step** — before choosing a tool, the model should reason about what it needs to do. This is basic chain-of-thought prompting.

**Step 4: Test with multi-step tasks (2 hrs)**

- "List the files in the sandbox directory, then read any Python files you find and summarize what they do"
- "Write a Python script that generates 100 random numbers, save it to sandbox/random.py, run it, and tell me the average"
- "Read sandbox/broken.py, find the bug, fix it, write the fixed version, and run it" (create a buggy file first)

### What you're learning
- **Tool use / function calling**: this is the #1 pattern in agent development. Every framework (LangChain, CrewAI, Anthropic's tool use API) is built on this.
- **Structured output parsing**: getting models to return machine-readable actions.
- **Multi-step planning**: the model has to sequence multiple tool calls to accomplish a goal.
- **The "action-observation" pattern**: the model acts, sees the result, then decides what to do next. This is the foundation of ReAct (Reasoning + Acting), the most influential agent paper.

### Definition of done
Your agent can complete the "read, fix, and run broken code" task autonomously.

---

## Phase 3: Context Management (Week 3)

**Time:** 8–10 hours
**Goal:** Handle the context window intelligently — the hardest real-world agent problem.

### Why this matters

Your 7B model has a limited context window (typically 4K–32K tokens depending on the model). Every message in the conversation eats into that limit. In a real agent run, you might have 10+ tool calls, each returning output. You'll blow through the context window fast. This is the problem that separates toy agents from real ones.

### Steps

**Step 1: Add token counting (2 hrs)**

Install `tiktoken` or write a rough word-based estimator. Start tracking how many tokens each conversation uses. Log it. You'll be surprised how fast it grows.

**Step 2: Build a context manager (3 hrs)**

Implement strategies for staying within the window:
- **Truncation**: if a tool output is too long (e.g., a huge file), truncate it and tell the model it was truncated.
- **Summarization**: after N tool calls, summarize the conversation so far and start a fresh context with the summary.
- **Selective history**: keep the system prompt + last N messages, drop the middle.
- **Smart file reading**: instead of reading entire files, let the model read specific line ranges.

```python
def manage_context(messages, max_tokens=3000):
    """Keep conversation within token budget."""
    total = count_tokens(messages)
    if total <= max_tokens:
        return messages

    # Strategy: keep system prompt + first user message + last N messages
    system = messages[0]
    first_user = messages[1]
    # Keep trimming from the middle until we fit
    recent = messages[2:]
    while count_tokens([system, first_user] + recent) > max_tokens and len(recent) > 2:
        recent.pop(0)

    return [system, first_user] + recent
```

**Step 3: Add a "grep" / "search" tool (2 hrs)**

Instead of reading entire files, give the agent a search tool that returns only matching lines with context. This is how real coding agents handle large codebases without blowing up the context.

```python
def search_files(pattern: str, directory: str = "sandbox") -> str:
    """Search for a pattern in files. Returns matching lines with context."""
    result = subprocess.run(
        ["grep", "-rn", "--include=*.py", "-C", "3", pattern, directory],
        capture_output=True, text=True
    )
    return result.stdout[:2000]  # Truncate long results
```

**Step 4: Test with larger tasks (2 hrs)**

Create a small multi-file project in sandbox/ (3-4 files, a few hundred lines total) and give the agent tasks that require navigating across files:
- "Find all functions that take a 'user_id' parameter and add input validation"
- "This project has a bug where the /users endpoint returns a 500 error. Find and fix it."

### What you're learning
- **Context window management**: the single most practical skill for building real agents. Every production agent needs this.
- **Retrieval-augmented generation (RAG) at a basic level**: searching for relevant info instead of dumping everything in context.
- **The cost of information**: every token you spend on context is a token the model can't use for reasoning. Learning to be economical with context is critical.

### Definition of done
Your agent can work with a multi-file project (3+ files) without hitting context limits or losing track of what it's doing.

---

## Phase 4: Planning and Self-Evaluation (Week 4)

**Time:** 8–10 hours
**Goal:** Give the agent the ability to plan before acting and evaluate its own work.

### Steps

**Step 1: Add a planning step (3 hrs)**

Before the agent starts using tools, have it output a plan:

```
Given this task, outline your approach in 3-5 steps before taking action.
```

Then have it execute the plan step by step, checking off steps as it goes. This is a simplified version of what frameworks call "task decomposition."

Key insight: the plan often needs to change mid-execution. Build in a "replan" mechanism where the agent can update its plan after encountering unexpected results.

**Step 2: Add test running as a tool (2 hrs)**

Give the agent the ability to run tests (pytest). This is how it evaluates its own work:
1. Agent writes code
2. Agent writes tests (or runs existing ones)
3. Tests pass → done
4. Tests fail → agent reads failures and iterates

This is the **self-evaluation loop** — the agent has a way to check its own work without needing you.

```python
def run_tests(test_file: str = "sandbox/test_task.py") -> dict:
    result = subprocess.run(
        ["python3", "-m", "pytest", test_file, "-v", "--tb=short"],
        capture_output=True, text=True, timeout=30
    )
    return {"output": result.stdout + result.stderr, "passed": result.returncode == 0}
```

**Step 3: Add a "done" action (1 hr)**

The agent needs to explicitly decide when it's finished vs. when it should keep going. Add a "finish" tool:

```python
"finish": {
    "description": "Declare the task complete. Use only when tests pass or you've verified the output is correct.",
    "parameters": {"summary": "string"}
}
```

This teaches you about **stopping conditions** — a surprisingly hard problem. The agent might think it's done when it's not, or keep going in circles.

**Step 4: Integration test (3 hrs)**

Give it a complete task from scratch:
- "Create a Python module called `calculator.py` with functions for add, subtract, multiply, divide (handle division by zero). Then create `test_calculator.py` with pytest tests covering normal cases and edge cases. All tests should pass."

Watch it plan, execute, test, fix, and finish. Debug where it gets stuck.

### What you're learning
- **Task decomposition**: breaking complex tasks into steps. This is what makes agents handle real-world complexity.
- **Self-evaluation**: the agent checks its own work. Without this, agents just confidently produce garbage.
- **Stopping conditions**: knowing when to stop is non-trivial and critical for production agents.
- **The plan-execute-replan loop**: this is the architecture behind tools like Devin, Claude Code, and similar.

### Definition of done
Your agent can take the calculator task above and produce working, tested code without intervention.

---

## Phase 5: Polish and Reflect (Week 5-6)

**Time:** 6–10 hours
**Goal:** Clean up, benchmark, and extract the reusable framework.

### Steps

**Step 1: Build a benchmark (3 hrs)**

Create 10-15 coding tasks of varying difficulty:
- 5 easy (single-function, <20 lines)
- 5 medium (multi-function, file I/O, basic algorithms)
- 3-5 hard (multi-file, debugging, refactoring)

Run your agent on all of them. Track: success rate, average attempts, average time, common failure modes.

This gives you concrete data on what works and what doesn't — way more valuable than vibes.

**Step 2: Refactor into a reusable framework (3 hrs)**

Separate the "agent engine" from the "coding tools":

```
agent/
├── core.py        # The agent loop, context manager, planner
├── tools.py       # Abstract tool interface
├── prompts.py     # Prompt templates
└── config.py      # Model settings, retry limits, etc.

tools/
├── coding.py      # run_python, run_tests, etc.
├── filesystem.py  # read, write, list, search
└── shell.py       # run_shell
```

Now if you ever find that legitimate use case you were looking for, you can plug in new tools without rewriting the core.

**Step 3: Write up what you learned (2 hrs)**

Not for anyone else — for yourself. Answer these questions:
1. What's the actual bottleneck: the model's intelligence or my architecture?
2. Where did the 7B model surprise me? Where did it consistently fail?
3. What would change if I swapped in a more powerful model (Claude, GPT-4)?
4. What problems did I hit that no tutorial warned me about?
5. If I had to build an agent for [X domain], what would I reuse and what would change?

This reflection is where the learning solidifies.

### What you're learning
- **Evaluation**: how to measure agent performance objectively. This is critical in production.
- **Abstraction**: separating the agent pattern from the domain-specific tools.
- **The gap between local and frontier models**: understanding what model capability buys you vs. what architecture buys you.

---

## Phase 6: Multi-Agent Orchestration + Semantic RAG (Week 7-9)

⚠️ **DO NOT START THIS UNTIL PHASES 1-5 ARE COMPLETE AND YOUR SINGLE AGENT WORKS.**
If you skip ahead, you will waste your time. Multi-agent adds coordination complexity on top of single-agent complexity. If your single agent is broken, multiple broken agents coordinating is just chaos.

**Time:** 15–20 hours
**Goal:** Split your single agent into specialized agents that coordinate, and replace keyword search with semantic retrieval.

---

### Part A: Multi-Agent Orchestration (10-12 hrs)

#### Why now and not earlier

By this point, you've hit the limitations of a single agent. You've probably noticed things like:
- The agent's plan is great but its code is sloppy (or vice versa)
- It forgets the plan halfway through execution
- It writes code that "works" but doesn't match the original intent
- The context window is getting crushed trying to hold planning, coding, and reviewing all at once

These are the exact problems multi-agent solves. Each agent gets a focused role with a smaller context window dedicated to that role. You're not adding multi-agent because it's cool — you're adding it because you've felt the pain of not having it.

#### The Architecture

You're building four agents from your existing single agent:

```
┌─────────────────────────────────┐
│          ORCHESTRATOR           │
│  (routes tasks, tracks state)   │
└──────┬──────┬──────┬───────────┘
       │      │      │
       ▼      ▼      ▼
   ┌──────┐┌──────┐┌──────┐
   │PLANNER││CODER ││REVIEWER│
   └──────┘└──────┘└──────┘
```

- **Orchestrator**: receives the original task, decides which agent to call next, passes context between them, and decides when the task is complete.
- **Planner**: takes a task and produces a step-by-step plan. Doesn't write code. Doesn't review. Just plans.
- **Coder**: takes a single, specific subtask from the plan and writes/edits code. Has access to file tools and code execution. Doesn't plan or review.
- **Reviewer**: takes the coder's output and evaluates it. Runs tests, checks if it matches the plan, and reports back with pass/fail and specific feedback.

#### Step 1: Define agent roles as separate prompt configs (2 hrs)

Each agent is just your existing agent loop with a different system prompt and a different set of tools:

```python
AGENTS = {
    "planner": {
        "system_prompt": """You are a planning specialist. Given a coding task,
        break it into clear, specific subtasks. Output ONLY a numbered plan.
        Do not write code. Do not review code. Just plan.""",
        "tools": []  # Planner has no tools — it only thinks
    },
    "coder": {
        "system_prompt": """You are a coding specialist. You receive a specific
        subtask and implement it. Focus only on the subtask given.
        Do not plan ahead. Do not review your own work.""",
        "tools": ["run_python", "read_file", "write_file", "search_files", "run_shell"]
    },
    "reviewer": {
        "system_prompt": """You are a code reviewer. You receive code and a
        description of what it should do. Run the tests. Check if the code
        matches the requirements. Report: PASS or FAIL with specific feedback.
        Do not fix the code yourself.""",
        "tools": ["read_file", "run_tests", "search_files"]
    }
}
```

Notice what each agent *can't* do. Constraints are as important as capabilities. The planner can't code (so it won't jump ahead). The coder can't run tests (so it won't prematurely declare success). The reviewer can't edit files (so it stays objective).

#### Step 2: Build the orchestrator (4 hrs)

This is the new hard part. The orchestrator manages the flow:

```python
def orchestrate(task: str):
    # Step 1: Get a plan
    plan = call_agent("planner", f"Create a plan for: {task}")
    subtasks = parse_plan(plan)

    for i, subtask in enumerate(subtasks):
        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            # Step 2: Have the coder implement the subtask
            code_result = call_agent("coder", f"""
                Subtask {i+1}: {subtask}
                Project context: {get_project_summary()}
            """)

            # Step 3: Have the reviewer check it
            review = call_agent("reviewer", f"""
                Subtask was: {subtask}
                Review the current state of the code and run tests.
            """)

            if review_passed(review):
                break

            # Step 4: If failed, send feedback to coder for another attempt
            attempts += 1
            # Append reviewer feedback to next coder call

        if attempts == max_attempts:
            # Replan — maybe the subtask was too vague or impossible
            plan = call_agent("planner", f"""
                Original task: {task}
                Completed so far: {subtasks[:i]}
                Failed subtask: {subtask}
                Reviewer feedback: {review}
                Create a revised plan for the remaining work.
            """)
            subtasks = parse_plan(plan)

    return "Task complete"
```

Key decisions you'll make:
- **What context to pass between agents**: the orchestrator decides what each agent needs to know. This is where you learn that information routing is the real job of an orchestrator.
- **When to replan vs. retry**: if the coder fails 3 times, is the code wrong or is the plan wrong? This is a real design decision with no perfect answer.
- **State tracking**: the orchestrator needs to know what's been done, what's left, and what the current state of the codebase is.

#### Step 3: Build inter-agent communication (2 hrs)

Agents need a structured way to talk to each other. Build a simple message format:

```python
class AgentMessage:
    def __init__(self, from_agent, to_agent, content, message_type):
        self.from_agent = from_agent      # "planner", "coder", "reviewer"
        self.to_agent = to_agent
        self.content = content
        self.message_type = message_type  # "plan", "code_result", "review", "feedback"
        self.timestamp = time.time()
```

Keep a log of all inter-agent messages. This becomes your debugging tool — when something goes wrong, you can trace exactly what each agent said and where communication broke down.

#### Step 4: Test and compare (3 hrs)

Run your Phase 5 benchmark on the multi-agent version. Compare:
- Success rate: single agent vs. multi-agent
- Total time: multi-agent is slower per task (more model calls) — is the quality improvement worth it?
- Failure modes: does multi-agent fail differently? (Spoiler: yes. Coordination failures are a new category.)
- Token usage: are you actually using fewer total tokens because each agent has focused context?

Be honest with the data. Multi-agent might actually perform worse on simple tasks because the coordination overhead isn't worth it for easy problems. That's a real and important finding.

#### What you're learning
- **Agent orchestration**: routing tasks, managing state, and deciding which agent to call. This is the core skill behind frameworks like CrewAI, AutoGen, and LangGraph.
- **Separation of concerns**: why specialized agents can outperform generalist agents on complex tasks.
- **Inter-agent communication**: how agents pass information and context to each other. This is non-trivial — too much context overwhelms, too little context causes misunderstandings.
- **The coordination tax**: multi-agent isn't free. You're trading single-agent context limits for coordination complexity. Understanding this tradeoff is what makes you a good architect.
- **When NOT to use multi-agent**: if your benchmark shows single-agent wins on simple tasks, congratulations — you've learned the most important lesson about multi-agent, which is that it's not always the answer.

---

### Part B: Semantic RAG (5-8 hrs)

#### Why now

In Phase 3, you built grep-based search. That works great when you know the exact term to search for. But what if the agent needs to find "the function that handles user authentication" and it's actually called `verify_credentials`? Keyword search misses that. Semantic search catches it.

#### Step 1: Set up local embeddings (1 hr)

```bash
pip install sentence-transformers numpy
```

```python
from sentence_transformers import SentenceTransformer

# This model is ~80MB and runs fine on CPU — no GPU needed
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate an embedding
embedding = model.encode("function that authenticates users")
# Returns a 384-dimensional numpy array
```

#### Step 2: Build a code chunker (2 hrs)

You can't embed an entire file — you need to break code into meaningful chunks. The simplest approach for Python:

```python
import ast

def chunk_python_file(filepath: str) -> list[dict]:
    """Split a Python file into function/class-level chunks."""
    with open(filepath) as f:
        source = f.read()

    tree = ast.parse(source)
    chunks = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno
            chunk_text = "\n".join(source.splitlines()[start:end])
            chunks.append({
                "filepath": filepath,
                "name": node.name,
                "type": type(node).__name__,
                "start_line": node.lineno,
                "end_line": end,
                "content": chunk_text
            })

    return chunks
```

This is where you learn about **chunking strategy** — a core RAG concept. Bad chunks give bad retrieval. For code, function-level chunking is usually the right granularity. For documents, it might be paragraphs or sections. The principle is: each chunk should be a self-contained unit of meaning.

#### Step 3: Build the index (1 hr)

```python
import numpy as np

class CodeIndex:
    def __init__(self, model):
        self.model = model
        self.chunks = []
        self.embeddings = None

    def index_directory(self, directory: str):
        """Index all Python files in a directory."""
        self.chunks = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.endswith(".py"):
                    filepath = os.path.join(root, f)
                    self.chunks.extend(chunk_python_file(filepath))

        # Embed all chunks at once (batched = fast)
        texts = [c["content"] for c in self.chunks]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Find the most relevant code chunks for a query."""
        query_embedding = self.model.encode(query, normalize_embeddings=True)

        # Cosine similarity (since normalized, just dot product)
        similarities = np.dot(self.embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(similarities[idx])
            results.append(chunk)

        return results
```

No vector database needed. For codebases under 10K functions, numpy is plenty fast. You don't need Pinecone or ChromaDB yet — and using them would hide the mechanics you're trying to learn.

#### Step 4: Integrate as a tool (1 hr)

Add it alongside your existing grep tool:

```python
"semantic_search": {
    "description": "Search the codebase by meaning. Use when you need to find code related to a concept but don't know the exact function or variable names.",
    "parameters": {"query": "string", "top_k": "integer (default 5)"},
    "function": code_index.search
}
```

Now the agent has two search strategies: grep for when it knows what it's looking for, semantic search for when it knows what it *means* but not what it's called.

#### Step 5: Compare retrieval strategies (2 hrs)

Test both tools on your benchmark tasks:
- When does grep find the right code and semantic search doesn't?
- When does semantic search find the right code and grep doesn't?
- When do both fail?
- What happens when you combine them (search with both, merge results)?

This comparison is more valuable than the implementation itself. It gives you real intuition for when RAG helps and when it's overkill.

#### What you're learning
- **The full RAG pipeline**: chunking → embedding → indexing → retrieval → injection into context. This is the same pipeline whether you're searching code, documents, or knowledge bases.
- **Embedding models**: what they are, how they work, and that they run locally and cheaply.
- **Chunking strategy**: the most underrated part of RAG. Bad chunks = bad results regardless of your embedding model.
- **Retrieval tradeoffs**: keyword vs. semantic, precision vs. recall, and when to use which.
- **Why you don't always need a vector database**: for small-to-medium datasets, numpy is fine. You'll know when you actually need Pinecone vs. when it's resume-driven development.

---

### Phase 6: Definition of done

1. Your multi-agent system can solve at least 2-3 of the "hard" tasks from your Phase 5 benchmark that the single agent struggled with.
2. You have concrete benchmark data comparing single-agent vs. multi-agent performance.
3. Your semantic search finds relevant code that grep misses on at least a few test cases.
4. You can articulate — from experience, not from blog posts — when multi-agent is worth the complexity and when it isn't, and when semantic RAG beats keyword search and when it doesn't.

---

## What You'll Have at the End

1. **A working coding agent** that can autonomously write, test, and debug code
2. **A reusable agent framework** with swappable tools
3. **A multi-agent orchestration system** with planner, coder, reviewer, and orchestrator
4. **A semantic RAG pipeline** with embeddings-based code search
5. **Deep understanding of**: the agent loop, tool use, prompt engineering, context management, planning, self-evaluation, stopping conditions, agent orchestration, inter-agent communication, chunking, embeddings, and retrieval strategies
6. **Benchmark data** comparing single vs. multi-agent and keyword vs. semantic search
7. **Clear intuition** for when each technique is actually worth using vs. when it's overengineering

These are exactly the skills that transfer to building agents for any domain. The next time you spot a real use case — whether that's at a job, a side project, or a startup idea — you won't need a tutorial. You'll know the pattern.

---

## Key Resources

**Papers (skim, don't deep-read):**
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) — the foundational agent pattern you're implementing
- [Toolformer](https://arxiv.org/abs/2302.04761) — how models learn to use tools
- [SWE-bench](https://www.swebench.com/) — the benchmark for coding agents (for context on where the field is)

**Code references:**
- [Ollama Python library](https://github.com/ollama/ollama-python) — cleaner than raw HTTP if you prefer
- [SWE-agent](https://github.com/princeton-nlp/SWE-agent) — a real coding agent you can study once you've built yours (don't look at it before you build yours — you'll learn more by hitting the problems yourself first)

**Phase 6 specific:**
- [sentence-transformers docs](https://www.sbert.net/) — the library you'll use for local embeddings
- [LangGraph conceptual docs](https://langchain-ai.github.io/langgraph/) — read AFTER you build your own orchestrator, to see how a framework approaches the same problem
- [AutoGen](https://github.com/microsoft/autogen) — Microsoft's multi-agent framework, good to study after building yours
- [Building RAG from scratch (blog)](https://simonwillison.net/) — Simon Willison writes excellent practical RAG content

**When you're ready to level up:**
- Try your framework with a stronger model via a free API tier (Anthropic, OpenAI, Google all offer some free credits)
- The difference in capability will teach you exactly what model intelligence buys you vs. what good architecture buys you
