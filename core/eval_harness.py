import pandas as pd
from .tools import run_agent
from .llm import generate

# Replace/extend with your AI-C2 test set
CASES = [
    {"q": "Write a 2-sentence post about remote work and list keywords.",
     "expect_tool": "extract_keywords", "must_contain": ["remote"]},
    {"q": "How long would it take to read this: 'AI agents are changing software.'?",
     "expect_tool": "reading_time", "must_contain": []},
    {"q": "Write a short social post about open source with hashtags.",
     "expect_tool": "suggest_hashtags", "must_contain": ["#"]},
    {"q": "Summarise in one sentence: LLMs predict the next token.",
     "expect_tool": None, "must_contain": ["token"]},
]


def judge(q, a) -> int:
    out = generate(f"Rate 1-5 how well the answer fulfils the request. Reply with one digit.\n"
                   f"REQUEST: {q}\nANSWER: {a}", temperature=0.0)
    digits = [c for c in out if c.isdigit()]
    return int(digits[0]) if digits else 0


def run_evals() -> pd.DataFrame:
    rows = []
    for c in CASES:
        r = run_agent(c["q"])
        used = {t["tool"] for t in r["tool_calls"]}
        rows.append({
            "request": c["q"][:50],
            "tool_ok": (c["expect_tool"] in used) if c["expect_tool"] else True,
            "content_ok": all(k.lower() in r["answer"].lower() for k in c["must_contain"]),
            "judge_1to5": judge(c["q"], r["answer"]),
        })
    df = pd.DataFrame(rows)
    return df