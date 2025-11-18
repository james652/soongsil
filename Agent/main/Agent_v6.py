#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Execution Monitor (Module-integrated) + LangChain Memory Summaries
-------------------------------------------------------------------
- Streams stdout/stderr per step
- Saves logs to central dir
- Artifact checks
- RAG (Chroma) optional
- **Summaries are saved to LangChain memory**
- **Each new summary is generated from: [previous summaries] + [current RAG] + [current log]**

Requires:
  pip install openai langchain langchain-openai langchain-community chromadb

Env:
  OPENAI_API_KEY (or OPENAI_APIKEY)
  USE_RAG=0/1      (default 1)
  LOG_TAIL=int     (default 20000)
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# --- (Optional) quiet LangChain tracer to avoid KeyError('output') noise ---
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
# If LangSmith key is present, the tracer may attach automatically; remove to be safe.
os.environ.pop("LANGSMITH_API_KEY", None)

# ---- Import shared specs/types from external module ----
#   File path: /home/jun/work/soongsil/Agent/module/attack_spec.py
from attack_spec import (
    LOG_DIR,  # centralized log dir
    StepSpec,  # dataclass for step specification
    build_brainwash_specs,
    build_accumulative_specs,
    build_test_specs,
    build_analyze_brainwash_specs,
    build_analyze_accumulative_specs,
)

# ---------- Optional: OpenAI (guarded import) ----------
try:
    from openai import OpenAI  # pip install openai
    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False

# ---------- LangChain (memory) ----------
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda


# ======================================================
# Utilities (memory helpers)
# ======================================================
def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def get_history_store(session_id: str) -> FileChatMessageHistory:
    mem_dir = _ensure_dir(os.path.join(LOG_DIR, "memory"))
    return FileChatMessageHistory(os.path.join(mem_dir, f"{session_id}.json"))

def save_summary_to_memory(session_id: str, step_title: str, summary_text: str) -> None:
    """Store only the summary into LangChain memory (as AIMessage JSON)."""
    hist = get_history_store(session_id)
    hist.add_message(HumanMessage(content=f"[{step_title}] save summary"))
    hist.add_message(AIMessage(content=json.dumps({"summary": summary_text or ""}, ensure_ascii=False)))

def get_previous_summaries(session_id: str, max_items: int = 5) -> List[str]:
    """Extract previous summaries from memory (oldest -> newest)."""
    hist = get_history_store(session_id)
    summaries: List[str] = []
    for msg in reversed(hist.messages):
        if isinstance(msg, AIMessage):
            try:
                obj = json.loads(msg.content or "")
                if isinstance(obj, dict) and isinstance(obj.get("summary"), str):
                    summaries.append(obj["summary"])
            except Exception:
                pass
        if len(summaries) >= max_items:
            break
    return list(reversed(summaries))


# ======================================================
# Result model
# ======================================================
@dataclass
class StepResult:
    """A dataclass to hold the results of a single execution step."""
    exit_code: int
    duration_sec: float
    log_path: str
    found_artifacts: Dict[str, List[str]]
    gpt_summary: Optional[str] = None
    gpt_flags: Optional[Dict[str, Any]] = None
    title: Optional[str] = None  # For report generation
    analysis_prompt: Optional[str] = None
    rag_request: Optional[str] = None


# ======================================================
# CORE: run and stream
# ======================================================
def run_and_stream(spec: StepSpec) -> StepResult:
    """Executes a command, streams its output, and checks for artifacts."""
    t0 = time.time()
    os.makedirs(os.path.dirname(spec.log_path) or ".", exist_ok=True)

    with open(spec.log_path, "w", encoding="utf-8") as logf:
        logf.write(f"--- RUN START: {spec.title} ---\n")
        logf.write(f"CMD: {spec.command}\n\n")
        logf.flush()

        # Start process
        proc = subprocess.Popen(
            spec.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Stream loop with optional timeout
        while True:
            if spec.timeout_sec and (time.time() - t0) > spec.timeout_sec:
                proc.kill()
                line = f"[MONITOR] Timeout reached ({spec.timeout_sec}s), process killed.\n"
                print(line, end="")
                logf.write(line)
                break

            line = proc.stdout.readline() if proc.stdout else ""
            if not line and proc.poll() is not None:
                break

            print(line, end="")
            logf.write(line)

        exit_code = proc.poll() if proc else -1
        logf.write(f"\n--- RUN END ({spec.title}) exit_code={exit_code} ---\n")

    # Artifact scan (recursive glob)
    found_artifacts: Dict[str, List[str]] = {}
    for patt in spec.expected_artifacts:
        found_artifacts[patt] = sorted(glob.glob(patt, recursive=True))

    return StepResult(
        exit_code=exit_code or 0,
        duration_sec=time.time() - t0,
        log_path=spec.log_path,
        found_artifacts=found_artifacts,
        title=spec.title,
        analysis_prompt=getattr(spec, "analysis_prompt", None),
        rag_request=getattr(spec, "rag_request", None),
    )


# ======================================================
# Local verification summary
# ======================================================
def summarize_local(result: StepResult) -> str:
    """Provides a simple PASS/CHECK summary based on local checks."""
    art_ok = True if not result.found_artifacts else any(
        len(v) > 0 for v in result.found_artifacts.values()
    )
    verdict = "PASS" if (result.exit_code == 0 and art_ok) else "CHECK"

    artifacts_count = {k: len(v) for k, v in result.found_artifacts.items()}

    lines = [
        f"[{verdict}] exit={result.exit_code} time={result.duration_sec:.1f}s log={result.log_path}",
        f"  artifacts: {artifacts_count}",
    ]
    return "\n".join(lines)


# ======================================================
# GPT verification (optional, legacy path)
# ======================================================
def gpt_verify_step(step_log_path: str, step_title: str, model: str = "gpt-5") -> Dict[str, Any]:
    """Sends the log to GPT for analysis and returns a structured JSON response."""
    if not _OPENAI_AVAILABLE:
        return {"error": "openai library not available"}

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)

    try:
        with open(step_log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_text = f.read()[-200000:]
    except Exception as e:
        return {
            "step_executed": False,
            "errors": [f"read error: {e}"],
            "warnings": [],
            "evidence": [],
            "summary": f"로그 파일({step_log_path})을 읽지 못했습니다: {e}",
        }

    prompt = f"""
Below is the log of the '{step_title}' step. Please respond only in the following JSON format (with exactly 5 keys).
{{
  "step_executed": true/false,
  "errors": ["..."],
  "warnings": ["..."],
  "evidence": ["3-5 representative lines from the log"],
  "summary": "A short, human-readable summary"
}}
log:
-----
{log_text}
-----
""".strip()

    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Return valid JSON with exactly the keys: step_executed, errors, warnings, evidence, summary.",
                },
                {"role": "user", "content": prompt},
            ],
            # do not pass temperature/max_tokens for newer models
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        last_lines = [ln for ln in log_text.strip().splitlines()[-10:]]
        summary_guess = (
            "로그 말미:\n" + "\n".join(last_lines[:5]) if last_lines else "요약 불가(모델 응답 파싱 실패)"
        )
        data = {
            "step_executed": None,
            "errors": [f"gpt_parse_error: {e}"],
            "warnings": [],
            "evidence": last_lines[:3] if last_lines else [],
            "summary": summary_guess,
        }

    for k, default in [
        ("step_executed", None),
        ("errors", []),
        ("warnings", []),
        ("evidence", []),
        ("summary", "요약 없음"),
    ]:
        data.setdefault(k, default)

    return data


# ======================================================
# Orchestrator (legacy path)
# ======================================================
def monitor_pipeline(specs: List[StepSpec], use_gpt: bool = True, gpt_model: str = "gpt-5") -> List[StepResult]:
    """Runs a pipeline of steps, monitors them, and optionally calls GPT for analysis."""
    results: List[StepResult] = []
    for spec in specs:
        print("\n" + "=" * 80)
        print(f"[RUN] {spec.title}")
        res = run_and_stream(spec)
        print("-" * 80)
        print(summarize_local(res))

        if use_gpt:
            g = gpt_verify_step(spec.log_path, spec.title, model=gpt_model)
            res.gpt_flags = g if isinstance(g, dict) else {"raw": str(g)}
            res.gpt_summary = (g or {}).get("summary")
            print("[GPT]", json.dumps(g, ensure_ascii=False, indent=2))

        results.append(res)

    return results


# ======================================================
# Helper: build overall report + print JSON to stdout
# ======================================================
def _step_local_verdict(s: StepResult) -> bool:
    """Determines if a single step passed based on exit code and artifacts."""
    art_ok = True if not s.found_artifacts else any(len(v) > 0 for v in s.found_artifacts.values())
    return (s.exit_code == 0) and art_ok


def build_report(steps: List[StepResult], gpt_model: str = "gpt-5") -> Dict[str, Any]:
    """Builds a final JSON report from all step results."""
    report_steps: List[Dict[str, Any]] = []
    all_pass = True

    for s in steps:
        local_pass = _step_local_verdict(s)
        all_pass = all_pass and local_pass

        step_entry: Dict[str, Any] = {
            "title": s.title,
            "local_pass": local_pass,
            "exit_code": s.exit_code,
            "duration_sec": round(s.duration_sec, 1),
            "log_path": s.log_path,
            "found_artifacts": {k: len(v) for k, v in s.found_artifacts.items()},
            "gpt": s.gpt_flags or {},
        }

        report_steps.append(step_entry)

    return {
        "overall_verdict": "PASS" if all_pass else "CHECK",
        "steps": report_steps,
    }


# ======================================================
# (Optional) RAG with Chroma
# ======================================================
def rag_chroma_analyze(
    client,
    model: str,
    rag_query: str,
    log_text: str,
    chroma_collection_name: str = "papers",
    chroma_host: str = "127.0.0.1",
    chroma_port: int = 8000,
    top_k: int = 3,
) -> Dict[str, Any]:
    if client is None:
        return {"ok": False, "answer": "", "sources": [], "error": "OpenAI client unavailable or API key not set"}

    # 1) Chroma 연결
    try:
        import chromadb
        chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        collection = chroma_client.get_collection(name=chroma_collection_name)
    except Exception as e:
        return {"ok": False, "answer": "", "sources": [], "error": f"ChromaDB connection failed: {e}"}

    # 2) 질의어
    query = (rag_query or "").strip()
    if not query:
        tail = (log_text or "")[-2000:]
        query = tail[:200] if tail else "poisoning attack analysis"

    # 3) 쿼리 임베딩 직접 생성
    try:
        from chromadb.utils import embedding_functions
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-large",
        )
        q_emb = ef([query])

        qres = collection.query(query_embeddings=q_emb, n_results=top_k)
        docs  = qres.get("documents", [[]])[0]
        ids   = qres.get("ids", [[]])[0]
        metas = qres.get("metadatas", [[]])[0]
        sources = [{"id": i, "doc": d, "meta": m} for i, d, m in zip(ids, docs, metas)]
    except Exception as e:
        return {"ok": False, "answer": "", "sources": [], "error": f"ChromaDB query error: {e}"}

    # 4) GPT 호출
    kb_context = "\n---\n".join([
        f"[{s['id']}] {s['meta'].get('title','(no title)')} ({s['meta'].get('year','-')})\n{s['doc']}"
        for s in sources
    ]) if sources else "(no kb match)"

    tail = (log_text or "")[-2000:]
    user_msg = (
        "The following contains excerpts from the knowledge base and execution logs.\n"
        " Evaluate the signs of attack or collapse, citing specific evidence and numerical indicators.\n"
        f"[RAG Query]\n{query}\n\n[KB Snippets]\n{kb_context}\n\n[Log Tail]\n{tail}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a machine learning security and RAG analyst. Respond accurately, concisely, and based on evidence."},
                {"role": "user", "content": user_msg},
            ],
        )
        ans = (resp.choices[0].message.content or "").strip()
        return {"ok": True, "answer": ans, "sources": sources, "error": None}
    except Exception as e:
        return {"ok": False, "answer": "", "sources": sources, "error": f"gpt_request_error: {e}"}


# ======================================================
# Memory-driven summarization chain
# ======================================================
def summarize_with_memory_chain(
    session_id: str,
    step_title: str,
    rag_text: str,
    prev_summaries: List[str],
    log_text: str,
    gpt_model: str = "gpt-5",
) -> Dict[str, Any]:
    """
    Build a summary using: [previous summaries in LangChain FileChatMessageHistory]
                           + [current RAG snippet]
                           + [current log tail]
    Always return a dict containing 'output' to keep LangChain tracer happy.
    """
    # This model only supports temperature=1 (default). Set explicitly to 1.0.
    llm = ChatOpenAI(model=gpt_model, temperature=1.0)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict ML log analyst. You must respond **only** in JSON format: "
            "{{\"step_executed\": bool, \"errors\": list[str], \"warnings\": list[str], "
            "\"evidence\": list[str], \"summary\": str}}"
        ),
        MessagesPlaceholder(variable_name="history"),
        (
            "human",
            "Step: {step_title}\n\n"
            "Previous step summaries (max 5):\n{prev_summaries}\n\n"
            "RAG snippet (empty if none):\n{rag_text}\n\n"
            "Current log (last 200k chars):\n{log_text}\n\n"
            "Based on the above, generate the JSON output."
        ),
    ])

    parser = JsonOutputParser()

    def _ensure_output_key(x: Any) -> Dict[str, Any]:
        # Make sure 'output' exists for tracer; prefer 'summary' if available.
        if isinstance(x, dict):
            if "output" not in x:
                out = x.get("summary") or json.dumps(x, ensure_ascii=False)
                return {"output": out, **x}
            return x
        return {"output": str(x)}

    chain = (prompt | llm | parser) | RunnableLambda(_ensure_output_key)

    with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id=session_id: get_history_store(session_id),
        input_messages_key="step_title",
        history_messages_key="history",
    )

    res = with_history.invoke(
        {
            "step_title": step_title,
            "prev_summaries": ("\n- " + "\n- ".join(prev_summaries)) if prev_summaries else "(none)",
            "rag_text": rag_text or "",
            "log_text": (log_text or "")[-200000:],
        },
        config={"configurable": {"session_id": session_id}},
    )

    for k, default in [
        ("step_executed", None),
        ("errors", []),
        ("warnings", []),
        ("evidence", []),
        ("summary", ""),
        ("output", ""),
    ]:
        res.setdefault(k, default)

    return res


# ======================================================
# Analysis-only (uses memory summaries)
# ======================================================
def analyze_only(specs: List[StepSpec], gpt_model: str = "gpt-5", session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Read logs and produce CURRENT summary from: [PREV summaries in memory] + [CURRENT RAG] + [CURRENT log].
    Save EACH summary into LangChain memory, and also write analysis.json.
    """
    # session id
    session_id = session_id or f"analysis_{time.strftime('%Y%m%d_%H%M%S')}"
    _ensure_dir(os.path.join(LOG_DIR, "memory"))

    # OpenAI client for RAG step
    client = None
    if _OPENAI_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        if api_key:
            try:
                client = OpenAI(api_key=api_key)
            except Exception:
                client = None

    USE_RAG = os.getenv("USE_RAG", "1") == "1"
    TRUNC = int(os.getenv("LOG_TAIL", "20000"))

    items = []
    for spec in specs:
        t0 = time.time()

        # 1) read log
        try:
            with open(spec.log_path, "r", encoding="utf-8", errors="ignore") as f:
                log_text_full = f.read()
        except Exception as e:
            items.append({
                "title": spec.title,
                "log_path": spec.log_path,
                "error": f"log read error: {e}",
                "rag": None,
                "summary": None,
            })
            continue
        log_text = log_text_full[-TRUNC:]

        # 2) previous summaries from memory
        prev_sums = get_previous_summaries(session_id, max_items=5)

        # 3) RAG (optional)
        rag_answer = ""
        rag_obj = {"ok": False, "answer": "", "sources": [], "error": "disabled"}
        if USE_RAG and client is not None:
            rag_obj = rag_chroma_analyze(
                client=client,
                model=gpt_model,
                rag_query=spec.rag_request or "",
                log_text=log_text,
                chroma_collection_name="papers",
                chroma_host="127.0.0.1",
                chroma_port=8000,
                top_k=1
            )
            rag_answer = (rag_obj or {}).get("answer") or ""

        # 4) build current summary from [prev_summaries + rag + log]
        g = summarize_with_memory_chain(
            session_id=session_id,
            step_title=spec.title,
            rag_text=rag_answer,
            prev_summaries=prev_sums,
            log_text=log_text,
            gpt_model=gpt_model
        )
        summary_text = (g or {}).get("summary") or ""

        # 5) save current summary into memory (always)
        save_summary_to_memory(session_id, spec.title, summary_text)

        # 6) collect
        items.append({
            "title": spec.title,
            "log_path": spec.log_path,
            "rag": rag_obj,
            "summary": {"text": summary_text, "error": None},
            "elapsed_sec": round(time.time() - t0, 2),
        })

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": gpt_model,
        "session_id": session_id,
        "items": items,
        "hints": {"USE_RAG": USE_RAG, "LOG_TAIL": TRUNC}
    }
    return report


# ======================================================
# CLI runner
# ======================================================
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    choice = input("어떤 프로그램을 실행할까요? (Brainwash / Accumulative / Test / Analysis/ Analysis_Accumulative) [Brainwash]: ").strip() or "Brainwash"

    if choice.lower() in ("analysis", "analyze", "a"):
        # 분석 스펙 불러오기
        specs = build_analyze_brainwash_specs()
        # 세션 ID 고정(원하면 외부에서 넘길 수도 있음)
        session_id = f"analysis_{time.strftime('%Y%m%d_%H%M%S')}"

        # 분석 수행 (메모리 요약 저장 포함)
        analysis_report = analyze_only(specs, gpt_model="gpt-5", session_id=session_id)

        # 파일로 저장
        out_path = os.path.join(LOG_DIR, "analysis.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(analysis_report, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 40)
        print("=== ANALYSIS REPORT (JSON) ===")
        print("=" * 40)
        print(json.dumps(analysis_report, ensure_ascii=False, indent=2))
        print(f"\nAnalysis saved to: {out_path}")

    elif choice.lower() in ("analysis_accumulative", "analyze_accumulative", "aa"):
        specs = build_analyze_accumulative_specs()
        session_id = f"analysis_accu_{time.strftime('%Y%m%d_%H%M%S')}"
        analysis_report = analyze_only(specs, gpt_model="gpt-5", session_id=session_id)

        out_path = os.path.join(LOG_DIR, "analysis_accumulative.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(analysis_report, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 40)
        print("=== ACCUMULATIVE ANALYSIS REPORT (JSON) ===")
        print("=" * 40)
        print(json.dumps(analysis_report, ensure_ascii=False, indent=2))
        print(f"\nAnalysis saved to: {out_path}")

    else:
        # 기존 파이프라인 실행 (레거시 경로; 메모리 요약 체인은 analysis 전용)
        specs = {
            "brainwash": build_brainwash_specs,
            "bw": build_brainwash_specs,
            "accumulative": build_accumulative_specs,
            "accu": build_accumulative_specs,
            "acc": build_accumulative_specs,
            "test": build_test_specs,
        }.get(choice.lower(), build_brainwash_specs)()

        out = monitor_pipeline(specs, use_gpt=True, gpt_model="gpt-5")
        report = build_report(out, gpt_model="gpt-5")
        print("\n" + "=" * 40)
        print("=== EXECUTION & MONITOR REPORT (JSON) ===")
        print("=" * 40)
        print(json.dumps(report, ensure_ascii=False, indent=2))

        out_path = os.path.join(LOG_DIR, f"monitor_summary_{choice.lower()}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to: {out_path}")
