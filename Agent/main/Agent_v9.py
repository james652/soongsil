#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agent Runner (Class-based, LangChain Memory + RAG + RAGAS)
-----------------------------------------------------------
- Per-step memory accumulation via LangChain FileChatMessageHistory
- Summarization uses: [previous summaries] + [current RAG] + [current log]
- Optional Chroma RAG and RAGAS-style metrics
- Backward-compatible CLI (Brainwash/Accumulative/Test/Analysis/Analysis_Accumulative)

Env:
  OPENAI_API_KEY (or OPENAI_APIKEY)
  USE_RAG=0/1      (default 1)
  LOG_TAIL=int     (default 20000)
  RAGAS_EMBED_MODEL (default "text-embedding-3-small")
"""

from __future__ import annotations

import os
import json
import time
import glob
import math
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------------------------------
# Attack specs (external module)
# ------------------------------------------------------------------------------------
# from module.attack_spec import ...  # 환경에 따라 조정
from attack_spec import (
    LOG_DIR,
    StepSpec,
    build_brainwash_specs,
    build_accumulative_specs,
    build_test_specs,
    build_analyze_brainwash_specs,
    build_analyze_accumulative_specs,
)

# ------------------------------------------------------------------------------------
# Optional OpenAI (guarded import)
# ------------------------------------------------------------------------------------
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False

# ------------------------------------------------------------------------------------
# LangChain (memory)
# ------------------------------------------------------------------------------------
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.pop("LANGSMITH_API_KEY", None)

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda


# ====================================================================================
# Utilities
# ====================================================================================
def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ====================================================================================
# Memory Manager
# ====================================================================================
class MemoryManager:
    def __init__(self, log_dir: str):
        self.mem_dir = _ensure_dir(os.path.join(log_dir, "memory"))

    def store_path(self, session_id: str) -> str:
        return os.path.join(self.mem_dir, f"{session_id}.json")

    def history(self, session_id: str) -> FileChatMessageHistory:
        return FileChatMessageHistory(self.store_path(session_id))

    def save_summary(self, session_id: str, step_title: str, summary_text: str) -> None:
        hist = self.history(session_id)
        hist.add_message(HumanMessage(content=f"[{step_title}] save summary"))
        hist.add_message(AIMessage(content=json.dumps({"summary": summary_text or ""}, ensure_ascii=False)))

    def get_previous_summaries(self, session_id: str, max_items: int = 5) -> List[str]:
        hist = self.history(session_id)
        out: List[str] = []
        for msg in reversed(hist.messages):
            if isinstance(msg, AIMessage):
                try:
                    obj = json.loads(msg.content or "")
                    if isinstance(obj, dict) and isinstance(obj.get("summary"), str):
                        out.append(obj["summary"])
                except Exception:
                    pass
            if len(out) >= max_items:
                break
        return list(reversed(out))


# ====================================================================================
# RAG Retriever (Chroma, optional)
# ====================================================================================
class RAGRetriever:
    def __init__(self, openai_client: Optional[OpenAI], model: str = "gpt-5"):
        self.client = openai_client
        self.model = model

    def analyze(
        self,
        rag_query: str,
        log_text: str,
        chroma_collection_name: str = "papers",
        chroma_host: str = "127.0.0.1",
        chroma_port: int = 8000,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        if self.client is None:
            return {"ok": False, "answer": "", "sources": [], "error": "OpenAI client unavailable or API key not set"}

        try:
            import chromadb
            chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            collection = chroma_client.get_collection(name=chroma_collection_name)
        except Exception as e:
            return {"ok": False, "answer": "", "sources": [], "error": f"ChromaDB connection failed: {e}"}

        query = (rag_query or "").strip()
        if not query:
            tail = (log_text or "")[-2000:]
            query = tail[:200] if tail else "poisoning attack analysis"

        try:
            from chromadb.utils import embedding_functions
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY"),
                model_name="text-embedding-3-large",
            )
            q_emb = ef([query])
            qres = collection.query(query_embeddings=q_emb, n_results=top_k)
            docs = qres.get("documents", [[]])[0]
            ids = qres.get("ids", [[]])[0]
            metas = qres.get("metadatas", [[]])[0]
            sources = [{"id": i, "doc": d, "meta": m} for i, d, m in zip(ids, docs, metas)]
        except Exception as e:
            return {"ok": False, "answer": "", "sources": [], "error": f"ChromaDB query error: {e}"}

        kb_context = "\n---\n".join([
            f"[{s['id']}] {s['meta'].get('title','(no title)')} ({s['meta'].get('year','-')})\n{s['doc']}"
            for s in sources
        ]) if sources else "(no kb match)"

        tail = (log_text or "")[-2000:]
        user_msg = (
            "The following contains excerpts from the knowledge base and execution logs.\n"
            "Evaluate attack/collapse indicators with concrete evidence and numbers.\n"
            f"[RAG Query]\n{query}\n\n[KB Snippets]\n{kb_context}\n\n[Log Tail]\n{tail}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict ML security analyst. Be concise, cite numbers."},
                    {"role": "user", "content": user_msg},
                ],
            )
            ans = (resp.choices[0].message.content or "").strip()
            return {"ok": True, "answer": ans, "sources": sources, "error": None}
        except Exception as e:
            return {"ok": False, "answer": "", "sources": sources, "error": f"gpt_request_error: {e}"}


# ====================================================================================
# Summarizer (LangChain chain with memory)
# ====================================================================================
class Summarizer:
    def __init__(self, memory_mgr: MemoryManager, model: str = "gpt-5", temperature: float = 1.0):
        self.memory_mgr = memory_mgr
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.parser = JsonOutputParser()

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a machine learning results analysis expert specializing in poisoning detection. "
                "Please write it in the form of a report. "
                "Your task is to infer which exact attack occurred (if any) from model logs, RAG evidence, and prior summaries. "
                "You must choose decisively among known attack types such as Brainwash, Accumulative Attack, Label Flipping, or No Attack (clean scenario). "
                "Avoid vague language and respond ONLY in JSON: "
                "{{\"step_executed\": bool, \"errors\": [], \"warnings\": [], \"evidence\": [], \"summary\": str}}"
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "Step: {step_title}\n\n"
                "Previous step summaries (max 5):\n{prev_summaries}\n\n"
                "RAG snippet (empty if none):\n{rag_text}\n\n"
                "Current log (last 200k chars):\n{log_text}\n\n"
                "Generate the JSON output."
            ),
        ])

        def _ensure_output_key(x: Any) -> Dict[str, Any]:
            if isinstance(x, dict):
                if "output" not in x:
                    out = x.get("summary") or json.dumps(x, ensure_ascii=False)
                    return {"output": out, **x}
                return x
            return {"output": str(x)}

        self.chain_core = (self.prompt | self.llm | self.parser) | RunnableLambda(_ensure_output_key)

    def summarize(self, session_id: str, step_title: str, rag_text: str, prev_summaries: List[str], log_text: str) -> Dict[str, Any]:
        with_history = RunnableWithMessageHistory(
            self.chain_core,
            lambda session_id=session_id: self.memory_mgr.history(session_id),
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
            ("step_executed", None), ("errors", []), ("warnings", []),
            ("evidence", []), ("summary", ""), ("output", "")
        ]:
            res.setdefault(k, default)
        return res


# ====================================================================================
# RAGAS Evaluator
# ====================================================================================
class RAGASEvaluator:
    def __init__(self, openai_client: Optional[OpenAI], model: str = "gpt-5"):
        self.client = openai_client
        self.model = model

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a)) or 1e-9
        nb = math.sqrt(sum(y*y for y in b)) or 1e-9
        return max(min(dot / (na * nb), 1.0), -1.0)

    @staticmethod
    def _safe_num(s: str, default: float = 0.0) -> float:
        try:
            return float((s or "").strip())
        except Exception:
            return default

    def evaluate(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        if self.client is None:
            return {"error": "openai client not available"}

        out: Dict[str, Any] = {}

        # Faithfulness
        try:
            f_prompt = (
                "Split the answer into minimal factual statements and check each against the context.\n"
                "Return ONLY one numeric value in [0,1] = fraction supported by context.\n\n"
                f"Context:\n{context}\n\nAnswer:\n{answer}"
            )
            f_resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return a single number [0,1]."},
                    {"role": "user", "content": f_prompt},
                ],
            )
            val = self._safe_num(f_resp.choices[0].message.content, 0.0)
            out["faithfulness"] = max(0.0, min(1.0, val))
        except Exception as e:
            out["faithfulness_error"] = f"{e}"

        # Answer Relevance
        try:
            emb_model = os.getenv("RAGAS_EMBED_MODEL", "text-embedding-3-small")
            q_emb = self.client.embeddings.create(input=question or "", model=emb_model).data[0].embedding
            a_emb = self.client.embeddings.create(input=answer or "", model=emb_model).data[0].embedding
            cos = self._cosine(q_emb, a_emb)
            out["answer_relevance"] = (cos + 1.0) / 2.0
        except Exception as e:
            out["answer_relevance_error"] = f"{e}"

        # Context Relevance
        try:
            cr_prompt = (
                "Estimate how focused and necessary the context is to answer the question.\n"
                "Return ONLY one numeric value in [0,1].\n\n"
                f"Question:\n{question}\n\nContext:\n{context}"
            )
            cr_resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return a single number [0,1]."},
                    {"role": "user", "content": cr_prompt},
                ],
            )
            val = self._safe_num(cr_resp.choices[0].message.content, 0.0)
            out["context_relevance"] = max(0.0, min(1.0, val))
        except Exception as e:
            out["context_relevance_error"] = f"{e}"

        return out


# ====================================================================================
# Runner (exec + local checks)
# ====================================================================================
@dataclass
class StepRunResult:
    exit_code: int
    duration_sec: float
    log_path: str
    found_artifacts: Dict[str, List[str]]
    title: Optional[str] = None
    analysis_prompt: Optional[str] = None
    rag_request: Optional[str] = None
    gpt_flags: Optional[Dict[str, Any]] = None
    gpt_summary: Optional[str] = None
    ragas_scores: Optional[Dict[str, Any]] = None


class Runner:
    @staticmethod
    def run_and_stream(spec: StepSpec) -> StepRunResult:
        t0 = time.time()
        os.makedirs(os.path.dirname(spec.log_path) or ".", exist_ok=True)
        with open(spec.log_path, "w", encoding="utf-8") as logf:
            logf.write(f"--- RUN START: {spec.title} ---\n")
            logf.write(f"CMD: {spec.command}\n\n")
            logf.flush()

            proc = subprocess.Popen(
                spec.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            while True:
                if getattr(spec, "timeout_sec", None) and (time.time() - t0) > spec.timeout_sec:
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

        found_artifacts: Dict[str, List[str]] = {}
        for patt in getattr(spec, "expected_artifacts", []):
            found_artifacts[patt] = sorted(glob.glob(patt, recursive=True))

        return StepRunResult(
            exit_code=exit_code or 0,
            duration_sec=time.time() - t0,
            log_path=spec.log_path,
            found_artifacts=found_artifacts,
            title=spec.title,
            analysis_prompt=getattr(spec, "analysis_prompt", None),
            rag_request=getattr(spec, "rag_request", None),
        )

    @staticmethod
    def summarize_local(result: StepRunResult) -> str:
        art_ok = True if not result.found_artifacts else any(len(v) > 0 for v in result.found_artifacts.values())
        verdict = "PASS" if (result.exit_code == 0 and art_ok) else "CHECK"
        artifacts_count = {k: len(v) for k, v in result.found_artifacts.items()}
        return "\n".join([
            f"[{verdict}] exit={result.exit_code} time={result.duration_sec:.1f}s log={result.log_path}",
            f"  artifacts: {artifacts_count}",
        ])


# ====================================================================================
# Analyzer (analysis-only, reused by monitor)
# ====================================================================================
class Analyzer:
    def __init__(self, log_dir: str, model: str = "gpt-5"):
        self.log_dir = log_dir
        self.model = model

        self.client = None
        if _OPENAI_AVAILABLE:
            _key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
            if _key:
                try:
                    self.client = OpenAI(api_key=_key)
                except Exception:
                    self.client = None

        self.memory = MemoryManager(log_dir)
        self.summarizer = Summarizer(self.memory, model=self.model, temperature=1.0)
        self.ragas = RAGASEvaluator(self.client, model=self.model)
        self.rag = RAGRetriever(self.client, model=self.model)

    def _read_log_tail(self, path: str, tail_chars: int) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[-tail_chars:]

    def analyze(self, specs: List[StepSpec], session_id: Optional[str] = None) -> Dict[str, Any]:
        session_id = session_id or f"analysis_{time.strftime('%Y%m%d_%H%M%S')}"
        _ = _ensure_dir(os.path.join(self.log_dir, "memory"))

        USE_RAG = os.getenv("USE_RAG", "1") == "1"
        TRUNC = int(os.getenv("LOG_TAIL", "20000"))

        items: List[Dict[str, Any]] = []
        for spec in specs:
            t0 = time.time()

            # 1) Read log
            try:
                log_text = self._read_log_tail(spec.log_path, TRUNC)
            except Exception as e:
                items.append({
                    "title": spec.title,
                    "log_path": spec.log_path,
                    "error": f"log read error: {e}",
                    "rag": None,
                    "summary": None,
                })
                continue

            # 2) Previous summaries
            prev_sums = self.memory.get_previous_summaries(session_id, max_items=5)

            # 3) RAG (optional)
            rag_answer = ""
            rag_obj = {"ok": False, "answer": "", "sources": [], "error": "disabled"}
            if USE_RAG and self.client is not None:
                rag_obj = self.rag.analyze(
                    rag_query=spec.rag_request or "",
                    log_text=log_text,
                    chroma_collection_name="papers",
                    chroma_host="127.0.0.1",
                    chroma_port=8000,
                    top_k=1,
                )
                rag_answer = (rag_obj or {}).get("answer") or ""

            # 4) Summarize with memory + rag + log
            g = self.summarizer.summarize(
                session_id=session_id,
                step_title=spec.title,
                rag_text=rag_answer,
                prev_summaries=prev_sums,
                log_text=log_text,
            )
            summary_text = (g or {}).get("summary") or ""

            # 5) Save current summary into memory
            self.memory.save_summary(session_id, spec.title, summary_text)

            # 6) RAGAS scoring
            try:
                question = (spec.rag_request or spec.analysis_prompt or f"What happened in step: {spec.title}?")[:1000]
                context = f"rag_answer: {rag_answer}\n\nlog_text: {log_text}"
                answer = summary_text[:4000]
                ragas_scores = self.ragas.evaluate(question, context, answer)
            except Exception as e:
                ragas_scores = {"error": f"ragas_scoring_failed: {e}"}

            # 7) Collect
            items.append({
                "title": spec.title,
                "log_path": spec.log_path,
                "rag": rag_obj,
                "summary": {"text": summary_text, "error": None},
                "ragas_scores": ragas_scores,
                "elapsed_sec": round(time.time() - t0, 2),
            })

        # Aggregate RAGAS avg
        def _avg(xs: List[float]) -> Optional[float]:
            return round(sum(xs) / len(xs), 3) if xs else None

        f_vals, ar_vals, cr_vals = [], [], []
        for it in items:
            rs = it.get("ragas_scores") or {}
            f = rs.get("faithfulness"); ar = rs.get("answer_relevance"); cr = rs.get("context_relevance")
            if isinstance(f, (int, float)): f_vals.append(float(f))
            if isinstance(ar, (int, float)): ar_vals.append(float(ar))
            if isinstance(cr, (int, float)): cr_vals.append(float(cr))

        return {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "session_id": session_id,
            "ragas_avg": {
                "faithfulness": _avg(f_vals),
                "answer_relevance": _avg(ar_vals),
                "context_relevance": _avg(cr_vals),
            } if any([f_vals, ar_vals, cr_vals]) else None,
            "items": items,
            "hints": {"USE_RAG": os.getenv("USE_RAG", "1") == "1", "LOG_TAIL": int(os.getenv("LOG_TAIL", "20000"))},
        }


# ====================================================================================
# Legacy GPT verify (fallback only)
# ====================================================================================
def gpt_verify_step(step_log_path: str, step_title: str, model: str = "gpt-5") -> Dict[str, Any]:
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
Return valid JSON with exactly: step_executed, errors, warnings, evidence, summary.
log of '{step_title}':
-----
{log_text}
-----
""".strip()

    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return valid JSON with exactly the keys: step_executed, errors, warnings, evidence, summary."},
                {"role": "user", "content": prompt},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        last_lines = [ln for ln in log_text.strip().splitlines()[-10:]]
        summary_guess = ("로그 말미:\n" + "\n".join(last_lines[:5])) if last_lines else "요약 불가(모델 응답 파싱 실패)"
        data = {
            "step_executed": None,
            "errors": [f"gpt_parse_error: {e}"],
            "warnings": [],
            "evidence": last_lines[:3] if last_lines else [],
            "summary": summary_guess,
        }

    for k, default in [("step_executed", None), ("errors", []), ("warnings", []), ("evidence", []), ("summary", "요약 없음")]:
        data.setdefault(k, default)
    return data


# ====================================================================================
# Report Builders
# ====================================================================================
def _step_local_verdict(s: StepRunResult) -> bool:
    art_ok = True if not s.found_artifacts else any(len(v) > 0 for v in s.found_artifacts.values())
    return (s.exit_code == 0) and art_ok


def build_report(steps: List[StepRunResult], gpt_model: str = "gpt-5") -> Dict[str, Any]:
    report_steps: List[Dict[str, Any]] = []
    all_pass = True
    f_vals, ar_vals, cr_vals = [], [], []

    for s in steps:
        local_pass = _step_local_verdict(s)
        all_pass = all_pass and local_pass
        entry: Dict[str, Any] = {
            "title": s.title,
            "local_pass": local_pass,
            "exit_code": s.exit_code,
            "duration_sec": round(s.duration_sec, 1),
            "log_path": s.log_path,
            "found_artifacts": {k: len(v) for k, v in s.found_artifacts.items()},
            "gpt": s.gpt_flags or {},
        }
        if s.ragas_scores:
            entry["ragas_scores"] = s.ragas_scores
            f = s.ragas_scores.get("faithfulness"); ar = s.ragas_scores.get("answer_relevance"); cr = s.ragas_scores.get("context_relevance")
            if isinstance(f, (int, float)): f_vals.append(float(f))
            if isinstance(ar, (int, float)): ar_vals.append(float(ar))
            if isinstance(cr, (int, float)): cr_vals.append(float(cr))
        report_steps.append(entry)

    def _avg(xs: List[float]) -> Optional[float]:
        return round(sum(xs) / len(xs), 3) if xs else None

    ragas_avg = {
        "faithfulness": _avg(f_vals),
        "answer_relevance": _avg(ar_vals),
        "context_relevance": _avg(cr_vals),
    } if any([f_vals, ar_vals, cr_vals]) else None

    return {"overall_verdict": "PASS" if all_pass else "CHECK", "ragas_avg": ragas_avg, "steps": report_steps}


# ====================================================================================
# Monitor pipeline (now using LangChain-style Analysis by default)
# ====================================================================================
def monitor_pipeline(
    specs: List[StepSpec],
    use_langchain_analysis: bool = True,   # 기본값: Analysis 경로 사용
    fallback_gpt_verify: bool = False,     # 필요시만 True
    gpt_model: str = "gpt-5",
    pipeline_tag: str = "brainwash"
) -> List[StepRunResult]:
    """
    - 각 스텝 실행 후, Analysis와 동일한 방식(LangChain Memory + RAG + RAGAS)으로 요약 생성
    - 스텝 단위 요약 JSON 저장: monitor_summary_{pipeline_tag}_step{idx}.json
    - (옵션) fallback_gpt_verify=True 이면 gpt_verify_step도 병행/대체 수행
    """
    out: List[StepRunResult] = []

    # Prepare shared components for analysis-style summaries
    analyzer = Analyzer(LOG_DIR, model=gpt_model)
    session_id = f"monitor_{pipeline_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    _ = _ensure_dir(os.path.join(LOG_DIR, "memory"))
    USE_RAG = os.getenv("USE_RAG", "1") == "1"
    TRUNC = int(os.getenv("LOG_TAIL", "20000"))

    for idx, spec in enumerate(specs, start=1):
        print("\n" + "=" * 80)
        print(f"[RUN] {spec.title}")
        res = Runner.run_and_stream(spec)
        print("-" * 80)
        print(Runner.summarize_local(res))

        # ----- Analysis-style summary (default path)
        if use_langchain_analysis:
            # 1) Read log tail
            try:
                with open(spec.log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_text = f.read()[-TRUNC:]
            except Exception as e:
                log_text = f"[log read error: {e}]"

            # 2) Previous mem summaries
            prev_sums = analyzer.memory.get_previous_summaries(session_id, max_items=5)

            # 3) RAG
            rag_answer = ""
            rag_obj = {"ok": False, "answer": "", "sources": [], "error": "disabled"}
            if USE_RAG and analyzer.client is not None:
                rag_obj = analyzer.rag.analyze(
                    rag_query=spec.rag_request or "",
                    log_text=log_text,
                    chroma_collection_name="papers",
                    chroma_host="127.0.0.1",
                    chroma_port=8000,
                    top_k=1,
                )
                rag_answer = (rag_obj or {}).get("answer") or ""

            # 4) Summarize (LangChain + memory)
            g = analyzer.summarizer.summarize(
                session_id=session_id,
                step_title=spec.title,
                rag_text=rag_answer,
                prev_summaries=prev_sums,
                log_text=log_text,
            )
            summary_text = (g or {}).get("summary") or ""
            analyzer.memory.save_summary(session_id, spec.title, summary_text)

            # 5) RAGAS
            try:
                question = (spec.rag_request or spec.analysis_prompt or f"What happened in step: {spec.title}?")[:1000]
                context = f"rag_answer: {rag_answer}\n\nlog_text: {log_text}"
                answer = summary_text[:4000]
                ragas_scores = analyzer.ragas.evaluate(question, context, answer)
            except Exception as e:
                ragas_scores = {"error": f"ragas_scoring_failed: {e}"}

            # Attach to result (use the same fields as before)
            res.gpt_flags = g if isinstance(g, dict) else {"raw": str(g)}
            res.gpt_summary = summary_text
            res.ragas_scores = ragas_scores

        # ----- (Optional) Fallback legacy GPT verify
        elif fallback_gpt_verify:
            g = gpt_verify_step(spec.log_path, spec.title, model=gpt_model)
            res.gpt_flags = g if isinstance(g, dict) else {"raw": str(g)}
            res.gpt_summary = (g or {}).get("summary")
            if res.gpt_summary:
                try:
                    with open(spec.log_path, "r", encoding="utf-8", errors="ignore") as f:
                        _log = f.read()[-2000:]
                    question = (spec.rag_request or spec.analysis_prompt or f"Summarize step: {spec.title}")[:1000]
                    context = _log
                    answer = res.gpt_summary[:4000]
                    res.ragas_scores = analyzer.ragas.evaluate(question, context, answer)
                except Exception as e:
                    res.ragas_scores = {"error": f"ragas_scoring_failed: {e}"}

        # 🔹 Step-level report save
        step_report = build_report([res], gpt_model=gpt_model)
        step_path = os.path.join(LOG_DIR, f"monitor_summary_{pipeline_tag}_step{idx}.json")
        with open(step_path, "w", encoding="utf-8") as f:
            json.dump(step_report, f, ensure_ascii=False, indent=2)
        print(f"[STEP REPORT] saved: {step_path}")

        out.append(res)

    return out


# ====================================================================================
# CLI
# ====================================================================================
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    choice = input("어떤 프로그램을 실행할까요? (Brainwash / Accumulative / Test / Analysis / Analysis_Accumulative) [Brainwash]: ").strip() or "Brainwash"

    if choice.lower() in ("analysis", "analyze", "a"):
        specs = build_analyze_brainwash_specs()
        session_id = f"analysis_{time.strftime('%Y%m%d_%H%M%S')}"
        analyzer = Analyzer(LOG_DIR, model="gpt-5")
        report = analyzer.analyze(specs, session_id=session_id)

        out_path = os.path.join(LOG_DIR, "analysis.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 40)
        print("=== ANALYSIS REPORT (JSON) ===")
        print("=" * 40)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nAnalysis saved to: {out_path}")

    elif choice.lower() in ("analysis_accumulative", "analyze_accumulative", "aa"):
        specs = build_analyze_accumulative_specs()
        session_id = f"analysis_accu_{time.strftime('%Y%m%d_%H%M%S')}"
        analyzer = Analyzer(LOG_DIR, model="gpt-5")
        report = analyzer.analyze(specs, session_id=session_id)

        out_path = os.path.join(LOG_DIR, "analysis_accumulative.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 40)
        print("=== ACCUMULATIVE ANALYSIS REPORT (JSON) ===")
        print("=" * 40)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nAnalysis saved to: {out_path}")

    else:
        build = {
            "brainwash": build_brainwash_specs,
            "bw": build_brainwash_specs,
            "accumulative": build_accumulative_specs,
            "accu": build_accumulative_specs,
            "acc": build_accumulative_specs,
            "test": build_test_specs,
        }.get(choice.lower(), build_brainwash_specs)

        specs = build()
        # Brainwash/Accumulative/Test 실행도 LangChain+RAG+RAGAS 방식으로 스텝 요약 생성
        results = monitor_pipeline(
            specs,
            use_langchain_analysis=True,   # 기본: Analysis와 동일 경로 사용
            fallback_gpt_verify=False,     # 필요시 True로 바꿔서 레거시 요약도 병행 가능
            gpt_model="gpt-5",
            pipeline_tag=choice.lower()
        )
        report = build_report(results, gpt_model="gpt-5")

        print("\n" + "=" * 40)
        print("=== EXECUTION & MONITOR REPORT (JSON) ===")
        print("=" * 40)
        print(json.dumps(report, ensure_ascii=False, indent=2))

        out_path = os.path.join(LOG_DIR, f"monitor_summary_{choice.lower()}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to: {out_path}")
