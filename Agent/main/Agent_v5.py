"""
Execution Monitor (Module-integrated)
-------------------------------------
- Streams stdout/stderr per step
- Saves logs to central dir
- Artifact checks
- (Optional) GPT audit

Now imports pipeline specs from: /home/jun/work/soongsil/Agent/module/attack_spec.py
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

    # 실제 길이 집계가 출력되도록 수정
    artifacts_count = {k: len(v) for k, v in result.found_artifacts.items()}

    lines = [
        f"[{verdict}] exit={result.exit_code} time={result.duration_sec:.1f}s log={result.log_path}",
        f"  artifacts: {artifacts_count}",
    ]
    return "\n".join(lines)


# ======================================================
# GPT verification (optional)
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
아래는 '{step_title}' 단계의 로그입니다. 아래 JSON 형식(키 5개만)으로만 응답하세요.
{{
  "step_executed": true/false,
  "errors": ["..."],
  "warnings": ["..."],
  "evidence": ["로그의 대표 라인 3~5개"],
  "summary": "사람이 읽기 쉬운 짧은 요약"
}}
로그:
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
            # temperature=0.0,
            max_tokens=1200,
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

    # Ensure all keys exist in the final dictionary
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
# Orchestrator
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

        # 누락되었던 append 복구
        report_steps.append(step_entry)

    return {
        "overall_verdict": "PASS" if all_pass else "CHECK",
        "steps": report_steps,
    }


# (Optional) RAG with Chroma
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

    # 1) Chroma 연결 (임베딩 함수 전달 X)
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

    # 3) 쿼리 임베딩 직접 생성 → query_embeddings 사용 (인덱싱과 동일 모델: 3072차원)
    try:
        import os
        from chromadb.utils import embedding_functions
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-large",  # 인덱싱과 동일
        )
        q_emb = ef([query])  # 1×3072

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
        "다음은 지식베이스 발췌와 실행 로그입니다.\n"
        "근거와 수치를 명시하며, 공격/붕괴 정황을 평가하세요.\n\n"
        f"[RAG Query]\n{query}\n\n[KB Snippets]\n{kb_context}\n\n[Log Tail]\n{tail}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "당신은 머신러닝 보안/RAG 분석가입니다. 정확·간결·근거 기반으로 답변하세요."},
                {"role": "user", "content": user_msg},
            ],
            #temperature=0.0,
        )
        ans = (resp.choices[0].message.content or "").strip()
        return {"ok": True, "answer": ans, "sources": sources, "error": None}
    except Exception as e:
        return {"ok": False, "answer": "", "sources": sources, "error": f"gpt_request_error: {e}"}



def analyze_only(specs: List[StepSpec], gpt_model: str = "gpt-5") -> Dict[str, Any]:
    """
    스펙의 log_path 파일들만 읽어서 RAG로 분석하고, 결과를 JSON으로 반환
    파일 저장은 __main__에서 처리
    """
    # OpenAI 클라이언트
    client = None
    if _OPENAI_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        if api_key:
            try:
                client = OpenAI(api_key=api_key)
            except Exception:
                client = None

    results = []
    for spec in specs:
        # 로그 읽기
        try:
            with open(spec.log_path, "r", encoding="utf-8", errors="ignore") as f:
                log_text = f.read()
        except Exception as e:
            results.append({
                "title": spec.title,
                "log_path": spec.log_path,
                "error": f"log read error: {e}",
                "rag": None,
                "summary": None,
            })
            continue

        # RAG 분석
        rag = rag_chroma_analyze(
            client=client,
            model=gpt_model,
            rag_query=spec.rag_request or "",
            log_text=log_text,
            chroma_collection_name="papers",
            chroma_host="127.0.0.1",
            chroma_port=8000,
            top_k=3,
        )

        # (선택) 요약 분석: RAG 외에 간단 summary도 원하면 사용
        summary_text, summary_err = None, None
        if client and spec.analysis_prompt:
            try:
                resp = client.chat.completions.create(
                    model=gpt_model,
                    messages=[
                        {"role": "system", "content": "숙련된 ML 로그 분석가처럼 핵심만 정확하게 요약하세요."},
                        {"role": "user", "content": spec.analysis_prompt + "\n\n----- 로그 원문 (후미 일부) -----\n" + log_text[-200000:]},
                    ],
                    #temperature=0.0,  # max_tokens 제거
                )
                summary_text = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                summary_err = f"gpt_request_error: {e}"
        elif not client:
            summary_err = "openai client unavailable or API key not set"

        results.append({
            "title": spec.title,
            "log_path": spec.log_path,
            "rag": rag,                         # {ok, answer, sources, error}
            "summary": {"text": summary_text, "error": summary_err},
        })

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": gpt_model,
        "items": results,
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
        # 분석 수행
        analysis_report = analyze_only(specs, gpt_model="gpt-5")

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
        analysis_report = analyze_only(specs, gpt_model="gpt-5")

        out_path = os.path.join(LOG_DIR, "analysis_accumulative.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(analysis_report, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 40)
        print("=== ACCUMULATIVE ANALYSIS REPORT (JSON) ===")
        print("=" * 40)
        print(json.dumps(analysis_report, ensure_ascii=False, indent=2))
        print(f"\nAnalysis saved to: {out_path}")
        

    else:
        # 기존 파이프라인 실행
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
