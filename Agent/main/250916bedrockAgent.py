"""
Execution Monitor (Bedrock-integrated)
--------------------------------------
- Streams stdout/stderr per step
- Saves logs to central dir
- Artifact checks
- LLM audit via AWS Bedrock (Anthropic Claude 3.5 Sonnet by default)

NOTE:
- Ensure attack_spec.py exports: LOG_DIR, StepSpec, build_brainwash_specs, build_accumulative_specs, build_test_specs
- Adjust the import line ('from attack_spec import ...') if your module path differs.

AWS REQUIREMENTS:
- env: AWS_REGION (e.g., us-east-1), AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (and optional AWS_SESSION_TOKEN)
- permission: bedrock:InvokeModel
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ==== Pipeline specs import (adjust path if needed) ====
from attack_spec import (
    LOG_DIR,  # centralized log dir
    StepSpec,  # dataclass for step specification
    build_brainwash_specs,
    build_accumulative_specs,
    build_test_specs,
)

# ==== AWS Bedrock (boto3) =================================
try:
    import boto3
    _BOTO3_AVAILABLE = True
except Exception:
    boto3 = None  # type: ignore
    _BOTO3_AVAILABLE = False


def _make_bedrock_client():
    """
    Create a Bedrock Runtime client.
    Requires AWS_REGION and AWS credentials in env or config.
    """
    if not _BOTO3_AVAILABLE:
        raise RuntimeError("boto3 not installed. `pip install boto3`")
    region = os.getenv("AWS_REGION", "us-east-1")
    return boto3.client("bedrock-runtime", region_name=region)


def bedrock_chat_json(
    user_prompt: str,
    *,
    system_prompt: str = "Return ONLY valid JSON.",
    model_id: str = "arn:aws:bedrock:us-east-1:171916340229:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    max_tokens: int = 1200,
    temperature: float = 0.0,
) -> dict:
    """
    Calls Anthropic Claude 3.5 Sonnet on AWS Bedrock and returns parsed JSON.
    If parsing fails, returns {"error": "...", "raw": "<model text>"}.
    """
    brt = _make_bedrock_client()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ],
    }

    resp = brt.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    payload = json.loads(resp["body"].read().decode("utf-8"))

    # Claude on Bedrock: text is under content[0].text
    text = ""
    if payload.get("content") and isinstance(payload["content"], list):
        first = payload["content"][0]
        if isinstance(first, dict):
            text = first.get("text", "") or ""

    # Try strict JSON slice
    try:
        start = text.find("{")
        end = text.rfind("}")
        return json.loads(text[start : end + 1])
    except Exception:
        return {"error": "JSON parse failed", "raw": text}


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
    gpt_summary: Optional[str] = None       # kept name for compatibility; filled by Bedrock
    gpt_flags: Optional[Dict[str, Any]] = None
    title: Optional[str] = None             # For report generation
    analysis_prompt: Optional[str] = None   # Optional per-step extra analysis prompt


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

    lines = [
        f"[{verdict}] exit={result.exit_code} time={result.duration_sec:.1f}s log={result.log_path}",
        f"  artifacts: {{k: len(v) for k, v in result.found_artifacts.items()}}",
    ]
    return "\n".join(lines)


# ======================================================
# Bedrock verification (LLM audit)
# ======================================================
def llm_verify_step_bedrock(
    step_log_path: str,
    step_title: str,
    model_id: str = "arn:aws:bedrock:us-east-1:171916340229:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
) -> Dict[str, Any]:
    """Sends the log to Bedrock (Claude) for analysis and returns a structured JSON response."""
    try:
        with open(step_log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_text = f.read()[-200000:]
    except Exception as e:
        return {
            "step_executed": None,
            "errors": [f"read error: {e}"],
            "warnings": [],
            "evidence": [],
            "summary": f"로그 파일({step_log_path})을 읽지 못했습니다: {e}",
        }

    user_prompt = f"""
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

    system_prompt = (
        "반드시 유효한 JSON만 출력하세요. 키는 정확히 5개(step_executed, errors, warnings, evidence, summary)입니다."
    )

    data = bedrock_chat_json(
        user_prompt,
        system_prompt=system_prompt,
        model_id=model_id,
        max_tokens=1000,
        temperature=0.0,
    )

    # Ensure keys exist
    for k, default in [
        ("step_executed", None),
        ("errors", []),
        ("warnings", []),
        ("evidence", []),
        ("summary", "요약 없음"),
    ]:
        if not isinstance(data, dict) or k not in data or data[k] is None:
            if isinstance(data, dict):
                data[k] = default
            else:
                data = {"error": "invalid response", "raw": str(data)}
                data[k] = default
    return data


# ======================================================
# Build specs by name (modular)
# ======================================================
def get_specs_by_name(name: str) -> List[StepSpec]:
    """Returns a list of StepSpec objects for a named pipeline."""
    key = name.strip().lower()
    if key in ("brainwash", "bw"):
        return build_brainwash_specs()
    if key in ("accumulative", "accu", "acc"):
        return build_accumulative_specs()
    if key in ("test",):
        return build_test_specs()
    raise SystemExit(f"Unknown pipeline: {name}")


# ======================================================
# Orchestrator
# ======================================================
def monitor_pipeline(
    specs: List[StepSpec],
    use_llm: bool = True,
    model_id: str = "arn:aws:bedrock:us-east-1:171916340229:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
) -> List[StepResult]:
    """Runs a pipeline of steps, monitors them, and optionally calls Bedrock for analysis."""
    results: List[StepResult] = []
    for spec in specs:
        print("\n" + "=" * 80)
        print(f"[RUN] {spec.title}")
        res = run_and_stream(spec)
        print("-" * 80)
        print(summarize_local(res))

        if use_llm:
            g = llm_verify_step_bedrock(res.log_path, spec.title, model_id=model_id)
            res.gpt_flags = g if isinstance(g, dict) else {"raw": str(g)}
            res.gpt_summary = (g or {}).get("summary")
            print("[Bedrock]", json.dumps(g, ensure_ascii=False, indent=2))

        results.append(res)

    return results


# ======================================================
# Helper: build overall report + (optional) per-step extra analysis
# ======================================================
def _step_local_verdict(s: StepResult) -> bool:
    """Determines if a single step passed based on exit code and artifacts."""
    art_ok = True if not s.found_artifacts else any(len(v) > 0 for v in s.found_artifacts.values())
    return (s.exit_code == 0) and art_ok


def build_report(
    steps: List[StepResult],
    model_id: str = "arn:aws:bedrock:us-east-1:171916340229:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
) -> Dict[str, Any]:
    """Builds a final JSON report; if a step has `analysis_prompt`, runs Bedrock with it."""
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
            "llm": s.gpt_flags or {},
        }

        # Optional: per-step extra analysis using the provided prompt
        if s.analysis_prompt:
            try:
                with open(s.log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_text = f.read()[-200000:]
            except Exception as e:
                step_entry["analysis"] = {"error": f"read error: {e}"}
            else:
                user_content = s.analysis_prompt.strip() + "\n\n----- 로그 (후미 일부) -----\n" + log_text
                out = bedrock_chat_json(
                    user_content,
                    system_prompt=(
                        '숙련된 머신러닝 로그 분석가처럼 간결·정확하게 요약하되 JSON만 반환하세요. {"summary":"..."} 스키마.'
                    ),
                    model_id=model_id,
                    max_tokens=1000,
                    temperature=0.0,
                )
                step_entry["analysis"] = out

        report_steps.append(step_entry)

    return {
        "overall_verdict": "PASS" if all_pass else "CHECK",
        "steps": report_steps,
    }


# ======================================================
# CLI runner
# ======================================================
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    choice = input("어떤 프로그램을 실행할까요? (Brainwash / Accumulative / Test) [Brainwash]: ").strip() or "Brainwash"
    specs = get_specs_by_name(choice)

    # Choose Bedrock model
    BEDROCK_MODEL = os.getenv("BEDROCK_MODEL_ID", "arn:aws:bedrock:us-east-1:171916340229:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0")

    # Run the pipeline with Bedrock verification
    out = monitor_pipeline(specs, use_llm=True, model_id=BEDROCK_MODEL)

    # Build and print the final report
    report = build_report(out, model_id=BEDROCK_MODEL)
    print("\n" + "=" * 40)
    print("=== EXECUTION & MONITOR REPORT (JSON) ===")
    print("=" * 40)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Save to file as well
    out_path = os.path.join(LOG_DIR, f"monitor_summary_{choice.lower()}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to: {out_path}")
