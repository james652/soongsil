"""
Execution Monitor (Module-integrated)
-------------------------------------
- Streams stdout/stderr per step
- Saves logs to central dir
- Local regex + artifact checks
- (Optional) GPT audit

Now imports pipeline specs from: /home/jun/work/soongsil/Agent/module/attack_spec.py
"""
from __future__ import annotations
import os, time, json, re, glob, subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
ANALYZE_DIR = "/home/jun/work/soongsil/Agent/analyze_log"
# ---- Import shared specs/types from external module ----
#   File path: /home/jun/work/soongsil/Agent/module/attack_spec.py
from module.attack_spec import (
    LOG_DIR,  # centralized log dir
    StepSpec,  # dataclass for step specification
    build_brainwash_specs,
    build_accumulative_specs,
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
    exit_code: int
    duration_sec: float
    log_path: str
    matched_required: Dict[str, bool]
    matched_forbidden: Dict[str, bool]
    found_artifacts: Dict[str, List[str]]
    gpt_summary: Optional[str] = None
    gpt_flags: Optional[Dict[str, Any]] = None

# ======================================================
# CORE: run and stream
# ======================================================

def run_and_stream(spec: StepSpec) -> StepResult:
    t0 = time.time()
    os.makedirs(os.path.dirname(spec.log_path) or ".", exist_ok=True)

    # compile regex (case-insensitive)
    required_compiled = [(pat, re.compile(pat, re.IGNORECASE)) for pat in spec.required_regex]
    forbidden_compiled = [(pat, re.compile(pat, re.IGNORECASE)) for pat in spec.forbidden_regex]
    matched_required = {pat: False for pat, _ in required_compiled}
    matched_forbidden = {pat: False for pat, _ in forbidden_compiled}

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
        last_lines: List[str] = []
        while True:
            if spec.timeout_sec and (time.time() - t0) > spec.timeout_sec:
                proc.kill()
                line = f"[MONITOR] Timeout reached ({spec.timeout_sec}s), process killed.\n"
                print(line, end="")
                logf.write(line)
                break
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            # Echo to console & log
            print(line, end="")
            logf.write(line)

            # Keep short buffer for GPT/evidence
            last_lines.append(line)
            if len(last_lines) > 5000:
                last_lines.pop(0)

            # Regex checks
            for pat, creg in required_compiled:
                if not matched_required[pat] and creg.search(line):
                    matched_required[pat] = True
            for pat, creg in forbidden_compiled:
                if creg.search(line):
                    matched_forbidden[pat] = True

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
        matched_required=matched_required,
        matched_forbidden=matched_forbidden,
        found_artifacts=found_artifacts,
    )

# ======================================================
# Local verification summary
# ======================================================

def summarize_local(result: StepResult) -> str:
    ok_req = all(result.matched_required.values()) if result.matched_required else True
    ok_forbid = not any(result.matched_forbidden.values())
    art_ok = any(len(v) > 0 for v in result.found_artifacts.values()) if result.found_artifacts else True
    verdict = "PASS" if (result.exit_code == 0 and ok_req and ok_forbid and art_ok) else "CHECK"

    lines = [
        f"[{verdict}] exit={result.exit_code} time={result.duration_sec:.1f}s log={result.log_path}",
        f"  required: {result.matched_required}",
        f"  forbidden: {result.matched_forbidden}",
        f"  artifacts: {{k: len(v) for k,v in result.found_artifacts.items()}}",
    ]
    return "\n".join(lines)

# ======================================================
# GPT verification (optional)
# ======================================================

def gpt_verify_step(step_log_path: str, step_title: str, model: str = "gpt-4o") -> Dict[str, Any]:
    if not _OPENAI_AVAILABLE:
        return {"error": "openai lib not available"}

    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_APIKEY")
    )
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)

    try:
        with open(step_log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_text = f.read()[-200000:]
    except Exception as e:
        # 파일 문제 시에도 summary는 채워서 돌려줌
        return {
            "step_executed": False,
            "errors": [f"read error: {e}"],
            "warnings": [],
            "evidence": [],
            "summary": f"로그 파일({step_log_path})을 읽지 못했습니다: {e}"
        }

    # 프롬프트 강화: 정확히 이 5개 키만 JSON으로
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
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return valid JSON with exactly the keys: step_executed, errors, warnings, evidence, summary."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1200,
        )
        text = resp.choices[0].message.content
        data = json.loads(text)
    except Exception as e:
        # ✅ 폴백 요약(간단 규칙 기반) — 절대 빈 summary가 되지 않도록
        last_lines = [ln for ln in log_text.strip().splitlines()[-10:]]
        summary_guess = "로그 말미:\n" + "\n".join(last_lines[:5]) if last_lines else "요약 불가(모델 응답 파싱 실패)"
        data = {
            "step_executed": None,
            "errors": [f"gpt_parse_error: {e}"],
            "warnings": [],
            "evidence": last_lines[:3] if last_lines else [],
            "summary": summary_guess
        }

    # 누락 방지: 모든 키 채움
    for k, default in [
        ("step_executed", None),
        ("errors", []),
        ("warnings", []),
        ("evidence", []),
        ("summary", "요약 없음"),
    ]:
        if k not in data or data[k] is None:
            data[k] = default

    return data


# ======================================================
# Build specs by name (modular)
# ======================================================

def get_specs_by_name(name: str) -> List[StepSpec]:
    key = name.strip().lower()
    if key in ("brainwash", "bw"):
        return build_brainwash_specs()
    if key in ("accumulative", "accu", "acc"):
        return build_accumulative_specs()
    raise SystemExit(f"Unknown pipeline: {name}")

# ======================================================
# Orchestrator
# ======================================================

def monitor_pipeline(specs: List[StepSpec], use_gpt: bool = True, gpt_model: str = "gpt-4o") -> List[StepResult]:
    results: List[StepResult] = []
    for spec in specs:
        print("\n" + "="*80)
        print(f"[RUN] {spec.title}")
        res = run_and_stream(spec)
        print("-"*80)
        print(summarize_local(res))

        if use_gpt:
            g = gpt_verify_step(spec.log_path, spec.title, model=gpt_model)
            res.gpt_flags = g if isinstance(g, dict) else {"raw": str(g)}
            res.gpt_summary = (g or {}).get("summary") if isinstance(g, dict) else None
            print("[GPT]", json.dumps(g, ensure_ascii=False)[:1000])

        results.append(res)
    return results

# ======================================================
# Helper: build overall report + print JSON to stdout
# ======================================================

def _step_local_verdict(s: StepResult) -> bool:
    ok_req = all(s.matched_required.values()) if s.matched_required else True
    ok_forbid = not any(s.matched_forbidden.values())
    art_ok = True if not s.found_artifacts else any(len(v) > 0 for v in s.found_artifacts.values())
    return (s.exit_code == 0) and ok_req and ok_forbid and art_ok


def build_report(steps: List[StepResult]) -> Dict[str, Any]:
    report_steps = []
    all_pass = True
    for s in steps:
        local_pass = _step_local_verdict(s)
        all_pass = all_pass and local_pass
        report_steps.append({
            "title": s.title,
            "exit_code": s.exit_code,
            "duration_sec": round(s.duration_sec, 1),
            "log_path": s.log_path,
            "local_pass": local_pass,
            "matched_required": s.matched_required,
            "matched_forbidden": s.matched_forbidden,
            "found_artifacts": {k: len(v) for k, v in s.found_artifacts.items()},
            "gpt": s.gpt_flags or {},
        })
    return {
        "overall": "PASS" if all_pass else "CHECK",
        "steps": report_steps,
    }

def analyze_logs_for_choice(choice: str) -> List[Dict[str, Any]]:
    """
    모니터 실행이 모두 끝난 뒤, 선택한 파이프라인의 로그 파일들을 GPT로 분석하여
    /home/jun/work/soongsil/Agent/analyze_log 아래에 저장한다.
    """
    os.makedirs(ANALYZE_DIR, exist_ok=True)
    key = (choice or "").strip().lower()

    if key in ("brainwash", "bw"):
        targets = [
            ("단계 1/4: 초기 모델 학습 (EWC)",      f"{LOG_DIR}/step1_train_ewc.log"),
            ("단계 2/4: Inverse Attack 수행",       f"{LOG_DIR}/step2_inversion.log"),
            ("단계 3/4: Brainwash (Reckless)",      f"{LOG_DIR}/step3_brainwash_reckless.log"),
            ("단계 4/4: 최종 평가",                 f"{LOG_DIR}/step4_eval.log"),
        ]
    elif key in ("accumulative", "accu", "acc"):
        targets = [
            ("Accumulative: Online Train",          f"{LOG_DIR}/accumulative_train.log"),
        ]
    else:
        print(f"[POST-GPT] 알 수 없는 choice='{choice}', 스킵.")
        return []

    results: List[Dict[str, Any]] = []
    for title, log_path in targets:
        if not os.path.exists(log_path):
            print(f"[POST-GPT] 로그 없음: {log_path}")
            continue

        print(f"[POST-GPT] 분석 시작: {title} -> {log_path}")
        g = gpt_verify_step(log_path, title, model="gpt-4o")
        # 파일명: gpt_analysis_<log-basename>.json
        base = os.path.splitext(os.path.basename(log_path))[0]
        out_path = os.path.join(ANALYZE_DIR, f"gpt_analysis_{_safe_name(base)}.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(g, f, ensure_ascii=False, indent=2)

        print(f"[POST-GPT] 분석 저장: {out_path}")
        print(f"[POST-GPT] summary: {g.get('summary', '(none)')}")
        results.append({"title": title, "log_path": log_path, "out_path": out_path, "gpt": g})
    return results

# ======================================================
# CLI runner
# ======================================================
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    choice = input("어떤 프로그램을 실행할까요? (Brainwash / Accumulative) [Brainwash]: ").strip() or "Brainwash"
    specs = get_specs_by_name(choice)

    out = monitor_pipeline(specs, use_gpt=True)
    report = build_report(out)

    print("\n=== EXECUTION & MONITOR REPORT (JSON) ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Save to file as well
    out_path = f"{LOG_DIR}/monitor_summary_{choice.lower()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved {out_path}")

    try:
        analyze_logs_for_choice(choice)
    
    except Exception as e:
        print(f"[POST-GPT] 분석 중 오류: {e}")
