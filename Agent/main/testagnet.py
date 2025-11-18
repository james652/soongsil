#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, re, argparse

# (필수) pip install openai
try:
    from openai import OpenAI
except Exception as e:
    print(f"[ERR] openai 라이브러리를 찾을 수 없습니다. pip install openai\n{e}")
    sys.exit(1)

def safe_name(s: str) -> str:
    return (re.sub(r"[^\w\-]+", "_", s.strip()) or "log")[:80]

def read_tail(path: str, tail_chars: int = 200_000) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        t = f.read()
    return t[-tail_chars:]

def analyze_with_gpt(log_path: str, step_title: str, model: str = "gpt-4o") -> dict:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")
    client = OpenAI(api_key=api_key)

    text = read_tail(log_path)

    # 요청하신 톤: "이 로그 파일을 확인해서 분석해 주세요"
    prompt = f"""
아래 경로의 로그 파일을 확인해서 분석해 주세요: {log_path}

당신은 머신러닝 파이프라인 감시 전문가입니다.
이건 어떤 데이터로 학습되었고 어떤 모델이 사용되었는지 알려주세요 
어떤 공격들에 많이 활용되었는지도 설명해주세요 

다음 항목을 JSON으로만 반환하세요:
{{
  "step_executed": true/false,        // 학습이 실제로 수행/진행되었는지
  "errors": ["..."],                  // 에러 라인 요약(없으면 빈 배열)
  "warnings": ["..."],                // 경고 라인 요약(없으면 빈 배열)
  "evidence": ["대표 근거 라인 3~5개"],  // Epoch/Accuracy/acc_mat/저장 관련 등
  "summary": "길게 작성"
}}

[로그 본문: {step_title}]
-----
{text}
-----
"""

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a strict, concise pipeline auditor. Always return valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=1200,
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except Exception:
        # 모델이 JSON이 아닌 걸 돌려줄 드문 경우 대비
        return {"raw": raw}

def main():
    parser = argparse.ArgumentParser(description="Analyze a log file with GPT (no execution).")
    parser.add_argument("--log", default="/home/jun/work/soongsil/Agent/logs/step1_train_ewc.log",
                        help="분석할 로그 파일 경로")
    parser.add_argument("--title", default="(SMOKE) 단계 1/4: 초기 모델 학습 (더미)",
                        help="스텝 제목(프롬프트 표시용)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI 모델 이름")
    args = parser.parse_args()

    log_path = args.log
    if not os.path.exists(log_path):
        print(f"[ERR] 로그 파일이 없습니다: {log_path}")
        sys.exit(2)

    try:
        result = analyze_with_gpt(log_path, args.title, model=args.model)
    except Exception as e:
        print(f"[ERR] GPT 분석 실패: {e}")
        sys.exit(3)

    # 결과 저장: 로그 파일과 같은 디렉토리에 JSON 저장
    out_dir = os.path.dirname(log_path) or "."
    out_name = f"gpt_analysis_{safe_name(os.path.splitext(os.path.basename(log_path))[0])}.json"
    out_path = os.path.join(out_dir, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] 분석 결과 저장: {out_path}")
    print("summary:", result.get("summary", str(result)[:200]))

if __name__ == "__main__":
    main()
