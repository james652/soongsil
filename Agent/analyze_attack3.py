import json
import chromadb
from openai import OpenAI
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

openai_client = OpenAI(api_key="")
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.01,  # 창의성 낮게
    openai_api_key=""
)

# =========================
# ChromaDB (없어도 동작)
# =========================
try:
    chroma_client = chromadb.HttpClient()
    collection = chroma_client.get_collection(name="papers")
    print(f"[ChromaDB] '{collection.name}' 컬렉션에 성공적으로 연결했습니다.")
    print("[ChromaDB] 문서 수:", collection.count())
except Exception as e:
    print(f"[경고] ChromaDB 연결 실패: {e}")
    print("poisoningpdf.py로 적재했는지 / 서버가 떠있는지 확인. (없어도 분석은 진행)")
    collection = None

def query_chroma_hint(query_text="Poisoning Attack", top_k=3):
    if collection is None:
        return "(ChromaDB 연결 실패)"
    try:
        results = collection.query(query_texts=[query_text], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "(ChromaDB에서 관련 내용을 찾지 못했습니다.)"
        return "\n---\n".join(f"- {doc}" for doc in docs)
    except Exception as e:
        return f"ChromaDB 쿼리 오류: {e}"

def query_chroma_multi(queries, top_k=2):
    if collection is None:
        return "(ChromaDB 연결 실패)"
    chunks = []
    for q in queries:
        try:
            r = collection.query(query_texts=[q], n_results=top_k)
            docs = r.get("documents", [[]])[0]
            for d in docs:
                chunks.append(f"- {d}")
        except Exception as e:
            chunks.append(f"(쿼리 오류: {q}, {e})")
    return "\n---\n".join(chunks) if chunks else "(관련 문서 없음)"

# =========================
# 로그 로드 / 전처리
# =========================
def load_accuracy_matrices(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    before = data["단계 1/5: 초기 모델 학습 (EWC)"]["accuracy_matrix"]
    after  = data["단계 5/5: 최종 평가"]["accuracy_matrix"]
    return before, after

def flatten_accuracy(matrix):
    # 마지막 행만 사용 (per-task)
    return [round(float(v), 4) for v in matrix[-1]]

def align_same_length(a, b):
    # 길이 불일치 시 긴 쪽 자름
    n = min(len(a), len(b))
    return a[:n], b[:n]

# =========================
# 로컬 휴리스틱
# =========================
def longest_monotone_decline(seq: List[float], min_drop=0.03) -> int:
    if not seq:
        return 0
    run = best = 0
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1] - min_drop:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best

def local_signals(before: List[float], after: List[float]) -> Dict[str, Any]:
    if not before or not after or len(before) != len(after):
        return {"error": "길이가 같은 before/after 리스트가 필요합니다."}

    n = len(before)
    mean_before = sum(before) / n
    mean_after  = sum(after) / n
    abs_drop = mean_before - mean_after
    rel_drop = abs_drop / (mean_before + 1e-8)

    diffs = [before[i] - after[i] for i in range(n)]
    max_drop = max(diffs)
    max_drop_task = diffs.index(max_drop)
    decline_run_after = longest_monotone_decline(after, min_drop=0.03)

    import statistics
    try:
        mu = statistics.mean(diffs); sd = statistics.pstdev(diffs)
    except statistics.StatisticsError:
        mu, sd = 0.0, 0.0
    threshold = mu + (sd if sd > 0 else 0.05)
    attacked_tasks = [i for i, d in enumerate(diffs) if d >= threshold]

    return {
        "len": n,
        "mean_before": round(mean_before, 4),
        "mean_after": round(mean_after, 4),
        "abs_drop": round(abs_drop, 4),
        "rel_drop": round(rel_drop, 4),
        "diffs": [round(x, 4) for x in diffs],
        "max_drop": round(max_drop, 4),
        "max_drop_task": max_drop_task,
        "decline_run_after": decline_run_after,
        "attacked_tasks_local": attacked_tasks
    }

# =========================
# 멀티턴을 위한 공용 히스토리 저장소
# =========================
_history: Dict[str, ChatMessageHistory] = {}
def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _history:
        _history[session_id] = ChatMessageHistory()
    return _history[session_id]

# =========================
# Step 1: 정황 판단 (다중 플래그) + 히스토리 기록
# =========================
detect_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 머신러닝 보안 전문가야. 정확도 변화와 로그를 바탕으로 공격 정황을 판단해. 반드시 **유효한 JSON 하나**로만 출력하고, 한국어로 답하라."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", """
[정확도 변화]
- 공격 전: {before_acc}
- 공격 후: {after_acc}

[출력 로그]
{previous_output_log}

Brainwash: 마지막 태스크 학습 이후 과거 태스크들의 정확도가 비정상적으로 급격히 하락(붕괴)하는 특징이 있어. 정황이 보이면 'brainwash_sign_detected': true 로 응답해.
Accumulative Poisoning: 여러 태스크/에폭/스트림을 거치며 정확도가 완만하지만 일관되게 누적 하락. 정황이 보이면 'accumulative_sign_detected': true.
Clean-label (Bullseye Polytope): 라벨은 정상으로 유지하나 미세 섭동으로 특정 타깃만 오분류. 정황이 보이면 'clean_label_sign_detected': true.
Label-flip: 일부 라벨을 뒤바꿔 전반 정확도 하락 + 특정 클래스 쌍 상호 오분류 증가. 정황이 보이면 'label_flip_sign_detected': true.
Backdoor/Trojan: 클린 acc 정상이나 트리거 입력에서 ASR 높음(클린↔트리거 괴리). 정황이 보이면 'backdoor_sign_detected': true.
Gradient-matching: 특정 타깃/구간에서 일관된 오분류 또는 학습 중 grad/loss 이상치. 정황이면 'gradient_matching_sign_detected': true.

JSON만 출력:
{{
  "brainwash_sign_detected": true or false,
  "accumulative_sign_detected": true or false,
  "clean_label_sign_detected": true or false,
  "label_flip_sign_detected": true or false,
  "backdoor_sign_detected": true or false,
  "gradient_matching_sign_detected": true or false,
  "reason": "<요약 근거>"
}}
""")
])

detect_chain = RunnableWithMessageHistory(
    detect_prompt | llm,
    lambda sid: get_history(sid),
    input_messages_key="detect_input",
    history_messages_key="chat_history"
)

def detect_signs(before_acc, after_acc, previous_output_log, session_id="attack-session"):
    inputs = {
        "before_acc": str(before_acc),
        "after_acc": str(after_acc),
        "previous_output_log": previous_output_log,
        "detect_input": "go"
    }
    cfg = {"configurable": {"session_id": session_id}}
    return detect_chain.invoke(inputs, config=cfg)

# =========================
# Step 2: 최종 분석 (4단계 결과 JSON) + 히스토리 이어받기
# =========================
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 머신러닝 보안 전문가야. 제공된 모든 정보를 종합해 **4단계 결과**를 반드시 유효한 JSON 하나로만 출력해"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", """
[정확도 변화]
- 공격 전: {before_acc}
- 공격 후: {after_acc}

[정황 판단(1단계) 결과]
{sign_result_json}

[연구 힌트(ChromaDB/논문 발췌)]
{chroma_hint}

[로컬 휴리스틱 계산값]
{local_signals}

[출력 로그]
{previous_output_log}

요구 JSON 스키마(반드시 이 키 전부 포함, JSON만 출력):
{{
  "is_attack": true or false,
  "why_attack": ["...", "..."],
  "is_poisoning": true or false or null,
  "why_poisoning": ["...", "..."],
  "attack_family": "<공격 타입 or '없음'>",
  "why_family": ["...", "..."],
  "attacked_tasks": [정수 리스트],
  "confidence": 0.0
}}
""")
])

conversation = RunnableWithMessageHistory(
    final_prompt | llm,
    lambda sid: get_history(sid),
    input_messages_key="question",
    history_messages_key="chat_history"
)

def analyze_attack(before_acc, after_acc, previous_output_log="", session_id="attack-session") -> Tuple[Any, Dict[str, Any]]:
    # 길이 자동 보정
    if len(before_acc) != len(after_acc):
        print(f"[경고] per-task 길이 불일치: before={len(before_acc)}, after={len(after_acc)} → 최소 길이에 맞춤")
        before_acc, after_acc = align_same_length(before_acc, after_acc)

    # (A) 1단계: 정황 판단 (히스토리에 기록됨)
    s1_raw = detect_signs(before_acc, after_acc, previous_output_log, session_id=session_id)
    try:
        sign_result = json.loads(s1_raw.content.strip().strip("```json").strip("```"))
    except Exception:
        print("[경고] 1단계 JSON 파싱 실패. 원문:")
        print(getattr(s1_raw, "content", str(s1_raw)))
        sign_result = {}

    print("\n=== [멀티턴 Step 1] 정황 판단 결과 ===")
    print(json.dumps(sign_result, ensure_ascii=False, indent=2))

    # (B) 힌트 수집: 감지된 플래그 기반으로 질의 다양화 (없으면 기본 문자열)
    queries = []
    if sign_result.get("brainwash_sign_detected"):
        queries.append("")
    if sign_result.get("accumulative_sign_detected"):
        queries.append("")
    if sign_result.get("label_flipping_sign_detected"):
        queries.append("")
    if not queries:
        queries = ["data poisoning attacks in continual learning"]
    chroma_hint = query_chroma_multi(queries, top_k=1)

    # 로컬 휴리스틱
    sig = local_signals(before_acc, after_acc)

    # (C) 2단계: 최종 분석 (같은 세션 → 1단계 히스토리 자동 반영)
    inputs = {
        "before_acc": str(before_acc),
        "after_acc": str(after_acc),
        "sign_result_json": json.dumps(sign_result, ensure_ascii=False),
        "chroma_hint": chroma_hint,
        "local_signals": json.dumps(sig, ensure_ascii=False),
        "previous_output_log": previous_output_log,
        "question": "최종 4단계 결과를 위 스키마로만 JSON 출력"
    }
    cfg = {"configurable": {"session_id": session_id}}
    s2_raw = conversation.invoke(inputs, config=cfg)

    # RAGAS 평가를 위한 컨텍스트 묶음 반환
    ctx_bundle = {
        "before_acc": before_acc,
        "after_acc": after_acc,
        "sign_result": sign_result,
        "chroma_hint": chroma_hint,
        "local_signals": sig,
        "previous_output_log": previous_output_log,
    }
    return s2_raw, ctx_bundle

# =========================
# 요약 텍스트 생성(사람 읽기용)
# =========================
def build_summary_text(final_json: Dict[str, Any]) -> str:
    parts = []
    parts.append(f"① 공격 여부: {'공격' if final_json.get('is_attack') else '정상'} (신뢰도 {final_json.get('confidence')})")
    if final_json.get("is_attack"):
        why = final_json.get("why_attack", [])
        if why:
            parts.append("② 공격 근거: " + " / ".join(why))
        ipo = final_json.get("is_poisoning")
        label = "Poisoning" if ipo is True else ("Evasion" if ipo is False else "판단 불가")
        parts.append(f"③ 왜 Poisoning인가: {label}")
        wp = final_json.get("why_poisoning", [])
        if wp:
            parts.append("   근거: " + " / ".join(wp))
        parts.append(f"④ 공격 타입: {final_json.get('attack_family', '없음')}")
        wf = final_json.get("why_family", [])
        if wf:
            parts.append("   타입 근거: " + " / ".join(wf))
        atk = final_json.get("attacked_tasks", [])
        parts.append(f"   공격 대상 태스크 후보: {atk}")
    return "\n".join(parts)

# =========================
# RAGAS 평가
# =========================

def load_ragas_metrics():
    try:
        # ragas 0.1.x 계열에서 주로 사용
        from ragas.metrics import faithfulness, answer_relevancy
        return faithfulness, answer_relevancy
    except Exception:
        # 일부 버전에서 answer_relevance 로 노출됨
        from ragas.metrics import faithfulness, answer_relevance
        return faithfulness, answer_relevance

def evaluate_summary_with_ragas(question: str, contexts: list[str], answer: str):
    """
    - pip install "ragas>=0.1.21,<0.2.0" datasets
    - metrics: faithfulness, answer_relevancy (또는 answer_relevance)
    """
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        faithfulness, ans_rel_metric = load_ragas_metrics()
    except Exception as e:
        print("[RAGAS] 패키지 미설치 또는 임포트 실패:", e)
        print('       -> pip install "ragas>=0.1.21,<0.2.0" datasets')
        return None

    try:
        ds = Dataset.from_dict({
            "question": [question],
            "contexts": [contexts],  # list[str]
            "answer":   [answer],
        })

        # LangChain LLM 인자 없이도 동작하도록 기본 호출
        # (필요시 ragas_evaluate(..., llm=llm)로 바꿔도 됨)
        result = ragas_evaluate(ds, metrics=[faithfulness, ans_rel_metric])

        # 점수 추출(컬럼명이 ragas 버전에 따라 다를 수 있어 유연 추출)
        scores = {}
        try:
            df = result.to_pandas()
            row = df.iloc[0]

            def pick(*names: str):
                cols = [c for c in df.columns]
                norm = lambda s: s.replace("_", "").replace("-", "").lower()
                for want in names:
                    for c in cols:
                        if norm(c) == norm(want):
                            return float(row[c])
                return None

            scores["faithfulness"] = pick("faithfulness")
            # 두 이름 중 존재하는 쪽을 채택
            scores["answer_relevancy"] = pick("answer_relevancy", "answer_relevance")

        except Exception:
            scores["faithfulness"] = None
            scores["answer_relevancy"] = None

        # 깔끔 출력 (\n 위주)
        print("\n[RAGAS] 평가 결과")
        print(f"Question:\n{question}")
        print("Contexts:")
        for i, c in enumerate(contexts, 1):
            print(f"- C{i}: {c}")
        print("Answer:")
        print(answer)
        print("Scores:")
        print(f"- faithfulness     : {scores.get('faithfulness')}")
        print(f"- answer_relevancy : {scores.get('answer_relevancy')}")
        return scores

    except Exception as e:
        print("[RAGAS] 평가 중 오류:", e)
        return None

# =========================
# main
# =========================
def main():
    try:
        before_mtx, after_mtx = load_accuracy_matrices("/home/jun/work/soongsil/Agent/analysis_log.json")
    except FileNotFoundError:
        print("[오류] analysis_log.json 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return
    except KeyError as e:
        print(f"[오류] analysis_log.json 구조가 예상과 다릅니다: {e}")
        return

    before_flat = flatten_accuracy(before_mtx)
    after_flat  = flatten_accuracy(after_mtx)

    print(f"\n[초기 데이터] Accuracy(per-task, 마지막 행)")
    print(f"- Before: {before_flat}")
    print(f"- After : {after_flat}")

    previous_output_log = "per-task 정확도 Before/After 비교로 4단계 멀티턴 판정 진행."

    # 멀티턴 파이프라인 실행 (같은 세션 유지)
    session_id = "attack-session"
    final_raw, ctx_bundle = analyze_attack(before_flat, after_flat, previous_output_log, session_id=session_id)

    print("\n=== [멀티턴 Step 2] 최종 결과(JSON) ===")
    try:
        final_json = json.loads(final_raw.content.strip().strip("```json").strip("```"))
        print(json.dumps(final_json, ensure_ascii=False, indent=2))
    except Exception:
        print("[경고] 최종 JSON 파싱 실패. 원문:")
        print(getattr(final_raw, "content", str(final_raw)))
        return

    # 사람 읽기용 요약
    print("\n[요약]")
    summary_text = build_summary_text(final_json)
    print(summary_text)

    # =========================
    # RAGAS 평가 실행
    # =========================
    # 질문 정의
    question_text = "주어진 로그와 힌트를 바탕으로 공격 여부·근거·poisoning 여부·공격을 정확히 설명해."

    # 컨텍스트 구성: 분석에 사용된 핵심 근거들을 문자열로 묶음
    contexts = [
        f"정확도 변화: Before={ctx_bundle['before_acc']}, After={ctx_bundle['after_acc']}",
        f"로컬 휴리스틱: {json.dumps(ctx_bundle['local_signals'], ensure_ascii=False)}",
        f"정황 판단(1단계): {json.dumps(ctx_bundle['sign_result'], ensure_ascii=False)}",
        f"연구 힌트: {ctx_bundle['chroma_hint']}",
        f"출력 로그/메모: {ctx_bundle['previous_output_log']}",
    ]

    _ = evaluate_summary_with_ragas(
        question=question_text,
        contexts=contexts,
        answer=summary_text
    )

if __name__ == "__main__":
    main()
