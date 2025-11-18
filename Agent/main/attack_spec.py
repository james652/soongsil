# attack_specs.py
# ----------------
# Standalone module that defines:
# - Constants: LOG_DIR, BASE_DIR
# - Data model: StepSpec
# - Pipeline builders: build_brainwash_specs, build_accumulative_specs
#
# This file is meant to be imported by your agent/runner (e.g., attack_agent.py).

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

import os
import glob
import json

# ------------------------------------------------------------------------------------
# Centralized paths
# ------------------------------------------------------------------------------------
LOG_DIR = "/home/jun/work/soongsil/Agent/logs"
BASE_DIR = "/home/jun/work"  # adjust if needed
_TOPK_DIR = os.path.join(LOG_DIR, "LLM_topk")


@dataclass
class StepSpec:
    title: str
    command: str
    log_path: str
    #forbidden_regex: List[str] = field(default_factory=list)
    expected_artifacts: List[str] = field(default_factory=list)  # glob patterns
    timeout_sec: int = 0  # 0 = no timeout
    analysis_log_path: Optional[str] = None
    analysis_prompt: Optional[str] = None
    rag_request: Optional[str] = None


# ------------------------------------------------------------------------------------
# Helpers to load Top-k terms saved by Analyzer/Runner
# ------------------------------------------------------------------------------------
def _read_json_safe(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _pick_latest(paths: List[str]) -> Optional[str]:
    if not paths:
        return None
    try:
        return max(paths, key=lambda p: os.path.getmtime(p))
    except Exception:
        # if mtime fails, just pick lexicographically last
        return sorted(paths)[-1]


def _collect_terms_from_file(obj: dict, limit: int = 20) -> str:
    # Accepted schema:
    # {
    #   "attack_terms": ["...", ...],
    #   "non_attack_terms": ["...", ...],
    #   "evidence_phrases": ["...", ...]  # optional
    # }
    atk = obj.get("attack_terms") or []
    non = obj.get("non_attack_terms") or []
    evd = obj.get("evidence_phrases") or []

    # flatten to short, comma-joined strings
    def _take(xs): return ", ".join([str(x) for x in xs[:limit] if isinstance(x, (str, int, float))])

    parts = []
    if atk:
        parts.append(f"topk_attack_terms: {_take(atk)}")
    if non:
        parts.append(f"topk_non_attack_terms: {_take(non)}")
    if evd:
        parts.append(f"topk_evidence: {_take(evd)}")
    return " | ".join(parts)


def _load_topk_for_step(step_key: str, limit: int = 20) -> str:
    """
    step_key 예: 'step1', 'step2', 'step3', 'step4'
    파일명 예: analysis_20251107_153012__step1.json
    """
    if not os.path.isdir(_TOPK_DIR):
        return ""

    # 우선 해당 step에 매칭되는 파일만 찾아본다
    PAT = os.path.join(_TOPK_DIR, f"*__{step_key}.json")
    cand = glob.glob(PAT)
    chosen = _pick_latest(cand)
    if chosen:
        obj = _read_json_safe(chosen)
        if isinstance(obj, dict):
            return _collect_terms_from_file(obj, limit=limit)

    # 없으면 전체 중 최신 하나라도 사용 (graceful fallback)
    all_json = glob.glob(os.path.join(_TOPK_DIR, "*.json"))
    chosen = _pick_latest(all_json)
    if chosen:
        obj = _read_json_safe(chosen)
        if isinstance(obj, dict):
            return _collect_terms_from_file(obj, limit=limit)

    return ""


def _concat_rag(base: str, addon: str) -> str:
    base = (base or "").strip()
    addon = (addon or "").strip()
    if not addon:
        return base
    if base:
        return f"{base} || {addon}"
    return addon


# ------------------------------------------------------------------------------------
# Spec builders
# ------------------------------------------------------------------------------------
def build_brainwash_specs() -> List[StepSpec]:
    base = BASE_DIR

    # 공통 베이스 쿼리
    base_query = (
        "brainwash continual learning poisoning ewc split_cifar100 cifar100 "
        "fisher diagonal catastrophic forgetting average accuracy BWT"
    )

    # 각 단계별로 직전 스텝의 LLM_topk를 불러와 rag_request에 주입
    # step1: 초기 학습이므로 topk 없음
    step1_topk = ""  # no prior step
    step2_topk = _load_topk_for_step("step1")
    step3_topk = _load_topk_for_step("step2")
    step4_topk = _load_topk_for_step("step3")

    return [
        StepSpec(
            title="단계 1/4: 초기 모델 학습 (EWC)",
            command=(
                "CUDA_VISIBLE_DEVICES=1 python /home/jun/work/soongsil/Brainwash/main_baselines.py "
                "--experiment split_cifar100 --approach ewc --lasttask 9 --tasknum 10 --nepochs 20 "
                "--batch-size 16 --lamb 500000 --lamb_emp 100 --clip 100. --lr 0.01"
            ),
            log_path=f"{LOG_DIR}/step1.log",
            expected_artifacts=[f"{base}/**/*.pkl"],
            analysis_prompt=(
                "Analyze the initial EWC training stage. "
                "Summarize training metrics, "
                "detect any irregularities (NaNs, divergence, or instability), "
                "and describe what these suggest about the model's baseline behavior. "
                f"Please analyze the following log file.: {LOG_DIR}/step1.log"
            ),
            rag_request=_concat_rag(base_query, step1_topk),
        ),
        StepSpec(
            title="단계 2/4: 수행",
            command=(
                "python /home/jun/work/soongsil/Brainwash/main_inv.py "
                "--pretrained_model_add=/home/jun/work/soongsil/Brainwash/"
                "ewc_lamb_500000.0__model_type_resnet_dataset_split_cifar100_class_num_10_bs_16_lr_0.01_"
                "n_epochs_20__model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_32__last_task_9____"
                "optim_name_sgd.pkl --num_samples=128 --save_dir=/home/jun/work/soongsil/Brainwash/output "
                "--task_lst=0,1,2,3,4,5,6,7,8 --save_every=1000 --batch_reg --init_acc --n_iters=10000"
            ),
            log_path=f"{LOG_DIR}/step2.log",
            expected_artifacts=[f"{base}/continual-learning-baselines/test_jun/Brainwash/model_inv/**/*.npz"],
            analysis_prompt=(
                "Summarize training metrics, "
                "detect any irregularities (NaNs, divergence, or instability), "
                "and describe what these suggest about the model's baseline behavior. "
                f"Please analyze the following log file.: {LOG_DIR}/step2.log"
            ),
            rag_request=_concat_rag(base_query, step2_topk),
        ),
        StepSpec(
            title="단계 3/4: 적용",
            command=(
                "CUDA_VISIBLE_DEVICES=0 python /home/jun/work/soongsil/Brainwash/main_brainwash.py "
                "--extra_desc=reckless_test "
                "--pretrained_model_add=/home/jun/work/soongsil/Brainwash/"
                "ewc_lamb_500000.0__model_type_resnet_dataset_split_cifar100_class_num_10_bs_16_lr_0.01_"
                "n_epochs_20__model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_32__last_task_9____"
                "optim_name_sgd.pkl --mode='reckless' --target_task_for_eval=0 --delta=0.3 --seed=0 "
                "--eval_every=10 --distill_folder=/home/jun/work/soongsil/Brainwash/output --init_acc "
                "--noise_norm=inf --cont_learner_lr=0.001 --n_epochs=50 --save_every=100 "
            ),
            log_path=f"{LOG_DIR}/step3.log",
            expected_artifacts=[f"{base}/**/*.pkl"],
            analysis_prompt=(
                "Explain performance changes, instability, or potential evidence of model collapse. "
                "Summarize training metrics, "
                "detect any irregularities (NaNs, divergence, or instability), "
                "and describe what these suggest about the model's baseline behavior. "
                f"Please analyze the following log file.: {LOG_DIR}/step3.log"
            ),
            rag_request=_concat_rag(base_query, step3_topk),
        ),
        StepSpec(
            title="단계 4/4: 최종 평가",
            command=(
                "CUDA_VISIBLE_DEVICES=0 python /home/jun/work/soongsil/Brainwash/main_baselines.py "
                "--experiment split_cifar100 --approach ewc --lasttask 9 --tasknum 10 --nepochs 20 "
                "--batch-size 16 --lr 0.01 --clip 100. --lamb 500000 --lamb_emp 100 "
                "--checkpoint /home/jun/work/soongsil/Agent/"
                "noise_ewc_wcur_1.0_cautious_test__delta_0.3_dataset_split_cifar100_target_task_0_"
                "attacked_task_9_noise_optim_lr_0.005__n_iters_1_n_epochs_5000_seed_0_mode_cautious__"
                "w_cur_1.0___min_acc_target_9.pkl --init_acc --addnoise"
            ),
            log_path=f"{LOG_DIR}/step4.log",
            expected_artifacts=[f"{base}/**/acc_mat_*.npy"],
            analysis_prompt=(
                "Based on the analyses from steps 1–3, identify the most probable attack type. "
                "Choose exactly one from the following: Brainwash, Accumulative Attack, or Label Flipping. "
                "Do not hedge with expressions like 'similar to' or 'Brainwash-like' — be decisive. "
                "Provide supporting numerical evidence (accuracy, BWT, or collapse pattern) from the logs. "
                f"Please analyze the following log file. {LOG_DIR}/step4.log"
            ),
            rag_request=_concat_rag(base_query, step4_topk),
        ),
    ]

# attack_spec.py

def build_accumulative_specs() -> List[StepSpec]:
    base_query = "accumulative attack online training PGD Linf epsilon trigger collapse"

    # 실행 커맨드(줄바꿈 시 공백 빠지지 않게 각 라인 끝에 공백 포함)
    step1_cmd = (
        "python /home/jun/work/soongsil/PoisoningAttack/AccumulativeAttack/online_accu_train.py "
        "--batch_size 100 --epoch 100 --test_batch_size 500 --log_name log_test_online.txt "
        "--resume /home/jun/work/soongsil/PoisoningAttack/AccumulativeAttack/checkpoints_base_bn --use_bn --model_name epoch40.pth "
        "--mode 'eval' --onlinemode 'train' --lr 1e-1 --momentum 0.9 "
        "--beta 1. --only_reg --threshold 0.18 --use_advtrigger "
        )

    # 두 번째 스텝에 주입할 Top-K (첫 스텝 결과에서 뽑힌 것)
    step2_topk = _load_topk_for_step("step1")

    return [
        # 1) 온라인 학습 실행 스텝 (로그 생성)
        StepSpec(
            title="Accumulative: Online Train",
            command=step1_cmd,
            log_path=f"{LOG_DIR}/accumulative_train.log",
            expected_artifacts=[f"{BASE_DIR}/**/*.pth", f"{BASE_DIR}/**/*.pt", f"{BASE_DIR}/**/*.pkl"],
            analysis_prompt=(
                "Summarize online training: report accuracy/loss trends per epoch, "
                "note when threshold is breached, and describe any progressive degradation."
            ),
            rag_request=_concat_rag(base_query, ""),  # step1은 외부 주입 없음
        ),

        # 2) 분석 전용 스텝 (실행 없이, 1번 스텝 로그를 읽어 분석)
        StepSpec(
            title="Accumulative: Analyse log",
            command="",  # 실행 없음 → 모니터/애널라이저가 log_path만 읽어 요약
            log_path=f"{LOG_DIR}/accumulative_train.log",
            analysis_prompt=(
                "Based on the analyses from step 1, determine what type of attack (if any) occurred. "
                "If it matches a known method, specify which paper's technique it resembles. "
                "If it is not an attack, explain why. "
                "Finally, concisely summarize the observed behavior and key metrics. "
                "Choose exactly one: Brainwash, Accumulative Attack, Label Flipping, or No Attack."
            ),
            # ← 여기로 step1 Top-K 자동 주입
            rag_request=_concat_rag(base_query + " final evaluation analysis", step2_topk),
        ),
    ]




def build_analyze_brainwash_specs() -> List[StepSpec]:
    base_query = "average accuracy"
    # 분석 전용: 여기서는 Top-k를 꼭 붙일 필요는 없지만, 원한다면 동일 방식으로 주입 가능
    return [
        StepSpec(
            title="분석: 초기 모델 학습 (EWC) 로그",
            command="",
            log_path=f"{LOG_DIR}/step1.log",
            analysis_prompt=(
                "Analyze the initial EWC training stage. "
                "Summarize training metrics (loss, accuracy, BWT, etc.), "
                "detect any irregularities (NaNs, divergence, or instability), "
                "and describe what these suggest about the model's baseline behavior."
            ),
            rag_request=_concat_rag(base_query + " explain EWC training stability and baseline performance",
                                    _load_topk_for_step("step1")),
        ),
        StepSpec(
            title="분석: 역공격 (Inversion Attack) 로그",
            command="",
            log_path=f"{LOG_DIR}/step2.log",
            analysis_prompt=(
                "Summarize progress, identify any unusual patterns or warnings, "
                "and explain how the attack might influence the model's internal representations."
            ),
            rag_request=_concat_rag(base_query + " inversion attack effect on representations",
                                    _load_topk_for_step("step2")),
        ),
        StepSpec(
            title="분석: Brainwash Reckless Mode 실행 로그",
            command="",
            log_path=f"{LOG_DIR}/step3.log",
            analysis_prompt=(
                "Summarize performance changes, instability, or potential evidence of model collapse. "
            ),
            rag_request=_concat_rag(base_query + " brainwash reckless mode effect analysis",
                                    _load_topk_for_step("step3")),
        ),
        StepSpec(
            title="분석: 최종 평가 로그",
            command="",
            log_path=f"{LOG_DIR}/step4.log",
            analysis_prompt=(
                "Based on the analyses from steps 1–3, identify the most probable attack type. "
                "Choose exactly one from the following: Brainwash, Accumulative Attack, or Label Flipping. "
                "Do not hedge with expressions like 'similar to' or 'Brainwash-like' — be decisive. "
                "Provide supporting numerical evidence (accuracy, BWT, or collapse pattern) from the logs."
            ),
            rag_request=_concat_rag(base_query + " catastrophic forgetting", _load_topk_for_step("step3")),
        ),
    ]


def build_analyze_accumulative_specs() -> List[StepSpec]:
    """
    Accumulative Attack 로그를 RAG + GPT로 분석하기 위한 스펙 빌더.
    필요에 따라 로그 파일 경로를 추가/수정해도 됨.
    """
    base_query = (
        "You are a machine learning results analysis expert."
        "accuracy degradation detection rate"
    )
    return [
        StepSpec(
            title="Analyse log",
            command="",
            log_path=f"{LOG_DIR}/accumulative_train.log",
            analysis_prompt=(
                "Summarize the progress, outputs, and warning signals from the following log concisely."
                "Summarize concisely the effects of the beta and normalization settings, and the signs of collapse or stability, with supporting evidence."
            ),
            rag_request=_concat_rag(base_query + " online training log interpretation",
                                    _load_topk_for_step("train")),
        ),
        StepSpec(
            title="Analyse log",
            command="",
            log_path=f"{LOG_DIR}/accumulative_train.log",
            analysis_prompt=(
                "Based on the analyses from steps 1-2, "
                "summarize the overall behavioral pattern and determine what type of attack (if any) occurred. "
                "If it matches a known method, specify which paper's technique it resembles. "
                "If it is not an attack, explain why. "
                "Finally, describe the results obtained so far for steps 1 through 2 in a concise summary."
                "Choose exactly one from the following: Brainwash, Accumulative Attack, or Label Flipping. "
            ),
            rag_request=_concat_rag(base_query + " final evaluation analysis",
                                    _load_topk_for_step("train")),
        ),
    ]
