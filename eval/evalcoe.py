import json
import os
import argparse
import time
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区 =================
client = OpenAI(
    api_key="xxxxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=120.0
)

# 裁判模型
JUDGE_MODEL = "qwen3.5-plus"

# 文件路径配置 (注意：GT 路径适配到 eval 文件夹)
GT_JSONL = "label.jsonl"
PRED_JSONL = "predictions_gpt-5-mini.jsonl"
OUTPUT_EVAL_JSONL = "evaluation_gpt-5-mini.jsonl"

# ==== 指标加权权重配置 ====
WEIGHTS = {
    "metric1_scene": 0.4,       
    "metric2_emotion": 0.3,     
    "metric3_class": 0.3        
}

# ==== 四模态权重配置 ====
MODALITY_WEIGHTS = {
    "face": 0.25,   
    "body": 0.25,   
    "scene": 0.25,  
    "audio": 0.25   
}

MAX_WORKERS = 10
MAX_RETRIES = 3
RETRY_DELAY = 2.0
# ==========================================

def get_judge_system_prompt():
    return """You are an expert evaluator for Multimodal Emotion Recognition Chain-of-Thought (CoT).
Your task is to evaluate a Test Model's Predicted CoT against a Ground Truth (GT) CoT.

Modality Availability & Strict Hallucination Rules
The dataset has 7 Cases determining available modalities.
CRITICAL RULE: You must severely penalize "Modality Hallucination". If the Test Model describes details of a modality that is missing or heavily blurred, its score for that modality must be 0.

Case 1 (FM): Face, Body, Scene, Audio (All available).
Case 2 (FDM): Face (details missing but structures usable), Body, Scene, Audio.
Case 3 (FSM): Body, Scene, Audio. (NO Face available. Score 0 if hallucinated).
Case 4 (VMM): Audio ONLY. (NO Visuals available. Score 0 for face/body/scene if hallucinated).
Case 5 (FDAM): Face (details missing but usable), Body, Scene. (NO Audio. Score 0 if hallucinated).
Case 6 (FSAM): Body, Scene. (NO Face, NO Audio. Score 0 if hallucinated).
Case 7 (AMM): Face, Body, Scene. (NO Audio. Score 0 if hallucinated).

Evaluation Metrics (Per Modality)
For EACH available modality, evaluate TWO aspects:
- Scene Understanding (0-10): How accurately the model extracts objective cues from this modality.
- Emotional Analysis (0-10): How accurately the model maps the extracted cues from this modality to emotional reasoning.

Output Format
You MUST strictly output a JSON object containing ONLY the scores. Do NOT output any explanations.
If a modality is not available in the Case, set ALL its scores to null.

{
    "scene_understanding": {
        "face": <int 0-10 or null>,
        "body": <int 0-10 or null>,
        "scene": <int 0-10 or null>,
        "audio": <int 0-10 or null>
    },
    "emotional_analysis": {
        "face": <int 0-10 or null>,
        "body": <int 0-10 or null>,
        "scene": <int 0-10 or null>,
        "audio": <int 0-10 or null>
    }
}"""


def call_judge(case_id, gt_cot, pred_cot, retry_count=0):
    user_prompt = f"""### Inputs
Case ID: {case_id}
Ground Truth CoT:
{gt_cot}
Predicted CoT:
{pred_cot}
"""
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": get_judge_system_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=300
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        error_msg = str(e)
        if "DataInspectionFailed" in error_msg or "400" in error_msg:
            print(f"\r🚫 触发安全拦截，放弃重试 (跳过): {error_msg[:100]}...") 
            return {"error": "DataInspectionFailed"}

        if retry_count < MAX_RETRIES:
            print(f"\r⚠️ 裁判模型调用失败 (重试 {retry_count + 1}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_DELAY * (retry_count + 1))
            return call_judge(case_id, gt_cot, pred_cot, retry_count + 1)
        else:
            return {"error": "MaxRetriesExceeded"}

def get_valid_mods(case_id):
    """根据 Case ID 映射有效的模态"""
    mapping = {
        1: ["face", "body", "scene", "audio"],
        2: ["face", "body", "scene", "audio"],
        3: ["body", "scene", "audio"],
        4: ["audio"],
        5: ["face", "body", "scene"],
        6: ["body", "scene"],
        7: ["face", "scene", "body"]
    }
    return mapping.get(int(case_id), ["face", "body", "scene", "audio"])

def calculate_metrics(judge_res, true_label, pred_label, case_id):
    """严格按照 Case 的有效模态 (valid_mods) 计算 Metric 1-4"""
    valid_mods = get_valid_mods(case_id)
    
    # 1. Metric 1: 场景理解 (仅计算有效模态)
    m1_num, m1_den = 0, 0
    scene_scores = judge_res.get("scene_understanding", {})
    for mod in valid_mods:
        score = scene_scores.get(mod)
        if score is not None:
            try:
                m1_num += float(score) * MODALITY_WEIGHTS[mod]
                m1_den += MODALITY_WEIGHTS[mod]
            except: pass
    metric1_scene = (m1_num / m1_den) if m1_den > 0 else 0

    # 2. Metric 2: 情感分析 (仅计算有效模态)
    m2_num, m2_den = 0, 0
    emotion_scores = judge_res.get("emotional_analysis", {})
    for mod in valid_mods:
        score = emotion_scores.get(mod)
        if score is not None:
            try:
                m2_num += float(score) * MODALITY_WEIGHTS[mod]
                m2_den += MODALITY_WEIGHTS[mod]
            except: pass
    metric2_emotion = (m2_num / m2_den) if m2_den > 0 else 0

    # 3. Metric 3: 分类结果正确性
    metric3_class = 10 if str(true_label).strip().lower() == str(pred_label).strip().lower() else 0

    # 4. Metric 4: 整体思维链得分
    metric4_overall = (
        metric1_scene * WEIGHTS["metric1_scene"] +
        metric2_emotion * WEIGHTS["metric2_emotion"] +
        metric3_class * WEIGHTS["metric3_class"]
    )

    return {
        "metric1_scene": round(metric1_scene, 2),
        "metric2_emotion": round(metric2_emotion, 2),
        "metric3_class": metric3_class,
        "metric4_overall": round(metric4_overall, 2),
        "modality_scores": {
            "scene_understanding": scene_scores,
            "emotional_analysis": emotion_scores
        },
        "raw_judge_scores": judge_res
    }

def process_single_line(line, gt_dict, evaluated_vids):
    if not line.strip(): return None
    try:
        pred_item = json.loads(line.strip())
    except json.JSONDecodeError: return None

    vid = pred_item.get('video_id')
    if vid in evaluated_vids: return None

    gt_cot = gt_dict.get(vid)
    pred_cot = pred_item.get('pred_cot', '')
    if not gt_cot or not pred_cot: return None

    # 获取统一的 Case ID
    case_id = pred_item.get('case', 1) 
    true_label = pred_item.get('emotion', '')
    pred_label = pred_item.get('pred_label', '')

    judge_output = call_judge(case_id, gt_cot, pred_cot)

    if judge_output:
        if "error" in judge_output:
            return "ERROR", vid, judge_output["error"]
            
        metrics = calculate_metrics(judge_output, true_label, pred_label, case_id)
        pred_item['evaluation_metrics'] = metrics
        return "SUCCESS", pred_item, metrics
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default=PRED_JSONL)
    parser.add_argument("--gt_file", type=str, default=GT_JSONL)
    parser.add_argument("--out_file", type=str, default=OUTPUT_EVAL_JSONL)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    
    gt_dict = {}
    print("Loading Ground Truth...")
    with open(args.gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                gt_dict[item['video_id']] = item.get('cot')
                
    evaluated_vids = set()
    if os.path.exists(args.out_file):
        with open(args.out_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): evaluated_vids.add(json.loads(line).get('video_id'))

    with open(args.pred_file, 'r', encoding='utf-8') as f:
        pred_lines = f.readlines() 

    remaining_lines = [l for l in pred_lines if json.loads(l).get('video_id') not in evaluated_vids]

    sum_m1, sum_m2, sum_m3, sum_m4, valid_count = 0, 0, 0, 0, 0
    if os.path.exists(args.out_file):
        with open(args.out_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    metrics = json.loads(line.strip()).get('evaluation_metrics', {})
                    if metrics:
                        sum_m1 += metrics.get("metric1_scene", 0)
                        sum_m2 += metrics.get("metric2_emotion", 0)
                        sum_m3 += metrics.get("metric3_class", 0)
                        sum_m4 += metrics.get("metric4_overall", 0)
                        valid_count += 1
                except: pass

    print(f"🚀 Starting evaluation using Judge: {JUDGE_MODEL} (Workers: {args.workers})")

    with open(args.out_file, 'a', encoding='utf-8') as out_f, \
         open("error_samples_log_gpt-5-mini.jsonl", 'a', encoding='utf-8') as err_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_line, line, gt_dict, evaluated_vids): line for line in remaining_lines}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating CoTs"):
                result = future.result()
                if result:
                    if result[0] == "SUCCESS":
                        _, pred_item, metrics = result
                        sum_m1 += metrics["metric1_scene"]
                        sum_m2 += metrics["metric2_emotion"]
                        sum_m3 += metrics["metric3_class"]
                        sum_m4 += metrics["metric4_overall"]
                        valid_count += 1
                        out_f.write(json.dumps(pred_item, ensure_ascii=False) + '\n')
                        out_f.flush()
                    elif result[0] == "ERROR":
                        _, vid, error_type = result
                        err_f.write(json.dumps({"video_id": vid, "error_type": error_type}, ensure_ascii=False) + '\n')
                        err_f.flush()

    if valid_count > 0:
        print("\n" + "= "*30)
        print("     Chain-of-Thought Evaluation Report     ")
        print("= "*30)
        print(f"Total Evaluated Samples: {valid_count}")
        print(f"Metric 1 (Scene)   : {sum_m1/valid_count:.2f} / 10")
        print(f"Metric 2 (Emotion) : {sum_m2/valid_count:.2f} / 10")
        print(f"Metric 3 (Acc)     : {sum_m3/valid_count:.2f} / 10")
        print(f"Metric 4 (Overall) : {sum_m4/valid_count:.2f} / 10")

if __name__ == "__main__":
    main()