import json
import os
import cv2
import base64
import argparse
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import defaultdict

# ================= 配置区 =================
client = OpenAI(api_key="xxxxx",
               base_url="https://dashscope.aliyuncs.com/compatible-mode/v1") 

# 路径配置 (适配图中的新目录结构)
INPUT_JSONL = "label.jsonl"  # 纯净版元数据文件
MEDIA_DIR = "data"                # 包含 7 个 CASE 文件夹的数据目录

NUM_FRAMES = 8 
EMOTION_CLASSES = ["Happy", "Sad", "Neutral", "Anger", "Disgust", "Fear", "Surprise"]

# 定义所有支持的模型
MODELS = [
    "doubao-seed-2-0-lite",
    "gpt-5-mini",
    "gpt-5.4-nano",
    "gemini-3.1-flash-lite-preview",
    "qwen3.5-plus",
    "gpt-5-nano",
    "qwen3.5-27b"
]

DEFAULT_MODEL_CHOICE = "gpt-5-mini"
# ============================================

def normalize_match_key(name):
    """词性归一化"""
    name = name.lower()
    for old, new in {"angry": "anger", "happiness": "happy", "sadness": "sad"}.items():
        name = name.replace(old, new)
    return name

def find_media_file(video_id):
    """遍历子文件夹寻找对应媒体文件"""
    if not video_id or not os.path.exists(MEDIA_DIR): 
        return None, None
        
    safe_vid = normalize_match_key(video_id)
    
    # 遍历 data 目录下的各个 CASE 文件夹
    for root, dirs, files in os.walk(MEDIA_DIR):
        for fname in files:
            if fname.startswith('.'): continue
            safe_fname = normalize_match_key(os.path.splitext(fname)[0])
            
            if safe_vid == safe_fname:
                full_path = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()
                media_type = 'audio' if ext in ['.wav', '.mp3', '.flac', '.aac'] else 'video'
                return full_path, media_type
            
    return None, None

def get_case_id(item, media_path):
    """提取 Case ID (1-7)，可从 JSON 字段或路径解析"""
    if 'case' in item:
        c = str(item['case']).upper()
        for i in range(1, 8):
            if str(i) in c or f"CASE{i}" in c: return i
            
    # 从路径中提取 (如 data/CASE3_FSM/001.avi)
    if media_path:
        for i in range(1, 8):
            if f"CASE{i}" in media_path: return i
    return 1  # 默认 Case 1

def generate_prediction(item, model_name):
    """生成三段式思维链 (3-stage CoT)"""
    system_prompt = f"""You are an expert in multimodal emotion analysis. 
Analyze the emotion of the main character based on the provided inputs.

IMPORTANT: You must analyze the emotion based on available modalities (Face, Scene, Body Language, and Audio). Note that some inputs may lack certain modalities; please rely on the information that is present.

You must follow the "Let's think step by step" principle and structure your reasoning exactly into three sections:

### 1. Scene Understanding
(Analyze objective visual/audio clues, focusing on scene, body language, and environmental context. Strictly limit this section to 50-60 words.)

### 2. Emotional Analysis
(Analyze how these objective clues from available modalities logically lead to a specific emotional state. Strictly limit this section to 50-60 words.)

### 3. Conclusion
(State the final inferred emotion)

You must strictly output your response in the following JSON format:
{{
    "reasoning": "### 1. Scene Understanding\\n...\\n\\n### 2. Emotional Analysis\\n...\\n\\n### 3. Conclusion\\n...",
    "emotion_label": "Must be exactly one of the following: {', '.join(EMOTION_CLASSES)}"
}}"""

    content = []
    video_id = item.get('video_id', '')
    media_path, media_type = find_media_file(video_id)
    case_id = get_case_id(item, media_path)
    
    # 根据 7 种 Case 动态生成 Prompt 提示
    hint = ""
    if case_id == 2: hint = " (Note: Face details are missing/blurred, but general structures are visible. You can still use it alongside body and scene.)"
    elif case_id == 3: hint = " (Note: Face structures are heavily missing/blurred. Rely purely on body language, scene, and audio.)"
    elif case_id == 4: hint = " (Note: Visuals are completely missing. Rely entirely on audio cues.)"
    elif case_id == 5: hint = " (Note: Audio is missing, face details are blurred. Rely on visuals.)"
    elif case_id == 6: hint = " (Note: Audio and Face structures are missing. Rely strictly on body language and scene.)"
    elif case_id == 7: hint = " (Note: Audio is missing. Rely entirely on visual cues like face, body, and scene.)"

    if media_type == 'video' and case_id != 4:
        content.append({"type": "text", "text": f"Please analyze the character's emotion in the following video frame sequence{hint}:"})
        frames = extract_frames_base64(media_path, NUM_FRAMES)
        for frame in frames:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
            
    # 对于纯音频的 Case 4，或者有音频提取特征的情况
    if case_id in [1, 2, 3, 4] and item.get('stages', {}).get('stage1_extraction'):
        audio_features = item['stages']['stage1_extraction']
        content.append({
            "type": "text", 
            "text": f"Supplementary audio and semantic feature description:\n{audio_features}"
        })

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
            response_format={"type": "json_object"},
            temperature=0.2 
        )
        res = json.loads(response.choices[0].message.content)
    except Exception as e:
        return "", "Error", "api_error", str(e)

    pred_label = res.get("emotion_label", "").strip().capitalize()
    return res.get("reasoning", ""), pred_label if pred_label in EMOTION_CLASSES else "Unknown", "", ""

def extract_frames_base64(video_path, num_frames=8):
    if not video_path: return []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: return []
    
    frame_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    base64_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
    cap.release()
    return base64_frames

def print_metrics(y_true, y_pred, title=""):
    if not y_true: return
    acc = accuracy_score(y_true, y_pred) * 100
    p = precision_score(y_true, y_pred, labels=EMOTION_CLASSES, average='macro', zero_division=0) * 100
    wp = precision_score(y_true, y_pred, labels=EMOTION_CLASSES, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, labels=EMOTION_CLASSES, average='macro', zero_division=0) * 100
    wf1 = f1_score(y_true, y_pred, labels=EMOTION_CLASSES, average='weighted', zero_division=0) * 100
    r = recall_score(y_true, y_pred, labels=EMOTION_CLASSES, average='macro', zero_division=0) * 100
    print(f"\n{'-'*15} {title} {'-'*15}")
    print(f"Samples : {len(y_true)} | ACC: {acc:.2f}% | P: {p:.2f}% | WP: {wp:.2f}% | F1: {f1:.2f}% | WF1: {wf1:.2f}% | R: {r:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_CHOICE, choices=["all"] + MODELS)
    args = parser.parse_args()
    models_to_run = MODELS if args.model == "all" else [args.model]

    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for model_name in models_to_run:
        print("\n" + "="*80)
        print(f"🚀 开始评估模型: {model_name}")
        print("="*80)
        
        grouped_true, grouped_pred = defaultdict(list), defaultdict(list)
        all_true, all_pred = [], []
        
        safe_model_name = model_name.replace("/", "_")
        output_jsonl = f"predictions_{safe_model_name}.jsonl"
        failed_jsonl = f"failed_{safe_model_name}.jsonl"
        error_log = f"errors_{safe_model_name}.log"
        
        written_count, error_count, skipped_count = 0, 0, 0
        processed_video_ids = set()

        if os.path.exists(output_jsonl):
            with open(output_jsonl, 'r', encoding='utf-8') as f_exist:
                for line in f_exist:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line.strip())
                        vid = data.get('video_id')
                        if vid: processed_video_ids.add(vid)
                            
                        true_label = data.get('emotion', '').strip().capitalize()
                        pred_label = data.get('pred_label', '')
                        case_id = data.get('case', 1) # 也可以根据需求读取
                        
                        all_true.append(true_label)
                        all_pred.append(pred_label)
                        grouped_true[case_id].append(true_label)
                        grouped_pred[case_id].append(pred_label)
                        written_count += 1
                    except Exception: pass

        with open(output_jsonl, 'a', encoding='utf-8') as out_f, \
             open(failed_jsonl, 'a', encoding='utf-8') as failed_f, \
             open(error_log, 'a', encoding='utf-8') as error_log_f:
             
            pbar = tqdm(lines, desc=f"Evaluating {model_name}", unit="sample", total=len(lines))
            for line_idx, line in enumerate(lines):
                if not line.strip(): continue
                item = json.loads(line.strip())
                video_id = item.get('video_id')
                
                if video_id in processed_video_ids:
                    skipped_count += 1
                    pbar.update(1)
                    continue
                
                true_label = item.get('emotion', '').strip().capitalize()
                
                pred_cot, pred_label, error_type, error_message = generate_prediction(item, model_name)
                
                # 记录该条数据的 Case ID，方便后续处理
                media_path, _ = find_media_file(video_id)
                case_id = get_case_id(item, media_path)
                item['case'] = case_id
                
                if pred_label == "Error":
                    error_count += 1
                    error_log_f.write(f"[{datetime.now().isoformat()}] video_id={video_id} error={error_message}\n")
                    error_log_f.flush()
                else:
                    item['pred_cot'], item['pred_label'] = pred_cot, pred_label
                    all_true.append(true_label)
                    all_pred.append(pred_label)
                    grouped_true[case_id].append(true_label)
                    grouped_pred[case_id].append(pred_label)
                    
                    out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    out_f.flush()
                    written_count += 1
                
                pbar.update(1)
                pbar.set_postfix(written=written_count, error=error_count, skipped=skipped_count)
            pbar.close()

        print("\n" + "="*60 + f"\n      Stage 1 Objective Perf: {model_name}      \n" + "="*60)
        print_metrics(all_true, all_pred, "Overall Performance")
        for c in sorted(grouped_true.keys()):
            print_metrics(grouped_true[c], grouped_pred[c], f"CASE {c}")

if __name__ == "__main__":
    main()