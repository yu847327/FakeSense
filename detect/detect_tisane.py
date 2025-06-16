import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# 加载API密钥
load_dotenv()
API_KEY = "e2518df46fcc423386f6e4249e69bddf"
API_URL = "https://api.tisane.ai/parse"
LANGUAGE = "zh-CN"  # "en" 或 "zh-CN"(简体中文)

# 读取数据
def load_data(filename):
    if filename.endswith(".csv"):
        df = pd.read_csv(filename)
        return df['original_text'].tolist()
    elif filename.endswith(".json"):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [item["original_text"] for item in data]
    else:
        raise ValueError("只支持 CSV 或 JSON 文件")

# 标准化文本
def normalize_text(text):
    return text.replace("\r", "").replace("\n", " ").strip()

# 发送请求
def analyze_text(text):
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": API_KEY
    }
    payload = {
        "language": LANGUAGE,
        "content": normalize_text(text),
        "settings": {
            "snippets": True,
            "document_sentiment": True,
            "explain":True
            }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def extract_results(text, response):
    # --- 滥用/攻击类内容 ---
    abuses = response.get("abuse", [])
    abuse_result = '; '.join(
        f'{a.get("type", "N/A")} | "{a.get("text", "")}" | Severity: {a.get("severity", "N/A")}'
        + (f' | Explanation: {a.get("explanation", "")}' if a.get("explanation") else '')
        for a in abuses if a.get("text")
    ) or "None"

    # --- 情感分析 ---
    # # 文档级情感（整体情感）：
    doc_sentiment = response.get("sentiment", "None")

    # 片段级情感
    sentiments = response.get("sentiment_expressions", [])
    sentiment_result = '; '.join(
        f'{s.get("polarity", "N/A")}: "{s.get("text", "")}"'
        + (f' [snippet: {s.get("snippet", "")}]' if s.get("snippet") else '')
        for s in sentiments if s.get("text")
    ) or "None"

    # --- 主题提取 ---
    topics = response.get("topics", [])
    topic_result = ', '.join(topics) or "None"

    # --- 实体提取（带子类型） ---
    entities_summary = response.get("entities_summary", [])
    entities_result = '; '.join(
        f'{e.get("type", "N/A")} - {e.get("name", "")}'
        + (f' [subtypes: {", ".join(e.get("subtypes", []))}]' if e.get("subtypes") else '')
        for e in entities_summary if e.get("name")
    ) or "None"


    return {
        "Text": text,
        "Abuse": abuse_result,
        "Sentiment Overall": doc_sentiment,
        "Sentiment Expressions": sentiment_result,
        "Topics": topic_result,
        "Entities": entities_result
    }

# 主函数
def main():
    input_file = "generated_texts.json"  # 文件路径
    texts = load_data(input_file)

    results = []
    for text in tqdm(texts[:10], desc="detecting", ascii=True):  # 默认只检测前10条
        response = analyze_text(text)
        result = extract_results(text, response)
        results.append(result)

    df_out = pd.DataFrame(results)
    df_out.to_csv("results.csv", index=False, encoding="utf-8-sig")
    print("Detection done, outputs saved to results.csv")

if __name__ == "__main__":
    main()
