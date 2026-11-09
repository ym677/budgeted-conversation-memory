"""用显式事实更新、语义检索和字符预算管理对话记忆。"""

import json
from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SPEC = {
    "title": "预算感知对话记忆",
    "subtitle": "在有限上下文里保留最新事实，并比较结构化记忆与最近消息窗口。",
    "slug": "06-budgeted-conversation-memory",
    "accent": "#725c9b",
    "sample_input": "结合我的专业、设备条件和最新职业目标，给下一阶段学习建议。预算450",
    "input_label": "基于合成历史的提问，可写“预算450”控制上下文字符数",
}

DATA_PATH = Path(__file__).parent / "data" / "synthetic_history.json"
FACT_PATTERN = re.compile(r"([\u4e00-\u9fffA-Za-z]{2,12})\s*=\s*([^；;\n。]+)")


def load_history():
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))["messages"]


def extract_facts(history):
    """仅采用用户明确写出的 key=value，后来的同名事实覆盖旧值。"""
    facts = {}
    conflicts = []
    for message in history:
        if message["role"] != "user":
            continue
        for key, value in FACT_PATTERN.findall(message["text"]):
            value = value.strip()
            if key in facts and facts[key]["value"] != value:
                conflicts.append({"key": key, "old": facts[key]["value"], "new": value,
                                  "source": message["id"]})
            facts[key] = {"key": key, "value": value, "source": message["id"]}
    return facts, conflicts


def parse_budget(question):
    match = re.search(r"预算\s*(\d+)", question)
    budget = int(match.group(1)) if match else 450
    if not 160 <= budget <= 2000:
        raise ValueError("字符预算应在 160～2000 之间")
    return budget


def recent_window(history, budget):
    retained = []
    used = 0
    for message in reversed(history):
        line = f"[{message['id']}] {message['role']} {message['text']}"
        extra = len(line) + (1 if retained else 0)
        if used + extra > budget:
            break
        retained.append((message["id"], line))
        used += extra
    retained.reverse()
    return "\n".join(line for _, line in retained), [item[0] for item in retained]


def has_outdated_fact(message, facts):
    for key, value in FACT_PATTERN.findall(message["text"]):
        if key in facts and facts[key]["value"] != value.strip():
            return True
    return False


def build_memory(history, question, budget):
    facts, conflicts = extract_facts(history)
    fact_items = list(facts.values())
    candidates = [message for message in history if not has_outdated_fact(message, facts)]
    documents = [f"{fact['key']} {fact['value']}" for fact in fact_items]
    documents += [message["text"] for message in candidates]
    if not documents:
        return {"context": "", "facts": [], "messages": [], "conflicts": conflicts}
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    matrix = vectorizer.fit_transform(documents)
    scores = cosine_similarity(vectorizer.transform([question]), matrix)[0]

    selected_facts = []
    selected_messages = []
    lines = []
    used = 0

    def append_line(line):
        nonlocal used
        extra = len(line) + (1 if lines else 0)
        if used + extra > budget:
            return False
        lines.append(line)
        used += extra
        return True

    # 每个字段只保留最新值，优先使用与当前问题相关的字段。
    fact_order = np.argsort(-scores[:len(fact_items)], kind="stable")
    for index in fact_order:
        fact = fact_items[index]
        line = f"[事实 {fact['source']}] {fact['key']}={fact['value']}"
        if append_line(line):
            selected_facts.append(fact)

    # 用剩余预算补充语义相关原文，每条消息作为一个原子单元保留。
    message_scores = scores[len(fact_items):]
    for index in np.argsort(-message_scores, kind="stable"):
        if message_scores[index] <= 0:
            continue
        message = candidates[index]
        line = f"[原文 {message['id']}] {message['role']} {message['text']}"
        if append_line(line):
            selected_messages.append(message)

    return {"context": "\n".join(lines), "facts": selected_facts,
            "messages": selected_messages, "conflicts": conflicts}


def run(payload, client):
    question = payload.get("text", "").strip()
    if not question:
        raise ValueError("请输入与历史记录相关的问题")
    budget = parse_budget(question)
    history = load_history()
    memory = build_memory(history, question, budget)
    window_text, window_ids = recent_window(history, budget)
    latest_facts, _ = extract_facts(history)
    window_fact_count = sum(f"{key}={fact['value']}" in window_text
                            for key, fact in latest_facts.items())
    if client.enabled:
        answer = client.complete(
            "你是学习规划助手。仅依据提供的合成对话记忆回答当前问题。使用最新事实，"
            "在结论后标明记忆中的消息编号，缺少依据就说明。不要把旧目标当作当前目标。",
            f"记忆上下文\n{memory['context']}\n\n当前问题\n{question}", max_tokens=700)
    else:
        kept = {fact["key"]: fact["value"] for fact in memory["facts"]}
        fact_text = "；".join(f"{key}={value}" for key, value in kept.items())
        answer = f"本地记忆提要保留了这些最新字段：{fact_text}。\n"
        answer += "可以据此规划下一阶段，API 模式会根据这份上下文生成回答。"
    conflicts = [[item["key"], item["old"], item["new"], item["source"]]
                 for item in memory["conflicts"]]
    return {
        "summary": answer,
        "metrics": {"字符预算": budget, "结构化记忆字符": len(memory["context"]),
                    "结构化保留字段": len(memory["facts"]), "最近窗口保留字段": window_fact_count,
                    "覆盖更新次数": len(memory["conflicts"])},
        "sections": [
            {"title": "结构化与检索记忆", "text": memory["context"] or "预算内没有可保留内容"},
            {"title": "相同预算下的最近消息窗口", "text": window_text or "没有完整消息能装入预算",
             "items": ["保留消息 " + (", ".join(window_ids) or "无")]},
            {"title": "最新事实的来源", "table": {"columns": ["字段", "最新值", "来源"],
             "rows": [[item["key"], item["value"], item["source"]] for item in memory["facts"]]}},
            {"title": "事实更新记录", "table": {"columns": ["字段", "旧值", "新值", "更新来源"],
             "rows": conflicts}},
            {"title": "实验口径", "items": [
                "历史对话为本项目编写的合成记录，不是读取的个人聊天。",
                "预算统计 Python 字符数，不是模型的精确 token 数量。",
                "结构化字段只提取用户明确写出的 key=value，未覆盖任意自然语言事实抽取。",
                "字段数比较是这份示例历史的实际计算结果，不是通用记忆能力基准。"]},
        ],
    }
