import json
import re
import hashlib
from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/audit_knowledge.json")
CLEAN_JSON_PATH = Path("data/audit_knowledge_clean.json")
CLEAN_CSV_PATH = Path("data/audit_knowledge_clean.csv")


def parse_question_info(title: str):
    import re

    title = title.strip()
    match = re.match(r"问题\s*(\d+)\s*(.*)", title)

    if not match:
        return None, title

    question_id = int(match.group(1))
    question = match.group(2).strip()

    return question_id, question

def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def clean_title(title: str) -> str:
    title = title or ""
    title = title.replace("_审计署网站", "")
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def clean_date(date_text: str) -> str:
    """
    把 【2019年09月24日】 转成 2019-09-24
    """
    date_text = date_text or ""
    date_text = date_text.replace("【", "").replace("】", "").strip()

    match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", date_text)
    if not match:
        return date_text

    year, month, day = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def clean_source(source: str) -> str:
    source = source or ""
    source = source.replace("【", "").replace("】", "")
    source = source.replace("来源：", "").replace("来源:", "")
    return source.strip()


def clean_content(content: str, title: str = "") -> str:
    content = content or ""
    title = clean_title(title)

    lines = []
    for line in content.splitlines():
        line = line.strip().replace("\xa0", " ")

        if not line:
            continue

        # 删除网页导航和模板内容
        useless_lines = {
            "首页",
            ">",
            "审计之窗",
            "审计知识",
            "正文",
            "> 正文",
            "字号：",
            "字号:",
            "【大】",
            "【中】",
            "【小】",
            "【关闭】",
            "【打印】",
        }

        if line in useless_lines:
            continue

        # 删除发布时间、来源
        if "发布时间" in line:
            continue
        if "来源" in line and len(line) < 40:
            continue

        # 删除重复标题
        clean_line = clean_title(line)
        if clean_line == title:
            continue

        lines.append(line)

    text = "\n".join(lines)

    # 再做一次收尾清理
    text = text.replace("【关闭】", "")
    text = text.replace("【打印】", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


def main():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    clean_data = []

    for item in raw_data:
        title = clean_title(item.get("title", ""))
        publish_date = clean_date(item.get("publish_date", ""))
        source = clean_source(item.get("source", ""))
        url = item.get("url", "").strip()
        content = clean_content(item.get("content", ""), title)
        question_id, question = parse_question_info(title)

        clean_item = {
            "question_id": question_id,
            "title": title,
            "question": question,
            "url": url,
            "publish_date": publish_date,
            "source": source,
            "category": "审计知识",
            "doc_type": "qa",
            "content": content,
            "embedding_text": title + "\n" + content,
            "content_hash": md5_text(content),
            "char_count": len(content),
        }

        # 过滤空正文
        if clean_item["content"]:
            clean_data.append(clean_item)

    with open(CLEAN_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(clean_data)
    df.to_csv(CLEAN_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"原始数据：{len(raw_data)} 条")
    print(f"清洗后数据：{len(clean_data)} 条")
    print(f"已保存：{CLEAN_JSON_PATH}")
    print(f"已保存：{CLEAN_CSV_PATH}")


if __name__ == "__main__":
    main()
