import json
import time
import random
import hashlib
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from tqdm import tqdm


BASE_URL = "https://www.audit.gov.cn"
LIST_URL = "https://www.audit.gov.cn/n6/n37/index.html"

DATA_DIR = Path("data")
JSON_PATH = DATA_DIR / "audit_knowledge.json"
CSV_PATH = DATA_DIR / "audit_knowledge.csv"
FAILED_PATH = DATA_DIR / "failed_urls.json"

REQUEST_DELAY_MIN = 1.5
REQUEST_DELAY_MAX = 3.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


def random_sleep():
    time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))


def get_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.encoding = resp.apparent_encoding
    resp.raise_for_status()
    return resp.text


def clean_text(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def get_content_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def parse_articles_from_html(html: str, base_url: str = LIST_URL):
    """
    从当前页面 HTML 中提取文章标题和链接。
    这个函数既可以解析 requests 拿到的 HTML，也可以解析 Playwright 渲染后的 HTML。
    """
    soup = BeautifulSoup(html, "lxml")
    results = []

    for a in soup.find_all("a"):
        title = clean_text(a.get_text())
        href = a.get("href")

        if not href:
            continue

        if not title.startswith("问题"):
            continue

        full_url = urljoin(base_url, href)

        if "content" not in full_url:
            continue

        results.append({
            "title": title,
            "url": full_url,
        })

    # 单页去重
    seen = set()
    unique_results = []
    for item in results:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique_results.append(item)

    return unique_results


def collect_article_links_with_playwright(max_pages: int = 10):
    """
    用 Playwright 打开页面，然后模拟点击分页按钮。
    适合 URL 不变、页面内容通过 JS 改变的情况。
    """
    all_articles = []
    seen_urls = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=HEADERS["User-Agent"],
            viewport={"width": 1280, "height": 900},
        )

        print(f"正在打开列表页：{LIST_URL}")
        page.goto(LIST_URL, wait_until="networkidle", timeout=60000)
        random_sleep()

        for page_no in range(1, max_pages + 1):
            print(f"正在解析第 {page_no} 页...")

            html = page.content()
            articles = parse_articles_from_html(html, LIST_URL)

            new_count = 0
            for item in articles:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    all_articles.append(item)
                    new_count += 1

            print(f"第 {page_no} 页发现 {len(articles)} 条，新增 {new_count} 条，累计 {len(all_articles)} 条")

            # 尝试点击下一页
            next_button = page.locator("text=下页")

            if next_button.count() == 0:
                print("没有找到“下页”按钮，停止翻页")
                break

            # 如果已经没有新增数据，也可以停止，避免死循环
            if new_count == 0 and page_no > 1:
                print("当前页没有新增数据，停止翻页")
                break

            try:
                next_button.first.click(timeout=5000)
                page.wait_for_timeout(1500)
                page.wait_for_load_state("networkidle", timeout=30000)
                random_sleep()
            except Exception as e:
                print("点击“下页”失败，停止翻页")
                print(f"错误原因：{e}")
                break

        browser.close()

    return all_articles


def parse_detail_page(url: str):
    html = get_html(url)
    soup = BeautifulSoup(html, "lxml")

    h1 = soup.find("h1")
    if h1:
        title = clean_text(h1.get_text())
    else:
        title = clean_text(soup.title.get_text()) if soup.title else ""

    page_text = clean_text(soup.get_text("\n"))

    publish_date = ""
    source = ""

    for line in page_text.splitlines():
        if "发布时间" in line:
            publish_date = line.replace("发布时间：", "").replace("发布时间:", "").strip()
        if "来源" in line:
            source = line.replace("来源：", "").replace("来源:", "").strip()

    content_node = (
        soup.find(class_="TRS_Editor")
        or soup.find(class_="article-content")
        or soup.find(class_="content")
        or soup.find(id="zoom")
    )

    if content_node:
        content = clean_text(content_node.get_text("\n"))
    else:
        content = page_text

        for marker in ["来源：", "来源:"]:
            if marker in content:
                content = content.split(marker, 1)[-1]

        for marker in ["版权信息", "主办单位"]:
            if marker in content:
                content = content.split(marker, 1)[0]

        content = clean_text(content)

    return {
        "title": title,
        "url": url,
        "publish_date": publish_date,
        "source": source,
        "content": content,
        "content_hash": get_content_hash(content),
    }


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("开始使用 Playwright 自动翻页收集审计知识文章链接...")

    articles = collect_article_links_with_playwright(max_pages=10)

    print(f"所有页面共发现 {len(articles)} 篇文章")

    all_data = []
    failed_urls = []

    for item in tqdm(articles, desc="正在爬取详情页"):
        try:
            detail = parse_detail_page(item["url"])

            if not detail["title"]:
                detail["title"] = item["title"]

            all_data.append(detail)
            random_sleep()

        except Exception as e:
            print(f"爬取失败：{item['url']}")
            print(f"错误原因：{e}")

            failed_urls.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "error": str(e),
            })

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(all_data)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    with open(FAILED_PATH, "w", encoding="utf-8") as f:
        json.dump(failed_urls, f, ensure_ascii=False, indent=2)

    print("爬取完成！")
    print(f"JSON 文件：{JSON_PATH}")
    print(f"CSV 文件：{CSV_PATH}")
    print(f"失败记录：{FAILED_PATH}")
    print(f"成功保存 {len(all_data)} 条数据")
    print(f"失败 {len(failed_urls)} 条")


if __name__ == "__main__":
    main()
