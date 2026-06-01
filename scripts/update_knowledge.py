'''
全量更新脚本
'''
import subprocess
import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "update_knowledge.log"


def write_log(message: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {message}"
    print(line)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name: str, command: list[str]):
    write_log(f"开始执行：{name}")

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        write_log(f"{name} 输出：\n{result.stdout}")

    if result.stderr:
        write_log(f"{name} 错误输出：\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"{name} 执行失败，returncode={result.returncode}")

    write_log(f"完成执行：{name}")


def main():
    write_log("========== 开始更新审计知识库 ==========")

    try:
        run_step("爬取审计署数据", ["uv", "run", "python", "scripts/crawl_audit_playwright.py"])
        run_step("清洗审计数据", ["uv", "run", "python", "scripts/clean_audit_data.py"])
        run_step("写入 Qdrant 向量库", ["uv", "run", "python", "scripts/ingest_qdrant.py"])

        write_log("审计知识库更新成功")
        write_log("========== 更新流程结束 ==========")

    except Exception as e:
        write_log(f"审计知识库更新失败：{e}")
        write_log("========== 更新流程异常结束 ==========")
        raise


if __name__ == "__main__":
    main()
