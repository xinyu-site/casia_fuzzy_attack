#!/usr/bin/env python3
"""Run shell commands sequentially, moving failed/timed-out jobs to the queue tail."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Task:
    line_number: int
    command: str
    attempts: int = 0


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_tasks(path: Path) -> list[Task]:
    tasks = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        command = raw_line.strip()
        if command and not command.startswith("#"):
            tasks.append(Task(line_number, command))
    return tasks


def run_task(task: Task, timeout_seconds: int, workdir: Path) -> tuple[str, int | None]:
    """Return (result, exit code), where result is success/failure/timeout."""
    process = subprocess.Popen(
        task.command,
        cwd=workdir,
        shell=True,
        executable="/bin/bash",
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        # Kill the whole process group so children created by attack.py do not leak.
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return "timeout", None
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise

    return ("success" if return_code == 0 else "failure"), return_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="顺序执行任务；失败或超时的任务放到队尾，稍后重新执行。"
    )
    parser.add_argument(
        "task_file",
        nargs="?",
        default=None,
        help="任务文件（默认：examples/drqn_smac_attack.sh）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15,
        metavar="MINUTES",
        help="每次执行的超时分钟数（默认：15）",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="每个任务最多尝试次数；0 表示一直重试到成功（默认：0）",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=0,
        metavar="SECONDS",
        help="失败任务放回队尾前等待的秒数（默认：0）",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="执行命令的工作目录（默认：run_task_queue.py 所在目录）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout 必须大于 0")
    if args.max_attempts < 0 or args.retry_delay < 0:
        raise SystemExit("--max-attempts 和 --retry-delay 不能小于 0")

    task_file = (
        Path(args.task_file).expanduser().resolve()
        if args.task_file
        else Path(__file__).resolve().parent / "examples" / "smac_attack.sh"
    )
    if not task_file.is_file():
        raise SystemExit(f"任务文件不存在：{task_file}")

    workdir = (
        args.workdir.expanduser().resolve()
        if args.workdir
        else Path(__file__).resolve().parent
    )
    if not workdir.is_dir():
        raise SystemExit(f"工作目录不存在：{workdir}")

    queue = deque(load_tasks(task_file))
    total = len(queue)
    if not queue:
        print(f"任务文件中没有可执行命令：{task_file}")
        return 0

    timeout_seconds = int(args.timeout * 60)
    completed = 0
    abandoned = 0
    print(f"[{timestamp()}] 加载 {total} 个任务；单次超时 {args.timeout:g} 分钟")
    print(f"工作目录：{workdir}", flush=True)

    try:
        while queue:
            task = queue.popleft()
            task.attempts += 1
            print(
                f"\n[{timestamp()}] 开始：行 {task.line_number}，第 {task.attempts} 次尝试"
                f"（队列剩余 {len(queue)}）\n$ {task.command}",
                flush=True,
            )
            started = time.monotonic()
            result, return_code = run_task(task, timeout_seconds, workdir)
            elapsed = time.monotonic() - started

            if result == "success":
                completed += 1
                print(f"[{timestamp()}] 成功：行 {task.line_number}，耗时 {elapsed:.1f} 秒", flush=True)
                continue

            detail = "执行超时" if result == "timeout" else f"退出码 {return_code}"
            if args.max_attempts and task.attempts >= args.max_attempts:
                abandoned += 1
                print(
                    f"[{timestamp()}] 放弃：行 {task.line_number}（{detail}），"
                    f"已达到 {args.max_attempts} 次上限",
                    flush=True,
                )
            else:
                print(
                    f"[{timestamp()}] 延后：行 {task.line_number}（{detail}），已放到队尾",
                    flush=True,
                )
                if args.retry_delay:
                    time.sleep(args.retry_delay)
                queue.append(task)
    except KeyboardInterrupt:
        print(f"\n[{timestamp()}] 收到中断，调度器已停止。", file=sys.stderr)
        return 130

    print(f"\n[{timestamp()}] 结束：成功 {completed}/{total}，放弃 {abandoned}/{total}")
    return 1 if abandoned else 0


if __name__ == "__main__":
    raise SystemExit(main())
