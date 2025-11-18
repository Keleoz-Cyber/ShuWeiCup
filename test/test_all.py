#!/usr/bin/env python3
"""
Test script for task1-4 training startup.
Tests each task's training script until epoch 0 starts, then kills and cleans up.

"Talk is cheap. Show me the code." - Linus Torvalds
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Task configurations: (script, args, checkpoint_dir)
TASKS = [
    {
        "name": "Task 1",
        "script": "task1train.py",
        "args": [
            "--train-meta",
            "data/cleaned/metadata/train_metadata.csv",
            "--val-meta",
            "data/cleaned/metadata/val_metadata.csv",
            "--epochs",
            "50",
            "--batch-size",
            "64",
            "--lr",
            "3e-4",
            "--save-dir",
            "checkpoints/test_task1",
        ],
        "checkpoint_dir": "checkpoints/test_task1",
    },
    {
        "name": "Task 2",
        "script": "task2train.py",
        "args": [
            "--train-meta",
            "data/cleaned/metadata/train_metadata_fewshot_10.csv",
            "--val-meta",
            "data/cleaned/metadata/val_metadata.csv",
            "--epochs",
            "50",
            "--batch-size",
            "8",
            "--lr",
            "3e-4",
            "--proto-weight",
            "0.4",
            "--arcface-margin",
            "0.30",
            "--save-dir",
            "checkpoints/test_task2",
        ],
        "checkpoint_dir": "checkpoints/test_task2",
    },
    {
        "name": "Task 3",
        "script": "task3train.py",
        "args": [
            "--train-meta",
            "data/cleaned/metadata/train_metadata.csv",
            "--val-meta",
            "data/cleaned/metadata/val_metadata.csv",
            "--image-root",
            "data/cleaned/train",
            "--val-image-root",
            "data/cleaned/val",
            "--epochs",
            "30",
            "--batch-size",
            "64",
            "--lr",
            "3e-4",
            "--use-class-weights",
            "--gradcam-samples",
            "12",
            "--save-dir",
            "checkpoints/test_task3",
        ],
        "checkpoint_dir": "checkpoints/test_task3",
    },
    {
        "name": "Task 4",
        "script": "task4train.py",
        "args": [
            "--train-meta",
            "data/cleaned/metadata/train_metadata.csv",
            "--val-meta",
            "data/cleaned/metadata/val_metadata.csv",
            "--train-dir",
            "data/cleaned/train",
            "--val-dir",
            "data/cleaned/val",
            "--epochs",
            "25",
            "--batch-size",
            "64",
            "--lr",
            "3e-4",
            "--dynamic-task-weights",
            "--report-samples",
            "50",
            "--save-dir",
            "checkpoints/test_task4",
        ],
        "checkpoint_dir": "checkpoints/test_task4",
    },
]

# Regex patterns to detect epoch 0 start
EPOCH_PATTERNS = [
    re.compile(r"epoch\s*[:\[]?\s*0\b", re.IGNORECASE),
    re.compile(r"\bEpoch\s*0\b"),
    re.compile(r"\[0/\d+\]"),
    re.compile(r"Training epoch 0", re.IGNORECASE),
]


def cleanup_checkpoint(checkpoint_dir):
    """Delete checkpoint directory if exists."""
    path = Path(checkpoint_dir)
    if path.exists():
        print(f"\n[CLEANUP] Removing {checkpoint_dir}...", end=" ", flush=True)
        try:
            shutil.rmtree(path)
            print("✓ Done")
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print(f"\n[CLEANUP] No checkpoint directory found (OK)")


def test_task(task_config):
    """Test a single task's training startup."""
    name = task_config["name"]
    script = task_config["script"]
    args = task_config["args"]
    checkpoint_dir = task_config["checkpoint_dir"]

    print(f"\n{'=' * 70}")
    print(f"║ {name}: {script}")
    print(f"{'=' * 70}")

    # Build command
    cmd = [sys.executable, script] + args
    print(f"\n[COMMAND] {' '.join(cmd)}\n")
    print(f"[STATUS] Starting process...")
    print(f"[STATUS] Monitoring output for epoch 0 pattern...")
    print(f"[STATUS] Will auto-terminate when training starts\n")
    print(f"{'-' * 70}")

    # Start process
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        epoch_0_detected = False
        timeout = 300  # 5 minutes timeout
        start_time = time.time()
        line_count = 0

        # Read output line by line
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"\n[TIMEOUT] {timeout}s exceeded - killing process")
                process.kill()
                return False

            # Read line with timeout
            line = process.stdout.readline()

            if not line:
                # Process ended
                retcode = process.poll()
                if retcode is not None:
                    if retcode != 0:
                        print(
                            f"\n[FAILED] Process exited with code {retcode} before epoch 0 started"
                        )
                        return False
                    else:
                        print(f"\n[FAILED] Process completed without starting epoch 0")
                        return False
                time.sleep(0.1)
                continue

            # Print line with line counter
            line_count += 1
            # Show progress indicator every 10 lines
            if line_count % 10 == 0:
                print(f"[{line_count:04d}] ", end="", flush=True)
            print(line, end="", flush=True)

            # Check for epoch 0
            if not epoch_0_detected:
                for pattern in EPOCH_PATTERNS:
                    if pattern.search(line):
                        epoch_0_detected = True
                        print(f"\n{'=' * 70}")
                        print(f"[DETECTED] ✓ Epoch 0 pattern matched!")
                        print(f"[DETECTED] Pattern: {pattern.pattern}")
                        print(f"[DETECTED] Line {line_count}: {line.strip()}")
                        print(f"{'=' * 70}")
                        print(f"[TERMINATE] Sending SIGTERM to process...", flush=True)
                        process.terminate()
                        time.sleep(1)
                        if process.poll() is None:
                            print(
                                f"[TERMINATE] Process still alive, sending SIGKILL...", flush=True
                            )
                            process.kill()
                        break

            if epoch_0_detected:
                break

        # Wait for process to finish
        process.wait()

        if epoch_0_detected:
            print(f"\n[RESULT] ✓ {name} test PASSED ({line_count} lines, {elapsed:.1f}s)")
            return True
        else:
            print(f"\n[RESULT] ✗ {name} test FAILED")
            return False

    except KeyboardInterrupt:
        print(f"\n\n[INTERRUPT] Ctrl+C detected - killing process")
        if process.poll() is None:
            process.kill()
        return False
    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup checkpoint
        cleanup_checkpoint(checkpoint_dir)


def main():
    """Main test runner."""
    print("\n" + "=" * 70)
    print("║")
    print("║  ShuWeiCamp Task Training Startup Tests")
    print('║  "Talk is cheap. Show me the code." - Linus Torvalds')
    print("║")
    print("=" * 70)
    print("\n[INFO] Testing task1-4 training startup")
    print("[INFO] Each task runs until epoch 0 starts, then terminates")
    print("[INFO] Generated checkpoints will be cleaned up automatically")
    print(f"[INFO] Total tasks to test: {len(TASKS)}\n")

    results = {}
    start_time = time.time()

    for idx, task in enumerate(TASKS, 1):
        print(f"\n[PROGRESS] Task {idx}/{len(TASKS)}")
        success = test_task(task)
        results[task["name"]] = success

        if idx < len(TASKS):
            print(f"\n[WAIT] Pausing 2 seconds before next task...", flush=True)
            time.sleep(2)

    total_time = time.time() - start_time

    # Print summary
    print(f"\n\n{'=' * 70}")
    print("║ TEST SUMMARY")
    print(f"{'=' * 70}\n")

    for task_name, success in results.items():
        status = "[PASS] ✓" if success else "[FAIL] ✗"
        print(f"{status:12} {task_name}")

    # Overall result
    all_passed = all(results.values())
    passed_count = sum(1 for s in results.values() if s)
    failed_count = len(results) - passed_count

    print(f"\n{'-' * 70}")
    print(f"Passed: {passed_count}/{len(results)}")
    print(f"Failed: {failed_count}/{len(results)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'=' * 70}\n")

    if all_passed:
        print("[RESULT] ✓ All tests passed!\n")
        return 0
    else:
        print(f"[RESULT] ✗ {failed_count}/{len(results)} tests failed\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
