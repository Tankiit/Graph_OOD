"""
Demo script showing how to use the AbstentionBench task list from dataloader.py
"""

from dataloader import (
    ABSTENTION_BENCH_TASKS,
    TASK_METADATA,
    get_all_tasks,
    get_task_list,
    get_task_metadata,
    get_high_priority_tasks,
    get_tasks_by_category,
    get_abstention_train_test_dataloaders
)

def demo_task_list():
    """Demonstrate various ways to access and use the task list."""

    print("=== AbstentionBench Task List Demo ===\n")

    # 1. Show all available tasks
    print("1. All Available Tasks:")
    print("-" * 50)
    for full_name, task_id in ABSTENTION_BENCH_TASKS.items():
        print(f"  {full_name:45} -> {task_id}")

    # 2. Get just the task identifiers (useful for iteration)
    print("\n2. Task Identifiers (for iteration):")
    print("-" * 50)
    task_ids = get_task_list()
    print(f"  {task_ids}")

    # 3. Get high-priority tasks
    print("\n3. High-Priority Tasks:")
    print("-" * 50)
    high_priority = get_high_priority_tasks()
    for task_id in high_priority:
        meta = get_task_metadata(task_id)
        print(f"  {task_id:25} - {meta['description']}")

    # 4. Get tasks by category
    print("\n4. Knowledge Detection Tasks:")
    print("-" * 50)
    knowledge_tasks = get_tasks_by_category("knowledge_detection")
    for task_id in knowledge_tasks:
        meta = get_task_metadata(task_id)
        print(f"  {task_id:25} - {meta['full_name']}")

    # 5. Example: How to iterate through all tasks for batch processing
    print("\n5. Example: Batch Processing Template")
    print("-" * 50)
    print("  for task_id in get_task_list():")
    print("      print(f'Processing {task_id}...')")
    print("      # Your processing code here")
    print("      # e.g., extract_hidden_states, train_probe, etc.")

    # 6. Example: How to get metadata for a specific task
    print("\n6. Task Metadata Example (GSM8K):")
    print("-" * 50)
    gsm8k_meta = get_task_metadata("gsm8k")
    for key, value in gsm8k_meta.items():
        print(f"  {key:20}: {value}")

    # 7. Example: Creating a task queue for experiments
    print("\n7. Example: Task Queue for Experiments")
    print("-" * 50)
    print("  experiment_queue = [")
    print("      {'task': 'gsm8k', 'stage': 'extraction', 'status': 'pending'},")
    print("      {'task': 'gsm8k', 'stage': 'training', 'status': 'pending'},")
    print("      {'task': 'gpqa', 'stage': 'extraction', 'status': 'pending'},")
    print("      {'task': 'gpqa', 'stage': 'training', 'status': 'pending'},")
    print("  ]")

def demo_task_usage_in_experiments():
    """Show how to use the task list in actual experiment workflows."""

    print("\n\n=== Using Task List in Experiments ===\n")

    # Example 1: Process all high-priority tasks
    print("Example 1: Process all high-priority tasks")
    print("-" * 50)
    high_priority_tasks = get_high_priority_tasks()
    print(f"Found {len(high_priority_tasks)} high-priority tasks")
    for task_id in high_priority_tasks:
        print(f"  Would process: {task_id}")

    # Example 2: Filter tasks by specific criteria
    print("\nExample 2: Get all reasoning-related tasks")
    print("-" * 50)
    reasoning_tasks = []
    for task_id, meta in TASK_METADATA.items():
        if "reasoning" in meta.get("category", ""):
            reasoning_tasks.append(task_id)
    print(f"Reasoning tasks: {reasoning_tasks}")

    # Example 3: Create a simple task progress tracker
    print("\nExample 3: Task Progress Tracker")
    print("-" * 50)
    task_progress = {}
    for task_id in get_task_list():
        task_progress[task_id] = {
            "extraction": "pending",
            "training": "pending",
            "evaluation": "pending"
        }

    # Mark some tasks as completed (example)
    task_progress["gsm8k"]["extraction"] = "completed"
    task_progress["gsm8k"]["training"] = "in_progress"

    print("Task Progress Status:")
    for task_id, stages in task_progress.items():
        if any(status != "pending" for status in stages.values()):
            print(f"  {task_id:15}: {stages}")

if __name__ == "__main__":
    demo_task_list()
    demo_task_usage_in_experiments()

    print("\n\n=== End of Demo ===")
    print("You can now import and use these functions in your experiment scripts!")