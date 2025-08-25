import os

def get_next_run_folder(base_dir="checkpoint"):
    """
    获取下一个可用的 run 文件夹名称。
    例如：如果已有 run1 和 run2，则返回 run3。
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)  # 如果 checkpoint 目录不存在，则创建

    run_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run")]

    if not run_folders:
        # 如果没有 run 文件夹，则创建 run1
        return os.path.join(base_dir, "run1")
    else:
        # 找到最大的 runX 文件夹
        run_numbers = [int(folder[3:]) for folder in run_folders]
        max_run_number = max(run_numbers)
        next_run_number = max_run_number + 1
        return os.path.join(base_dir, f"run{next_run_number}/")