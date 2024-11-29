import os
import shutil
import random


def move_random_files(src_path, dest_path, percentage=0.2):

    os.makedirs(dest_path, exist_ok=True)

    # 獲取 src_path 中的所有檔案（不包括子目錄）
    files = [
        f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))
    ]

    # 計算需要移動的檔案數量
    num_files_to_move = int(len(files) * percentage)

    # 隨機選擇檔案
    files_to_move = random.sample(files, num_files_to_move)

    # 移動檔案
    for file_name in files_to_move:
        src_file = os.path.join(src_path, file_name)
        dest_file = os.path.join(dest_path, file_name)
        shutil.move(src_file, dest_file)

    print(f"complete")


random.seed(42)
src_path = r"dataset_new\iPSC_Morphologies\train"
dest_path = r"dataset_new\iPSC_Morphologies\test"
for cat in ["Big", "Long", "Mitotic", "RAR-treated","Round"]:
    move_random_files(src_path + "\\" + cat, dest_path + "\\" + cat, 0.2)
    print("complete")
