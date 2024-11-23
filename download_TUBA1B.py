import quilt3
import pandas as pd
from pathlib import Path
import mylib as my
from tqdm import tqdm
# import mylib as my
pkg = quilt3.Package.browse("aics/hipsc_single_cell_image_dataset", registry="s3://allencell")
meta_df = pd.read_csv(
    r"D:\backup\user\Documents\16_learning_project\04_software_tool\03_python\workspace\machinelearning\final_project\smaller_metadata.csv"
)

print(meta_df.columns)
print(meta_df["structure_name"].unique())
print(len(meta_df))

num_sample = 1

data_TUBA1B = meta_df.loc[meta_df["structure_name"] == "TUBA1B", :]
print(len(data_TUBA1B))
data_TUBA1B.to_csv(my.parent_folder_or_file_under("select_TUBA1B.csv"))
print(data_TUBA1B["cell_stage"].value_counts())

data_TUBA1B_sample = (
    data_TUBA1B.groupby("cell_stage", group_keys=False).apply(
        lambda x: x.sample(n=64, random_state=42)
    )
    # random_state 確保結果可重現
)

data_TUBA1B_sample=data_TUBA1B_sample.iloc[-5:,:]

data_TUBA1B_sample.to_csv(my.parent_folder_or_file_under("select_TUBA1B_sample.csv"))



save_path = Path("F:/projects/allen_cell_data/")
save_path.mkdir(parents=True,exist_ok=True)
raw_path = save_path / Path("raw_image")
raw_path.mkdir(exist_ok=True)
seg_path = save_path / Path("segment_image")
seg_path.mkdir(exist_ok=True)

for i in tqdm(range(len(data_TUBA1B_sample))):
    print()
    row=data_TUBA1B_sample.iloc[i]
    
    fov_path=row["fov_path"]
    subdir_name = fov_path.split("/")[0]
    file_name = fov_path.split("/")[1]
    CellId = row["CellId"]
    local_fn = raw_path / f"{CellId}.tiff"
    pkg[subdir_name][file_name].fetch(local_fn)
    print()
    
    fov_seg_path=row["fov_seg_path"]
    subdir_name = fov_seg_path.split("/")[0]
    file_name = fov_seg_path.split("/")[1]
    local_fn = seg_path / f"{CellId}.tiff"
    pkg[subdir_name][file_name].fetch(local_fn)
    print()
    import os
    os.system('cls')
    
    
    


