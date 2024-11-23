import quilt3
import pandas as pd
from pathlib import Path
import mylib as my
# import mylib as my
pkg = quilt3.Package.browse(
    "aics/hipsc_single_cell_image_dataset", registry="s3://allencell"
)
meta_df = pd.read_csv(
    r"D:\backup\user\Documents\16_learning_project\04_software_tool\03_python\workspace\machinelearning\final_project\smaller_metadata.csv"
)

print(meta_df.columns)
print(meta_df["structure_name"].unique())
print(len(meta_df))

num_sample = 1
data = meta_df.groupby("cell_stage", group_keys=False)
data = data.apply(pd.DataFrame.sample, n=num_sample)
data = data.reset_index(drop=True)
print(data)
data=data[:5]
data.to_csv(my.parent_folder_or_file_under("select.csv"))
# prepare file paths
save_path = Path("C:/projects/allen_cell_data/")
save_path.mkdir(exist_ok=True)
raw_path = save_path / Path("raw_image")
raw_path.mkdir(exist_ok=True)


for row in data.itertuples():
    subdir_name = row.fov_path.split("/")[0]
    file_name = row.fov_path.split("/")[1]
    local_fn = raw_path / f"{row.structure_name}_{row.FOVId}_ch_{row.ChannelNumberStruct}_original.tiff"
    pkg[subdir_name][file_name].fetch(local_fn)