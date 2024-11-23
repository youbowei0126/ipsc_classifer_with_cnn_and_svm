import pandas as pd
import mylib as my

meta_df=pd.read_csv(my.parent_folder_or_file_under("smaller_metadata.csv"))
print(meta_df.columns)
print(meta_df["structure_name"].unique())
print(len(meta_df))
print(meta_df["cell_stage"].unique())