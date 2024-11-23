import pandas as pd
import quilt3
from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

# connect to quilt
pkg = quilt3.Package.browse("aics/hipsc_single_cell_image_dataset", registry="s3://allencell")
meta_df = pkg["metadata.csv"]()

# we use lamin B1 cell line for example (structure_name=='LMNB1')
meta_df_lamin = meta_df.query("structure_name=='LMNB1'")

# collapse the data table based on FOVId
meta_df_lamin.drop_duplicates(subset="FOVId", inplace=True)

# reset index
meta_df_lamin.reset_index(drop=True, inplace=True)

# prepare file paths
save_path = Path("C:/projects/allen_cell_data/")
save_path.mkdir(exist_ok=True)
raw_path = save_path / Path("raw_image")
raw_path.mkdir(exist_ok=True)
structure_path = save_path / Path("structure")
structure_path.mkdir(exist_ok=True)
seg_path = save_path / Path("structure_segmentation")
seg_path.mkdir(exist_ok=True)

# download all FOVs or a certain number
num = 5 # or num = row.shape[0]
for row in meta_df_lamin.itertuples():
    
    if row.Index >= num:
        break
    
    # fetch the raw image
    subdir_name = row.fov_path.split("/")[0]
    file_name = row.fov_path.split("/")[1]
    
    local_fn = raw_path / f"{row.FOVId}_original.tiff"
    pkg[subdir_name][file_name].fetch(local_fn)
    
    # extract the structure channel
    structure_fn = structure_path / f"{row.FOVId}.tiff"
    reader = AICSImage(local_fn)
    with OmeTiffWriter(structure_fn) as writer:
        writer.save(
            reader.get_image_data("ZYX", C=row.ChannelNumberStruct, S=0, T=0),
            dimension_order='ZYX'
        )
        
    # fetch structure segmentation
    subdir_name = row.struct_seg_path.split("/")[0]
    file_name = row.struct_seg_path.split("/")[1]
    
    seg_fn = seg_path / f"{row.FOVId}_segmentation.tiff"
    pkg[subdir_name][file_name].fetch(seg_fn)
# prepare file paths
save_path = Path("C:/projects/allen_cell_data/")
save_path.mkdir(exist_ok=True)
raw_path = save_path / Path("raw_image")
raw_path.mkdir(exist_ok=True)
dye_path = save_path / Path("dye")
dye_path.mkdir(exist_ok=True)
seg_path = save_path / Path("fov_segmentation")
seg_path.mkdir(exist_ok=True)

# download all FOVs or a certain number
num = 5 # or num = row.shape[0]
for row in meta_df_lamin.itertuples():
    
    # fetch the raw image
    subdir_name = row.fov_path.split("/")[0]
    file_name = row.fov_path.split("/")[1]
    
    local_fn = raw_path / f"{row.FOVId}_original.tiff"
    pkg[subdir_name][file_name].fetch(local_fn)
    
    # extract the membrane dye and DNA dye channels
    dye_fn = dye_path / f"{row.FOVId}.tiff"
    reader = AICSImage(local_fn)
    with OmeTiffWriter(dye_fn) as writer:
        writer.save(
            reader.get_image_data(
                "CZYX",
                C=[row.ChannelNumber638, row.ChannelNumber405],
                S=0,
                T=0
            ),
            dimension_order='ZYX'
        )
        
    # fetch fov segmentation
    subdir_name = row.fov_seg_path.split("/")[0]
    file_name = row.fov_seg_path.split("/")[1]
    
    seg_fn = seg_path / f"{row.FOVId}_segmentation.tiff"
    pkg[subdir_name][file_name].fetch(seg_fn)