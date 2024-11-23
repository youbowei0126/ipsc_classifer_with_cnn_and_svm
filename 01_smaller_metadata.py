import pandas as pd
import mylib as my


columns_to_keep = [
    "CellId", "roi", "crop_raw", "crop_seg", "name_dict", "fov_path", "fov_seg_path",
    "struct_seg_path", "structure_name", "this_cell_nbr_complete", "this_cell_nbr_dist_2d",
    "scale_micron", "edge_flag", "FOVId", "this_cell_index", "PlateId", "WellId", 
    "cell_stage", "InstrumentId", "WorkflowId", "ProtocolId", "PiezoId", 
    "ChannelNumberStruct", "ChannelNumberBrightfield", "ChannelNumber405", "ChannelNumber638", 
    "meta_fov_position", "meta_imaging_mode", "meta_fov_outside_overview", "meta_fov_xcoord", 
    "meta_fov_ycoord", "meta_fov_edgedist", "meta_colony_label", "meta_colony_centroid", 
    "meta_colony_area", "meta_plate_bad_segmentation", "meta_plate_confluency", 
    "meta_well_passage_at_imaging", "meta_well_passage_at_thaw", "outlier", 
    "NUC_shape_volume", "NUC_position_depth", "NUC_roundness_surface_area", 
    "MEM_shape_volume", "MEM_position_depth", "MEM_roundness_surface_area", 
    "STR_shape_volume", "STR_connectivity_cc"
]

df = pd.read_csv(r"assest_csv\metadata.csv")
print("loaded")
df_filtered = df[columns_to_keep]
print("filtered")
df_filtered.to_csv(r"assest_csv\smaller_metadata.csv", index=False)
print("done")