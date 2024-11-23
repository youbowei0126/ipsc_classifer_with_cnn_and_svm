import pandas as pd
import quilt3
from pathlib import Path

pkg = quilt3.Package.browse(
    "aics/hipsc_single_cell_image_dataset", registry="s3://allencell"
)
save_path = "C:Projects/allen_cell_data/"
# pkg.fetch(save_path)
pkg["crop_seg"].fetch(save_path / Path("crop_seg"))
