import glob
import os
import sys
from indexer import Indexer

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 1:
        print("Requires parameters vertical")
        sys.exit(1)
    vertical = args[0]
    base_dir = "data/model/"
    index_dir = "data/indices/"
    base_image_dir = "data/structured_images/"
    config = {}
    config["model_layer"] = "visnet_model"
    # config["input_layer"] = "data_q"
    config["search_index_path"] = os.path.join(index_dir, vertical+"crop_ann_index.ann")
    config["path_to_model_file"] = os.path.join(base_dir, "fashion_lens_model.h5")
    imdir = os.path.join(base_image_dir, "wtbi_" + vertical + "_query_crop_256")#vertical+"_256")
    if not os.path.exists(imdir):
        imdir = os.path.join(base_image_dir, vertical)
    image_paths = glob.glob(imdir+"/*.jpg")
    indexer = Indexer(config, image_paths)
    indexer.index(64)