import os
from feature_extractor import FeatureExtractor
from annoy import AnnoyIndex

__author__ = 'latika'

'''
Index catalog images
'''
class Indexer(object):
    def __init__(self, config, image_paths):
        self.layer = config["model_layer"]
        self.search_index_path = config["search_index_path"]
        self.feature_extractor = FeatureExtractor(config["path_to_model_file"], embedding_layer=self.layer)
        self.image_paths = image_paths
        self.search_index = AnnoyIndex(4096, metric="euclidean")

    def index(self, batch_size, start_index=0, stop_index=None):

        batches = [self.image_paths]#[x:x + 10] for x in range(0, len(self.image_paths), 10)]
        print(len(batches))
        batch_num = 0
        if not stop_index:
            stop_index = len(batches)
        # print("Indexing batch " batch_num, len(batch))
        fv_dict = self.feature_extractor.extract_batch(batches[0], self.search_index)

        print(len(fv_dict))
        for i in range(len(fv_dict)):
            self.search_index.add_item(i+1, fv_dict[i])

        # TODO
        # to check if needs to be built repeatedly
        self.search_index.build(50)
        print(self.search_index_path)

        # save indices
        self.search_index.save(self.search_index_path)