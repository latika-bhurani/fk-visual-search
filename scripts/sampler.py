import glob
import json
import random
import csv
import os
#import create_features_vgg16 as cf

# dictionary to hold features of the images
image_feature_map = {}


def sample(verticals, output_file, train=True):
    base_dir = "/home/ubuntu/visualsearch/code/fk-visual-search/data/"
    meta_dir = os.path.join(base_dir, "meta", "json")
    base_image_dir = os.path.join(base_dir, "structured_images")
    print(base_image_dir)
    # number_of_n = 100

    number_of_n = 10
#   number_of_pos_neg_pairs = 3 # this will give out 100*10 triplets ##   !!!!!!!!!!!!!!!! update line 69!!!!!!!!!!!!!!!!!!!!
    prefix = "train" if train else "test"
    for vertical in verticals:
        print(vertical)
        # filename = train_pairs_dresses.json
        filename = prefix + "_pairs_" + vertical + ".json"

        # retreival_path = .../retreival_dresses.json
        retrieval_path = os.path.join(meta_dir, "retrieval_" + vertical + ".json")

        # image_dir = ./structures_images/dresses_256
        image_dir = os.path.join(base_image_dir, vertical + "_256")

        print(image_dir)

        # include vertical variable in output file
        output_path = os.path.join(base_dir, output_file)
        # query_path = ./structred_images/wtbi_dresses_query_crop_256
        query_dir = os.path.join(base_image_dir, "wtbi_" + vertical + "_query_crop_256")

        # open file train_pari_dresses.json
        with open(os.path.join(meta_dir, filename)) as jsonFile:
            pairs = json.load(jsonFile)  # {"photo": 2847, "product": 47270, "bbox": {"width": 693, "top": 307, "left": 146, "height": 993}}

        photo_to_product_map = {}


        with open(retrieval_path) as jsonFile:
            data = json.load(jsonFile) # {"photo": 163478, "product": 1}


        for info in data:
            photo_to_product_map[info["photo"]] = info["product"]  # photo : product

        product_to_photo_map = {}

        for photo in photo_to_product_map:
            product = photo_to_product_map[photo]

            if product not in product_to_photo_map:
                product_to_photo_map[product] = set()
            product_to_photo_map[product].add(photo)  # product : set(photos)


        # list of all the image names in ./structured_images/dresses_256
        universe = [int(os.path.splitext(os.path.basename(x))[0]) for x in
                    glob.glob(image_dir + "/*.jpg")]


        for pair in pairs:
            photo = pair["photo"]
            product = pair["product"]
            p_s = []


            for i in product_to_photo_map[product]:
                p_s.append(i) # list of all the photos of a product

            triplets = []
            # print(p_s)
            # exit(0)
            for p in p_s:
                # print('the new p is ', p)
                for j in range(number_of_n):
                    q_id = str(photo)
                    p_id = str(p)
                    n_index = random.randint(0, len(universe) - 1)
                    n = universe[n_index]
                    # n_id is any random photo not in the photos we consider

                    if n not in p_s and n!=photo:

                        n_id = str(n)
                        if os.path.exists(query_dir + "/" + q_id + ".jpg") and os.path.exists(
                                image_dir + "/" + p_id + ".jpg") and os.path.exists(
                            image_dir + "/" + n_id + ".jpg"):

                            # print('triplets len',len(triplets))
                            # if len(triplets) == number_of_pos_neg_pairs*10:
                            #     break
                            triplets.append([q_id, p_id, n_id, vertical])

                # print(triplets)
                # exit(0)

                with open(output_path, "a", newline='') as csvFile:
                    writer = csv.writer(csvFile)

                    count = 0
                    pos_count = 0
                    triplets = [[os.path.join(query_dir, x[0] + ".jpg"), os.path.join(image_dir, x[1] + ".jpg"),
                             os.path.join(image_dir, x[2] + ".jpg"), x[3]] for x in triplets]
                    writer.writerows(triplets)
                    triplets = []


verticals = ['skirts']
#output_path = "/home/ubuntu/visualsearch/code/fk-visual-search/model"
#output_path = os.path.join(output_path,"triplets_skirts_10_sample_new2.csv")
#print(output_path)

# exit(0)

print('started')
sample(verticals, "triplets_skirts_10_sample_new.csv")
print('ended')
