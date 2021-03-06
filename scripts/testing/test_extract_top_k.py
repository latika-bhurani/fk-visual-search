import glob
import json
import random
import csv
import os
# import create_features_vgg16 as cf
from pprint import pprint


# dictionary to hold features of the images
image_feature_map = {}


def sample(verticals, output_path, train=True):
    base_dir = ".\\"
    meta_dir = os.path.join(base_dir, "meta", "json")
    base_image_dir = os.path.join(base_dir, "structured_images")
    print(base_image_dir)
    # number_of_n = 100

    number_of_n = 1
    number_of_pos_neg_pairs = 10 # this will give out 100*10 triplets ##   !!!!!!!!!!!!!!!! update line 69!!!!!!!!!!!!!!!!!!!!
    prefix = "train" #if train else "test"
    for vertical in verticals:
        print(vertical)
        # filename = train_pairs_dresses.json
        filename = prefix + "_pairs_" + vertical + ".json"

        # retreival_path = .../retreival_dresses.json
        retrieval_path = os.path.join(meta_dir, "retrieval_" + vertical + ".json")

        # image_dir = ./structures_images/dresses_256
        image_dir = os.path.join(base_image_dir, vertical + "_256")

        print(image_dir)

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

        skirts_dir = '.\\structured_images\\skirts\\'

        similar_items = {}
        similar_items_product_to_photo = {}

        for key, value in product_to_photo_map.items():
            if len(value)>1:
                for val in list(value):
                    if os.path.isfile(os.path.join(skirts_dir, str(val) + '.jpg')):
                        similar_items_product_to_photo[key] = value


        print('printing product to map......', end='\n\n')
        pprint(product_to_photo_map)
        print('printing similar items product to map......', end='\n\n')
        pprint(similar_items_product_to_photo)
        # exit(0)


        # list of all the image names in ./structured_images/dresses_256
        universe = [int(os.path.splitext(os.path.basename(x))[0]) for x in
                    glob.glob(image_dir + "/*.jpg")]


        # print(len(pairs))
        # exit(0)
        # print(pairs)
        # exit(0)
        query_to_similar_photos = {}
        for pair in pairs[:]:
            photo = pair["photo"]
            product = pair["product"]
            p_s = []

            query_to_similar_photos[photo] = product_to_photo_map[product]


        import pickle

        pickle.dump(query_to_similar_photos, open('similar_item_pickle.p', 'wb'))


        print('printing alll similar itemsssss. ................................', end='\n\n\n\n')
        pprint(query_to_similar_photos)
        exit(0)

'''
            for i in product_to_photo_map[product]:
                p_s.append(i) # list of all the photos of a product

            pprint(p_s)
            exit(0)
            

            triplets = []
            # print(p_s)
            # exit(0)
            for p in p_s:
                # print('the new p is ', p)
                for j in range(number_of_n):
                    q_id = str(photo)
                    p_id = str(p)

                    query_to_similar_photos[q_id] = p_id




                    n_index = random.randint(0, len(universe) - 1)
                    n = universe[n_index]
                    # n_id is any random photo not in the photos we consider
                    if n not in p_s and n!=photo:
                        n_id = str(n)
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
                    for triplet in triplets:
                        if os.path.isfile(os.path.join(query_dir, triplet[0] + '.jpg')) and os.path.isfile(os.path.join(image_dir, triplet[1] + '.jpg')) and os.path.isfile(os.path.join(image_dir, triplet[2] + '.jpg')):
                            if triplet[0] not in image_feature_map:
                                image_feature_map[triplet[0]] = cf.get_features(os.path.join(query_dir, triplet[0] + '.jpg'))
                            if triplet[1] not in image_feature_map:
                                image_feature_map[triplet[1]] = cf.get_features(os.path.join(image_dir, triplet[1] + '.jpg'))
                            if triplet[2] not in image_feature_map:
                                image_feature_map[triplet[2]] = cf.get_features(os.path.join(image_dir, triplet[2] + '.jpg'))







                            # print('positive')
                            # print(os.path.join(query_dir, triplet[0] + '.jpg'))
                            # print(os.path.isfile(os.path.join(query_dir, triplet[0] + '.jpg')))
                            # print('negative')
                            # print(os.path.join(image_dir, triplet[1] + '.jpg'))
                            # print(os.path.isfile(os.path.join(image_dir, triplet[1] + '.jpg')))
                            # print('random')
                            # print(os.path.join(image_dir, triplet[2] + '.jpg'))
                            # print(os.path.isfile(os.path.join(image_dir, triplet[2] + '.jpg')))
                            # print(triplet[0])
                            pos_count +=1
                        else:
                            count += 1

                    # print('count', count)
                    # print('pos count', pos_count)
                    # exit(0)

                    # if len([os.path.join(query_dir, x[0] + ".jpg") for x in triplets]) > 10:
                    #     print(triplets)
                    #     exit(0)
                    #     # print(x[0])
                    #     # print('its there')
                    #     # exit(0)


                    print(image_feature_map.keys())
                    triplets = [[os.path.join(query_dir, x[0] + ".jpg"), os.path.join(image_dir, x[1] + ".jpg"),
                             os.path.join(image_dir, x[2] + ".jpg"), x[3]] for x in triplets]
                    writer.writerows(triplets)
                    triplets = []
'''

verticals = ['skirts']
output_path = os.getcwd()
output_path = os.path.join(output_path,"samples.csv")
print(output_path)

# exit(0)

print('started')
sample(verticals, output_path)
print('ended')
