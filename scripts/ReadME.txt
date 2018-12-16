Steps

Image Downloader --> Download images to .\fk-visual-search-master\scripts\images

change_image_len_names --> update the names of the images into readable format

Create_structured_images --> create symlinks into folders under structured_images

create_wtbi_crops --> create folders and cropped images under the folders in structured_images

sampler --> create csv of triplets
trainer ---> builds the model

app.py --> runs the application server