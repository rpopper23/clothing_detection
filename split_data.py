import json
from macpath import split
import os
import copy
import random
import shutil

split_vars = {
    'image_dir':'datasets/coco/images',
    "percentagetest": "10",
    "percentagetrain": "80",
    "percentageval": "10",
    "seed": "42"
}

train_dir = 'datasets/train'
val_dir = 'datasets/val'
test_dir = 'datasets/test'
ann_path = "datasets/coco/annotations/"
ann_orig_path = "datasets/modanet/annotations/"
sets_names = ['train', 'val', 'test']

#create folders for train_test and split
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

#split data
if not os.path.isfile(ann_path + 'instances_all.json'):
    # copy the modanet instances to the annotations folder
    print('Copying annotations from the original path')
    with open(ann_orig_path + 'modanet2018_instances_' + sets_names[0] + '.json') as f:
        instances = json.load(f)
    with open(ann_path + 'instances_all.json', 'w') as outfile:
        json.dump(instances, outfile)


# Integrity Check
'''
anns = json.load(open(r"C:\Users\ruben.popper\Desktop\train_maskrcnn\datasets\coco\annotations\instances_all.json"))
aa_imgs = os.listdir(r"C:\Users\ruben.popper\Desktop\train_maskrcnn\datasets\coco\images")
aa_imgs = [int(img_path.replace('.jpg', '')) for img_path in aa_imgs]
new_json = {}
new_json['year'] = anns['year']
new_json['categories'] = anns['categories']
new_json['annotations'] = [annot for annot in anns['annotations'] if annot['image_id'] in aa_imgs]
new_json['licenses'] = anns['licenses']
new_json['type'] = anns['type']
new_json['info'] = anns['info']
new_json['images'] = [annot for annot in anns['images'] if annot['id'] in aa_imgs]
with open('datasets/coco/annotations/' + 'instances_filtered.json', 'w') as outfile:
    json.dump(new_json, outfile)
'''

        
sets_percentages = [int(split_vars['percentagetrain']), int(split_vars['percentageval']), int(split_vars['percentagetest'])]
random.seed(int(split_vars['seed']))
print("Doing " + str([str(p) + '% ' + n for p, n in zip(sets_percentages, sets_names)]))

with open(ann_path + 'instances_all.json') as f:
    instances = json.load(f)
print("Annotations:" + str(len(instances['annotations'])))
print("Images:" + str(len(instances['images'])))

train_ann = {
	'year': instances['year'],
	'categories': instances['categories'],
	'annotations': [],
	'licenses': instances['licenses'],
	'type': instances['type'],
	'info': instances['info'],
	'images': []
}
val_ann = copy.deepcopy(train_ann)
test_ann = copy.deepcopy(train_ann)

if sum(sets_percentages) != 100:
    print("Values not valid, doing 80% train, 10 val and 10 test! Please update your sets percentages")
    sets_percentages = [80, 10, 10]
split_percs = [0]
for perc in sets_percentages:
    #make it cumulative
    last_perc = split_percs.pop()
    last_perc
    split_percs.extend([last_perc + perc] * 2)
split_percs.pop()

# has the images referenced by the annotations for each category, with key the id and value how many annotations has the image of that category
images_anns = [ [0] * 1115985 for i in range(len(instances['categories']) + 1)] #categories start from one

# has [category][set (train{0}, val{1}, test{2})] and as value the number of annotations in that set for that category. to be filled
cat_anns = [ [0, 0, 0] for i in range(len(instances['categories']) + 1) ]

# has as key, image id, as value, (train{0}, val{1}, test{2}). used for faster recovering of this info
images_set = [None] * 1115985

for ann in instances['annotations']:
	images_anns[ann['category_id']][ann['image_id']] += 1

print("Annotations categories for each image recorded")

sum_images_anns = [ sum(images_anns_cat) for images_anns_cat in images_anns ]

# now that I know what annotations images contain, apply probability
for img in instances['images']:
	img_id = img['id']
	p = random.random() * 100
	if p < split_percs[0]:
		train_ann['images'].append(img)
		images_set[img_id] = 0
		for cat_id in range(1, len(instances['categories']) + 1): # for each category
			cat_anns[cat_id][0] += images_anns[cat_id][img_id] # record how many annotations have been added to a set

	elif split_percs[0] <= p < split_percs[1]:
		val_ann['images'].append(img)
		images_set[img_id] = 1
		for cat_id in range(1, len(instances['categories']) + 1): # for each category
			cat_anns[cat_id][1] += images_anns[cat_id][img_id] # record how many annotations have been added to a set
			
	elif p <= split_percs[2]:
		test_ann['images'].append(img)
		images_set[img_id] = 2
		for cat_id in range(1, len(instances['categories']) + 1): # for each category
			cat_anns[cat_id][2] += images_anns[cat_id][img_id] # record how many annotations have been added to a set
			

for cat_id in range(1, len(cat_anns)):
	print("Category ID: " + str(cat_id) + "\tCat Anns: " + str(sum(cat_anns[cat_id])) + 
		"\tCat Percs:" + str([i / float(sum(cat_anns[cat_id])) * 100 for i in cat_anns[cat_id]]))



print()
print("Adding annotations..")
for ann in instances['annotations']:
	# add the annotations to the correct sets
	img_id = ann['image_id']
	if images_set[img_id] == 0:
		train_ann['annotations'].append(ann)
	elif images_set[img_id] == 1:
		val_ann['annotations'].append(ann)
	elif images_set[img_id] == 2:
		test_ann['annotations'].append(ann)
print()
print("Result sum annotations:" + str(sum([len(train_ann['annotations']), len(val_ann['annotations']), len(test_ann['annotations'])])))
print("Result sum images:" + str(sum([len(train_ann['images']), len(val_ann['images']), len(test_ann['images'])])))
print()
# print("Now writing files..")

#save annotations files 
with open(train_dir + '/instances_train.json', 'w') as outfile:
	json.dump(train_ann, outfile)

with open(val_dir + '/instances_val.json', 'w') as outfile:
	json.dump(val_ann, outfile)

with open(test_dir + '/instances_test.json', 'w') as outfile:
	json.dump(test_ann, outfile)
    
#move images 
#images_dir = 'datasets/coco/images'
data_folders = [train_dir , test_dir, val_dir ]
ann_files = ['/instances_train.json', '/instances_test.json', '/instances_val.json']
seen_images = {}

for el in range(len(data_folders)):
    file = data_folders[el]+ann_files[el]
    anns_json = json.load(open(file))
    dest_folder = data_folders[el] + '/images'
    os.makedirs(dest_folder, exist_ok = True)

    for image in anns_json['images']:
        image_id = image['id']
        if image_id in seen_images:
            print("Warning: Skipping duplicate image id: {}".format(image))
        else:
            seen_images[image_id] = image
            try:
                image_file_name = image['file_name']
            except KeyError as key:
                print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
        img_path_origin = os.path.abspath(os.path.join(split_vars['image_dir'], image_file_name))
        img_path_dest = os.path.abspath(os.path.join(dest_folder, image_file_name))
        try:
            shutil.copy(img_path_origin, img_path_dest)
            print("File copied successfully.")
        except:
            print('Image not downloaded yet')
        print(img_path_origin)
        print(img_path_dest)