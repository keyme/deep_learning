"""
This script helps convert the Tiny ImageNet dataset (200 classes) to a structure that Keras supports
"""
import os
import argparse
import shutil

from tqdm import tqdm


def process_training_data(path_to_data):
	"""Move all images such that they are directly under their cateogry folder
	"""
	old_train_path = os.path.join(path_to_data, 'train')
	new_train_path = os.path.join(path_to_data, 'train_keras')

	# create new training data path if it does not exist
	if not os.path.isdir(new_train_path):
		os.mkdir(new_train_path)

	for wnid in tqdm(os.listdir(old_train_path)):
		assert len(wnid) == 9
		# create a folder for this wnid if there is not one
		if not os.path.isdir(os.path.join(new_train_path, wnid)):
			os.mkdir(os.path.join(new_train_path, wnid))
		for img_file in os.listdir(os.path.join(old_train_path, wnid, 'images')):
			src = os.path.join(old_train_path, wnid, 'images', img_file)
			dst = os.path.join(new_train_path, wnid, img_file)
			shutil.copyfile(src, dst)

def process_validation_data(path_to_data):
	"""Move validation images to wnid folders based on their label
	"""
	old_val_path = os.path.join(path_to_data, 'val')
	new_val_path = os.path.join(path_to_data, 'val_keras')

	# create new training data path if it does not exist
	if not os.path.isdir(new_val_path):
		os.mkdir(new_val_path)
	
	# read from validation annotation file
	label_to_images = {}
	with open(os.path.join(old_val_path, 'val_annotations.txt'), 'r') as f:
		lines = f.readlines()
		for line in lines:
			img_file, label = line.split()[:2]
			if label not in label_to_images:
				label_to_images[label] = []
			label_to_images[label].append(img_file)

	# create a folder for each wnid and copy images
	for wnid in tqdm(label_to_images.keys()):
		assert len(wnid) == 9
		if not os.path.isdir(os.path.join(new_val_path, wnid)):
			os.mkdir(os.path.join(new_val_path, wnid))
		for img_file in label_to_images[wnid]:
			src = os.path.join(old_val_path, 'images', img_file)
			dst = os.path.join(new_val_path, wnid, img_file)
			shutil.copyfile(src, dst)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("path_to_data", default=None, help="Specify the path to imagenet data")
	args = parser.parse_args()

	try:
		# process training data
		print("start processing training data...")
		process_training_data(args.path_to_data)

		# process validation data
		print("start processing validation data...")
		process_validation_data(args.path_to_data)
	except:
		print("Exception caught. Removing copied images...")
		if os.path.isdir(os.path.join(path_to_data, 'train_keras')):
			shutil.rmtree(os.path.join(path_to_data, 'train_keras'))
		if os.path.isdir(os.path.join(path_to_data, 'val_keras')):
			shutil.rmtree(os.path.join(path_to_data, 'val_keras'))

if __name__ == "__main__":
	main()
