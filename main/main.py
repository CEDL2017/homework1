import os
import zipfile



def extract_file(name):
	zipfile.ZipFile(base_path+'/'+name).extractall()

def extract_all():
	f_list = ['/nthu-dataset.zip', '/frames.zip', '/labels.zip']
	for f in f_list:
		extract_file(f)

def transform():
	pass


if __name__ == '__main__':
	base_path = os.environ.get("GRAPE_DATASET_DIR")
	extract_all()
	l = os.listdir(base_path)
	print(l)
	print(os.listdir(base_path + '/' + l[0]))
