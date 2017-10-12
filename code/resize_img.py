import os
from PIL import Image
src = "test"
width = 390
height = 224
for root, dirs, files in os.walk(src):
	for f in files:
		if not f.startswith('.'):
			path = os.path.join(root, f)
			print("copy/"+path[len(src)+1:])
			img = Image.open(path)
			img = img.resize((width, height), Image.ANTIALIAS)
			if not os.path.exists("copy/"+path[0:len(path)-len(f)]):
				os.makedirs("copy/"+path[0:len(path)-len(f)])
			print("copy/"+path[0:len(path)-len(f)])
			img.save("copy/"+path)
			print("copy/"+path)