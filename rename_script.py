# prepare images total number

place = ['house', 'lab', 'office']
# place = ['house']
num = ['1', '2', '3']
# num = num[0:1]
kind = ['Lhand', 'Rhand']
# kind = kind[0:1]
img_num = 0

for p in place:
    for n in num:
        for k in kind:
            path = '../frames/train/' + p + '/' + n + '/' + k
            png_list = os.listdir(path)
            img_num += len(png_list)
            for pic in png_list:
                npic = pic.replace('Image', '')
                npic = npic.replace('.png', '')
                if int(npic) < 100:
                    npic = 'Image' + npic.zfill(3) + '.png'
                    os.rename(path + '/' + pic, path + '/' + npic)

for k in kind:
    path = '../frames/train/' + 'lab' + '/' + '4' + '/' + k
    png_list = os.listdir(path)
    img_num += len(png_list)
    for pic in png_list:
        npic = pic.replace('Image', '')
        npic = npic.replace('.png', '')
        if int(npic) < 100:
            npic = 'Image' + npic.zfill(3) + '.png'
            os.rename(path + '/' + pic, path + '/' + npic)

print('number of images:', img_num)