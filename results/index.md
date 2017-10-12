# 陳聖諺 <span style="color:red">(105061604)</span>

#Project 5: Deep Classification

## Overview
The project is related to 
> 1. Deep-learning-based method
  2. Alexnet
  3. Classification

## Implementation
[Load Input] 2-way
1. Using os.listdir  (need to sort)    

code example:
    for im in os.listdir(pathh1):
        
        test_path = os.path.join(pathh1, im)
        test_path_list.append(test_path)
        test_path_list = sorted(test_path_list, key=lambda x: int(re.sub('\D', '', x)))

      test_path_listr.extend(test_path_list)
      test_path_list=[]
2. Using range (read images in order but need to know exactly how many data in a file)

code example:

    for i in range(572):
        i=i+1
        img_path = '../Image'+str(i)+'.png'
        imgs_path_list.append(img_path)

[Difference between loading Image and Label]

	When loading image, we have to load the path first. If we load all the images at the same time, the RAM will be overloaded and the computer will be crashed. Therefore, we will actually read the image until we are doing the training process. However, since the labels are not quite a big file, we can load them all in the beginning.




## Installation
* Other required packages.
* How to compile from source?

### Results

<table border=1>
<tr>
<td>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg"  width="24%"/>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg" width="24%"/>
</td>
</tr>

<tr>
<td>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg"  width="24%"/>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg" width="24%"/>
</td>
</tr>

</table>


