# Your Name <span style="color:red">(id)</span>

#Project 5: Deep Classification

## Overview
The project is heavily reference the code : https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c
omoindrot teaches us how to tune a vgg model
> quote


## Implementation
1. Fine-tune a vgg-16
	* restore a pretrained weight (exclude fc8)
	* train a new fc8
	* train the whole model
	* save the trained weight
2. test.py
	* load testing data
	* calculate accuracy

```

```

## Installation
* tensorflow
* download VGG pretrained weight, "vgg_16.ckpt", put it in "code" file.
* download trained weight, "model.ckpt.data-00000-of-00001", put it in the "saved_model" file
* download url : https://drive.google.com/open?id=0B3vEmk5Bd7lNRmhPSDcxbjhqMFE

### Results
Training accuracy : 0.482
Testing accuracy : not sure

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


