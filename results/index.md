# 高穎  106062525 <span style="color:red">(id)</span>

#Project 5: Deep Classification

## Overview
The project is related to 
> tensorflow
> vgg16


## Implementation
1. tensorflow_finetune
	用ImageNet pretrain好的vgg16參數,前七層不改,最後一層fc重新train.每train一個epoch就把train好的權重記下來
2. load_datas
	load training和test的label
3.tensorflow_test
	用記下來的權重加上test的檔案計算準確度


## Installation
* 用anaconda建一個有tensorflow的虛擬環境,利用spyder編譯

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


