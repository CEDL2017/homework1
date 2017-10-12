# Your Name <span style="color:red">(id)</span>

#Homework 1: Deep Classification

## Overview
The project is related to 
> Recognition from Hand Cameras: A Revisit with Deep Learning [Chen et. al. 2016]


## Implementation
1. Single-Stream ResNet50
	* ResNet 可以更快將 gradients 傳到前面的 layers，ImageNet Top 1/ Top 5 Classification 也有較好的 accuracy。 因此將 paper 中使用的 AlexNet 換成 ResNet。
	* item
2. Two-Stream ResNet18

```
# Creates the network model for transfer learning
def create_model(base_model_file, feature_node_name, last_hidden_node_name, num_classes, input_head, input_hand, freeze=False):
    # Load the pretrained classification net and find nodes
    base_model   = load_model(base_model_file)
    feature_node = find_by_name(base_model, feature_node_name)
    last_node    = find_by_name(base_model, last_hidden_node_name)

    # Clone the desired layers with fixed weights
    cloned_layers_head = combine([last_node.owner]).clone(
        CloneMethod.freeze if freeze else CloneMethod.clone,
        {feature_node: placeholder(name='features_head')})

    cloned_layers_hand = combine([last_node.owner]).clone(
        CloneMethod.freeze if freeze else CloneMethod.clone,
        {feature_node: placeholder(name='features_hand')})

    # Add new dense layer for class prediction
    feat_norm  = input_head - Constant(114)
    cloned_out_head = cloned_layers_head(feat_norm)
    z_head = Dense(2048, activation=None, name='Head_stream') (cloned_out_head)

    feat_norm  = input_hand - Constant(114)
    cloned_out_hand = cloned_layers_hand(feat_norm)
    z_hand = Dense(2048, activation=None, name='Hand_stream') (cloned_out_hand)

    z = C.layers.Sequential([
          Dense(4096, activation=None, name='Conca'),
          Dense(2048, activation=None, name='fc7'),
          Dense(num_classes, activation=None, name=new_output_node_name)
         ])(C.splice(z_head, z_hand, axis=0))

    return z
```

## Installation
* Other required packages: CNTK v2.2
* How to compile from source? python train.py

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


