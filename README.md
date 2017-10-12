# CEDL HW1 Deep Classification       
## 黃柏瑜 105061535
## Installation & usage
1. Install pytorch in conda
    ```
    conda install pytorch torchvision cuda80 -c soumith
    ```
2. Clone this repo
3. Download data in ''data'' directory. The structure looks like below
    ```
    ./data/frames
    ./data/labels
    ```

4. Training 
    ```
    python main.py  --batch-size 128  --arch resnet18  --workers 4 --pretrained  --model_name resnet18
    python main.py  --batch-size 128  --arch resnet50  --workers 4 --pretrained  --model_name resnet50
    ```

## Implementation
1. Dataloader
首先，用 pytorch 定義的 Dataset Class ，寫一個HandCam_Dataset。裏面會取出對應的圖片以及標註。
接著對圖片做處理：
將圖片normalize，縮小到224*224大小，再隨機flip圖片做到Data Augmentation
    ```
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],                                  
                                std=[0.229, 0.224, 0.225])
    self.transform = transforms.Compose([
                transforms.Scale(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    ```

2. Model
使用 pytorch 提供的 resnet18 model 和 resnet50 model。並且直接使用已經已經 pretrained 在 imagenet 的參數，這樣開始 train 會比較快。
pretrain model 再最後一層是 Fully-Connected (512 -> 1000) 和 Fully-Connected (2048 -> 1000)，因為 imagenet 有 1000 class，而現在的 dataset 只有 24 class 所以要改成 Fully-Connected (512 -> 24) 和 Fully-Connected (2048 -> 24)

3. Training Procedure
Learning rate : 0.01 (learning rate decay every 5 epoch)
Batch size ： 128
Epoch : 30

## Results
### Best Result

Resnet18 : 70.28%

Resnet50 : 72.3%


### Training Curve

#### Resnet18
<img src="Resnet18_loss" alt="overview" style="float:middle;">
<img src="Resnet18_acc" alt="overview" style="float:middle;">


#### Resnet50
<img src="Resnet50_loss" alt="overview" style="float:middle;">
<img src="Resnet50_acc" alt="overview" style="float:middle;">
