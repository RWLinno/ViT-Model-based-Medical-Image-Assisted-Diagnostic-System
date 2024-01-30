# My Vision Transformer

Github :[https://github.com/RWLinno/ViT-Model-based-Medical-Image-Assisted-Diagnostic-System](https://github.com/RWLinno/ViT-Model-based-Medical-Image-Assisted-Diagnostic-System)



### Environments

You can replicate the environment I was experimenting in with the following commands:

1. Creating a Virtual Environment

```
conda create -n ViT python==3.7.16
conda activate ViT
```

2. Quickly install the dependencies with the following command (or torch is all your need)

```
pip3 install -r requirements.txt
```

3. Download the **flower_photos dataset**, or you can just use the code in train.ipynb to unzip `flower_photos.tar`

```
gdown https://drive.google.com/uc?id=1J5UryTNkXDSEpbmPoMH3Hry9iRXwaSES
```

4. You can optionally download my pre-trained model and put it into the `models` folder

```
# ViT_pre_train_5_epochs.pth
gdown https://drive.google.com/uc?id=1ejwfSjadBnxJy2-Q5sZUbJFzt6F3jz_y

# ViT_pre_train_10_epochs.pth
gdown https://drive.google.com/uc?id=1cRpmA3fGrHx_mOIdf9ZFBPBW6dIubKhx

# ViT_pre_train_10_epochs.pth
gdown https://drive.google.com/uc?id=19kc-YlXcjNzkBQ8iJkPX3OklSawt_myS
```



### Introduction

Our project files are structured as follows

```
MyViT/
│───README.md
└───data/
│   │───flower_photos/
│   │   │   daisy/
│   │   │   dandelion/
│   │   │   roses/
│   │   │   sunflowers/
│   │   │   tulips/
│   │   ...
│   └───samples/
└───models/
└───pic/
│   Mydataset.py
│   ViT.py
│   utils.py
│   train.ipynb
│   prediction.ipynb
│   flower_photos.tar
│   ...
```

- We've only trained on the **flower_photos** dataset so far, but I'd like to train the medical dataset when I have time!

  

### Training

- open your IDEs to run **train.ipynb** to traning a model based on your dataset
  - I recommend jupyter lab or vscode with the extension 'jupyter'
  - In the code you can adjust the following parameters yourself
    - dataset_path	
    - batch_size
    - epoch_num
    - learning_rate(relatively unimportant because we adaptively update the learning rate)
- Here's a preview of some of our training

![distribution](./pic/class_distribution.png)

![training_process](./pic/training_process.png)

### Prediction

**train.ipynb** provides an example of predicting a single image using a pre-trained model, please modify it for your own dataset!

![sample](./pic/sample_pic.jpg)

![prediction](./pic/prediction.jpg)



# Use ViT on the ChestXRay dataset

public dataset link: https://data.mendeley.com/datasets/rscbjbr9sj/2

you can use the same way to load the dataset and you get **2** folders:

```
MyViT/
│───README.md
└───data/
│   │───ChestXRay/
│   │   │   NORMAL/
│   │   │   PNEUMONIA/
```

 

run`chest_main.ipynb` and that includes what we coded for the training and prediction of flower photos.



I should tell you to change the support suffix in `utils.py` so we can deal with these pictures.



##### Results

- **Distribution**:  for NORMAL and PNEUMONIA, totally 2 classes.

![image-20240130190633825](https://s2.loli.net/2024/01/30/wfplyo7WcaXLD9h.png)

- **Training model**

you can modify the parameters such as epoch_num or learning_rate so that you can achieve a better accuracy and lower loss.

![image-20240131010554874](https://s2.loli.net/2024/01/31/23JYbfFdo9xj4uE.png)

- **Prediction ChestXRAY pictures**

![image-20240131010817488](https://s2.loli.net/2024/01/31/gB3sobLAZh1FcHN.png)

![image-20240131010827258](https://s2.loli.net/2024/01/31/n8sTcRBU4K5ku92.png)

Above are the sample and the probability.



you can download my pre-trained model for ChestXRAY visual data.

```
# ViT_for_chest.pth
gdown https://drive.google.com/uc?id=1YRYNG4uvMyojk3yS77_umATW_MZfm8M3
```





### Acknowledgment

##### Contributors

- [RWLinno](https://github.com/RWLinno)
- [Aritst](https://github.com/IcecreamArtist)



##### References

- [WZMIAOMIAO](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master)
- https://www.sohu.com/a/677833784_121119001
- https://zhuanlan.zhihu.com/p/385406085