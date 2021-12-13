# CycleGAN
A clean, simple and readable implementation of CycleGAN in PyTorch. I've tried to replicate the original paper as closely as possible, so if you read the paper the implementation should be pretty much identical. The results from this implementation I would say is on par with the paper, I'll include some examples results below.

## Results
The model was trained on Real Photo <-> Monet Style Photo dataset.

|1st column: Input / 2nd column: Generated / 3rd row: Re-converted|
|:---:|
|![](saved_images/generated_monet_17.png)|
|![](results/realimage_17.png)|


### Monet Style Dataset
The dataset can be downloaded from Kaggle: [link](https://www.kaggle.com/c/gan-getting-started/data).
Split the dataset into 2 folders "train" and "val" and add all monet style images inside "monet" folder and the real images inside "real" forlder.
The folder tree will look something like this:
├── monet_data
│   ├── train
│   │   ├── monet
│   │   └── real
│   ├── val
|   |   ├── monet
|   |   └── real

### Using pretrained weights
Clone the repository and put the pth.tar files in the directory with all the python files. Make sure you put LOAD_MODEL=True in the config.py file.

### Training
Edit the config.py file to match the setup you want to use. Then run train.py

## CycleGAN paper
### Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks by Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros

```
@misc{zhu2020unpaired,
      title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks}, 
      author={Jun-Yan Zhu and Taesung Park and Phillip Isola and Alexei A. Efros},
      year={2020},
      eprint={1703.10593},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
