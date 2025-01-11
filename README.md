## Image Inpainting and Image Merging Program Based on EdgeConnect Model
Hu Junyu| Li Xucheng| Li yi| Chen Junfeng

### Introduction:
This paper will use the EdgeConnect model for image inpainting training, enabling the model to generate edge images based on the input images and perform inpainting tasks accordingly. At the same time, we will attempt to combine the edge images of two pictures after generation and then proceed with subsequent image completion, aiming to seamlessly merge the two images. This approach will be our innovative point. It will enhance the repair effect of damaged photos with reference images and can also serve as a new method for photo editing.
## Prerequisites
- Python 3
- PyTorch 2.0
- NVIDIA GPU + CUDA cuDNN

## Datasets
### 1) Images
We use [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets. CelebA is a large-scale dataset for facial recognition and attribute classification tasks. It contains 202,599 celebrity images, each annotated with 40 attribute labels, such as age, gender, hair color, and other facial characteristics. The images are high-resolution, collected from the web, and include variations in pose, lighting, and background. CelebA is widely used in the computer vision community for training and evaluating models related to facial recognition, attribute prediction, and image inpainting tasks.

You should split the dataset into three part that will function as training dataset, testing dataset and evaluating dataset. Use "data_split.py" to help you do the work.

After spliting the dataset, you will have to use the following orders in terminal to generate documents in the form of ".flist" for the model as guidances to use the dataset. 

```bash
mkdir datasets
python ./scripts/flist.py --path path_to_places2_train_set --output ./datasets/places_train.flist
```

### 2) Masks generate
Since the model will need to learn about the masked images, so masked datasets are reqiured. Addopt "data_mask" on each dataset to generate the masked images. The detailed path to  your datasets should be changed based on where you actually save your datasets.

## Getting Started
### 1) Training
To train the model, rewrite the `config.yaml` file under your checkpoints directory. You need to fill in the corresponding.flist file path according to the instructions in the document.

EdgeConnect is trained in three stages: 1) training the edge model, 2) training the inpaint model and 3) training the joint model. To train the model:
```bash
python train.py --model [stage] --checkpoints [path to checkpoints]
```

For example to train the edge model on celeba dataset under `./checkpoints/celeba` directory:
```bash
python train.py --model 1 --checkpoints ./checkpoints/celeba
```

You can set the number of training iterations by changing `MAX_ITERS` value in the configuration file.

### 2) Testing
To test the model, create a `config.yaml` file under your checkpoints directory. 

You can test the model on all three stages: 1) edge model, 2) inpaint model and 3) joint model. In each case, you need to provide an input image (image with a mask) and a grayscale mask file. Please make sure that the mask file covers the entire mask region in the input image. To test the model:
```bash
python test.py \
  --model [stage] \
  --checkpoints [path to checkpoints] \
  --input [path to input directory or file] \
  --mask [path to masks directory or mask file] \
  --output [path to the output directory]
```

### 3) Evaluating
To evaluate the model, you need to first run the model in [test mode](#testing) against your validation set and save the results on disk. We provide a utility [`./scripts/metrics.py`](scripts/metrics.py) to evaluate the model using PSNR, SSIM and Mean Absolute Error:

```bash
python ./scripts/metrics.py --data-path [path to validation set] --output-path [path to model output]
```

### Generate damaged images
To generate damaged images for the model to fix, you will need to use "try.py" in the file src, it will gengerate the mask lines for the image you input. Remember to change the path of the image that you input, the output mask image will be saved under the file src, you can remove it to the file: data/masks. Then, use "init_png.py" to combine the image with its mask image, then you can move it to the file data/images.

### Fix the damaged image

You can use the terminal order below to fix the image you just created.

```bash
python test.py --model [mode]
--checkpoints [path to your .flist documents] 
--input [path to the damaged image]
--mask [path to the mask image] 
--output [path to save the output]
```

If you wan to generate edge images, the model mode should be "1"; if you want to color the edge image, the mode should be "2"; if you want to fix the damaged image directly, use mode "3".


## Our work
Since this project is implemented based on other people's models, the parts completed or modified independently by us are listed below:

**Independently Completed Parts**:
(1) init_png.py
(2) date_split.py
(3) data_mask.py
(4) try.py

**Modified Parts**:

**In models.py**:
(1) In the `__init__()` function of the `EdgeModel` class, code for confirming the save path has been added.
(2) In the `process()` function of the `EdgeModel` class, in order to find the cause of the color tone problem of the output image, code for outputting the shape and pixel values of the image has been added, and attempts have been made to add code for saving the intermediate generated edge images.
(3) In the `EdgeModel` class, the `imsave()` code and the `save_generated_edges()` function have been added to capture and save edge images.

**In util.py**:
(1) The `imsave()` function has been modified. Code has been added to check whether `img` is of the `torch.Tensor` type. If it is, it will be converted to `numpy.ndarray`.

This work is based on the papers <a href="https://arxiv.org/abs/1901.00212">EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning</a> and <a href="http://openaccess.thecvf.com/content_ICCVW_2019/html/AIM/Nazeri_EdgeConnect_Structure_Guided_Image_Inpainting_using_Edge_Prediction_ICCVW_2019_paper.html">EdgeConnect: Structure Guided Image Inpainting using Edge Prediction</a>.
