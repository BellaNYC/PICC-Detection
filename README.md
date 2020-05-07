# PICC (Peripherally Inserted Central Catheter) Detection using Deep Learning

## Introduction
Peripherally Inserted Central Catheter (PICC) is a soft, thin, flexible tube, providing intravenous access and giving treatment. Malposition of tips (with 5-31% incidence) can result in discomfort and pain to patients, and even induce life-threatening complications, such as cardiac arrhythmia. Although the error rate of radiologists analyzing tip location is low, a limited number of experts and high volume of images lead to delays in treatment initiation
<br>
Our goal is to help radiologists improve speed and quality of detecting and make sure PICC position is in the right place.

## Dataset
We collected de-identified HIPAA-compliant (Health Insurance Portability and Accountability Act) DICOM dataset that contain PICCs that obtained from NYU Langone Hospital. All images are unlabeled, we used basic ROI draw function in OsiriX to label the ground truth of PICC and box (safe window), then generated JSON (JavaScript Object Notation) files, which later used to convert COCO format.
<br>
For segmentation maps, we used png format. Input image and segmentation image size are both 1024 * 1024 
<br>
Finally got 600 images labeled in total.

## Methods
1. Transfer learning
2. Loss function: Pixel-wise cross-entropy on eahc pixel
3. Base model: VGG16
4. Semantic segmentation model: FCN, first decompress to capture features and then up sampling to recover spatial information lost in down sampling step
4. Object detection


## Directory and file
data --original images and binary segmentation label file, txt file to record the data used for training
<br>
config --global configuration
<br>
lanenet_data_feed_pipline --feed data and generate tf records
<br>



### References:
https://towardsdatascience.com/semantic-segmentation-popular-architectures-dff0a75f39d0
<br>
https://www.jeremyjordan.me/semantic-segmentation/
