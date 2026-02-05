# Brain Tumor Classifier
![Brain_tumor](brain-tumor-illustrate.jpg)

## 1. Intro to the problem
- A Brain tumor is considered as one of the aggressive diseases, among children and adults. Brain tumors account for 85 to 90 percent of all primary Central Nervous system (CNS) tumors.

- The 5-year survival rate for people with cancerous brain or CNS tumor is approximately 34% for men and 36% for women.

- Brain tumors are classified as: Benign tumor, Malignant Tumor, Pituitary Tumor, etc. Proper treatment, planning, and accurate diagnostics should be implemented to improve the life expectanct of patients.

- The best technique to detect brain tumors is Magnetic Resonance Imaging (MRI). A huge amount of image data is generated through the scans. These images are examined by radiologists. A manual examination can be error-prone due to the level of complexities involved in brain tumors and their properties.

- A brain tumor is considered one of the most aggressive diseases, among children and adults. With manual examination, it can be error-prone due to the level of complexity. Hence, adding Machine Learning (ML) and Artificial intelligence (AI) has consistently shown higher accuracy than manual classification.

## 2. Users and Benefits

## 3. Potential Impact
**Time and Cost Reduction in Healthcare**: facilitates early detection of brain tumors and diagnosis, helps reduce patients' waiting time and save doctors from burn-out.

**Assist doctors in treatment planning**: produces sencond opinion in a time-ly manner can alert the doctors during the treatment planning process and speed uo workflow efficiency

**Better performance Overall**:  When analyzing MRI, there must be a radiologist and a neurosurgeon on-site, hence, an automated system on Cloud can add a valuable second opinion in a timely manner

## 4. Data Source:
- Dataset: [Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) and [MRI Image Data](https://www.kaggle.com/datasets/alaminbhuyan/mri-image-data)

- Description: These 2 datasets contain over a total of 10,000 brain MRI images which are classified into 4 classes: no tumor, glioma, meningioma, and pituitary tumor.

# Brain Tumor Classifier Project

## Overview
This project aims to create and fine-tune different Convolutional Neural Network (CNN) using PyTorch, to find out which one performs the best. 

## How to Run:
- Make sure you have PyTorch and necessary libraries installed. 

- Follow the steps outlined in the notebook, from data loading to model evaluation. After the first few tries, feel free to experiment different parameters. 

## Key Components:
1. Data Loading and Preprocessing: 
During image processing, we'll resize and shuffle the dataset before training. Besides, we'll also experiment with data augmentation to strengthen the performance.
2. CNN model Architecture:
3. Training:
Splitting training and testing dataset at 9:1 ratio and run through different number of epochs to test the efficiency.
4. Evaluation
Evaluate each model on the validation set. 
  
## Technologies Used:
[Python](https://en.wikipedia.org/wiki/Python_(programming_language))

[Pandas](https://en.wikipedia.org/wiki/PANDAS)

[DifPy](https://pypi.org/project/difPy/)

[Tensorflow](https://www.tensorflow.org/)

[Scikit-learn](https://scikit-learn.org/stable/)

[PyTorch](https://pytorch.org/)

[Resnet34](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html)

## Model Accuracy Results:
In this project, we experience 3 different models with increasing complexities to see how the accuracy is improved over time:
- 1 layer of CNN (most basic)
- 9 layers of CNN
- Resnet34: a CNN architectures that is pre-trained on ImageNet Dataset containing 100,000+ images accross 200 different classes

The table below will display various accuracy achieved by diffrent models:

| Type of Tumor | Glioma | Meningioma | No tumor | Pituitary  |
| --- | --- | --- | --- | ---| 
| 1-layer CNN | 92% | 80% | 91% | 91% |
| 9-layer CNN | 88% | 90% | 97% | 93% |
| Resnet34 | 98% | 89% | 96% | 97% |


## Usage
- The future website/app is intended to classify MRI brain tumor into 4 clasess: no tumor, glioma, meningioma and pituitary.
- This website/app should be used as an add-on and shouldn't be considered as professional diagnosis. The main advantage is to give a second opinion in a timely manner

## Next Steps for Model Improvement
To further enhance the performance and capabilities as well as practicality of the CNN model, the following steps are recommended:
1. Model refinement
- Data input: The more data coming in, the better the model will be trained and learn different patterns of MRI. Currently, the model is being trained on a limited dataset, hence, the performance will somehow be limited
- Hyperparameter tuning: higher epochs don't necessary guarantee better performance, looking into playing with batch sizes, learning rates, etc. might give better performance

2. Implement Transfer Learning:
- In this project, we touch base of Transfer Learning with Resnet34. Feel free to experience with other Resnet architectures such as Resnet18, Resnet50, etc.


  
## Contact
| Contact Method | |
| --- | --- |
| Professional Email | dungvn1999@gmail.com |
| LinkedIn | https://www.linkedin.com/in/dungtran99/ |
| Project Link | [https://github.com/jtran2509/brain_tumor]() |

