# Traffic Sign Classification Using ResNet

## Project Overview

This project tackles the challenge of traffic sign classification using a ResNet-based Convolutional Neural Network (CNN). The model is trained and tested on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which consists of over 50,000 images across 43 traffic sign categories. The primary focus of this project is to address class imbalance issues in the dataset using data augmentation techniques.

## Key Features

- Model Architecture: A ResNet-inspired architecture was implemented, consisting of convolutional layers, residual blocks, batch normalization, and ReLU activations. The model is designed to handle the varying sizes, orientations, and lighting conditions present in the traffic sign images.
- Class Imbalance Handling: Data augmentation techniques, including color jittering and Gaussian blurring, were applied to undersampled classes to mitigate the class imbalance problem and improve model generalization.

## Experiments:

- Without Augmentation: The model was trained on the original dataset.
- With Augmentation: The model was trained on an augmented dataset to address the imbalance issue.

## Dataset

The GTSRB dataset consists of 39,209 training images and 12,630 test images, with a significant class imbalance where some classes have far fewer samples. Data preprocessing included resizing images to 112x112, normalization, and augmenting underrepresented classes.

## Model Architecture

The model follows a ResNet-based structure, including:

- Initial convolutional layers with ReLU activation and batch normalization.
- Six residual blocks, each designed with identity mappings and projection shortcuts for optimal feature learning.
- A fully connected layer with 43 output classes representing the traffic sign categories.
- Softmax activation for the final layer to generate classification probabilities.

## Results

- Without Augmentation:
    - Accuracy: 92.98% on the test set.
    - Challenges: The model struggled with some classes due to the class imbalance, particularly with classes 27 and 34, which had very few training samples.

- With Augmentation:
    - Accuracy: 95.62% on the test set.
    - Improvements: The data augmentation strategy effectively improved the model’s accuracy, especially for classes that had fewer examples. The model achieved a more balanced performance across all classes, though some classes (e.g., 21 and 22) still posed difficulties.

## Conclusion

This project demonstrates the effectiveness of a ResNet-based model in handling the traffic sign classification task, especially when combined with data augmentation techniques to address class imbalances. The model achieved a substantial improvement after applying augmentation, suggesting that enhancing the dataset’s variety is critical for generalization in imbalanced classification problems.

Future work could involve refining the augmentation techniques, exploring more complex network architectures, and testing regularization methods to further improve the model's performance.

## References
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
- Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.


## Notebook

The complete implementation of the project is presented as a Jupyter notebook. The notebook is divided into three parts:

1. Defining functions and classes
2. Experiment 1: training without data augmentation
3. Experiment 2: training with data augmentation

The first part has all the required functions and classes to perform the experiments. The following parts are just executions of the experiments using the functions provided in 1.

For the notebook to run, it needs to find a path to the train and test data as follows:

- Project.ipynb
- Final_Training
    - Images
        - 00000
            - 00000_00000.ppm
            - ...
- Final_Test
    - Images
        - 00000.ppm
        - 00001.ppm
        - ...
- Final_Training_Augmented
    - Images
        - 00000
            - 00000_00000.ppm
            - ...
- GT-final_test.csv

Final_Training, Final_Test, and GT-final_test.csv are the files directly downloaded from 

https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

Final_Training_Augmented is just a copy of Final_Training that will store the new augmented dataset.