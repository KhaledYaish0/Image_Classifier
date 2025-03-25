# Image_Classifier_Project

Image Classifier for Flower Species

This project was developed as part of the Udacity Deep Learning Nanodegree. It focuses on building an image classification model that accurately identifies different species of flowers from images using deep learning techniques. The core framework used for this project is TensorFlow.

The classifier leverages the power of transfer learning by using pre-trained convolutional neural networks (CNNs) such as VGG16 or ResNet. These networks were fine-tuned on a dataset containing various categories of flowers, enabling the model to learn high-level image features relevant to plant classification.

The training process includes data preprocessing, augmentation, and model optimization using GPU acceleration when available. After training, the model is saved and can be used to predict the species of a flower from a new image. A command-line interface was also implemented to allow users to load an image and return the top predicted flower species along with their probabilities.

This project demonstrates a practical application of deep learning in the field of computer vision and serves as a foundation for more complex image recognition systems.
