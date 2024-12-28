# shoppin_assignment
Autoencoder-Based Image Similarity Search

This project demonstrates how to use an autoencoder for image similarity search. The autoencoder compresses images into latent vectors (feature embeddings) and uses these representations to identify similar images based on cosine similarity.

Features

Autoencoder Architecture: Includes a convolutional encoder to compress images and a decoder to reconstruct them.

Image Similarity Search: Utilizes cosine similarity on latent vectors to find visually similar images.

Evaluation Metrics: Precision, Recall, and Retrieval Accuracy are calculated to evaluate the performance of the similarity search.

Visualization: Displays query images along with the top-k most similar images.

Datasets

The project uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 classes (e.g., airplanes, cars, birds, cats).

Requirements

Python 3.8+

PyTorch

torchvision

numpy

matplotlib

scikit-learn

Install dependencies using:

pip install torch torchvision numpy matplotlib scikit-learn


