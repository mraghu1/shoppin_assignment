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

Usage

1. Train the Autoencoder

Train the autoencoder on the CIFAR-10 dataset:

train_model(model, train_loader, criterion, optimizer, epochs=20)

2. Extract Latent Vectors

Extract latent vectors for images in the test dataset:

latent_vectors, test_images = extract_latent_vectors(model, test_loader)

3. Perform Similarity Search

Find the top-k similar images to a query image:

top_indices = find_similar_images(query_idx, latent_vectors, test_images, top_k=5)
display_query_and_results(query_image, top_indices, test_images, top_k=5)

4. Evaluate Similarity Search

Calculate Precision, Recall, and Retrieval Accuracy:

evaluate_similarity_search(model, latent_vectors, test_loader, top_k=5)

Project Structure

autoencoder.py: Contains the autoencoder model definition and training loop.

similarity_search.py: Functions for latent vector extraction, similarity search, and evaluation.

visualization.py: Utility for displaying query and similar images.

Results

The autoencoder achieves effective compression and reconstruction of images. The similarity search identifies images that are visually similar to the query image with reasonable precision and recall.

Future Enhancements

Use more advanced autoencoder architectures like Variational Autoencoders (VAEs) or Denoising Autoencoders.

Apply this approach to larger and more complex datasets (e.g., ImageNet).

Experiment with different similarity metrics and dimensionality reduction techniques.
