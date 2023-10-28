# Comment-Toxicity-Model


This repository contains code for a toxic comment classification model using TensorFlow. The code is designed to classify text comments into various toxic categories. The dataset used is from the Kaggle competition [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

## Dependencies

The code relies on the following dependencies:

- `os`
- `pandas`
- `tensorflow`
- `numpy`
- `gradio`

Make sure these dependencies are installed in your environment before running the code.

## Usage

### 1. Loading and Preprocessing the Dataset

The initial steps involve loading and preprocessing the dataset. The code uses the `pandas` library to load the data from the provided CSV file (`train.csv`). The `os` library is used to specify the file path. The loaded dataset is then split into input text (`X`) and target variables (`y`).

### 2. Text Vectorization

Text vectorization is an important step in natural language processing tasks. The code utilizes the `TextVectorization` layer from TensorFlow to convert the input text into numerical representations. The `max_tokens` parameter limits the vocabulary size, and the `output_sequence_length` parameter sets the fixed length of the sequence. The resulting vectorized text is stored in `vectorized_text`.

### 3. Data Preparation and Batching

The code prepares the dataset for training by performing several steps. It uses TensorFlow's `from_tensor_slices` function to create a slice dataset from the vectorized text and the labels (`y`). Additionally, it applies caching (`cache`), shuffling (`shuffle`), batching (`batch`), and prefetching (`prefetch`) to optimize data processing. The prepared dataset is divided into training, validation, and testing sets.

### 4. Model Architecture

The model architecture is established using TensorFlow's `Sequential` API. It consists of an embedding layer, a bidirectional LSTM layer, fully connected layers for feature extraction, and a final classification layer. The model is compiled with the `BinaryCrossentropy` loss function and the `Adam` optimizer.

### 5. Model Training

The model is trained using the training and validation datasets. The training history, including loss and validation metrics, is stored in `history`.

### 6. Model Evaluation

After training, the code demonstrates model evaluation using the testing dataset. It computes precision, recall, and accuracy metrics using TensorFlow's metrics classes for binary classification. The resulting metrics are printed.

### 7. Gradio Interface

The code includes a Gradio interface for real-time scoring of comments. The `score_comment` function takes an input comment, vectorizes it, and generates predictions for each toxic category. The interface allows users to input comments and see the predicted toxic categories.

### 8. Model Saving and Loading

The model is saved in the HDF5 format using the `save` method. It is then loaded from the saved file using the `load_model` function.

