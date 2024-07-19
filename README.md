# Fashion Mnist training TensorFlow JS

## These tasks solved with this repo

- Training a model on Fashion Mnist dataset using `build_model.js`
- Testing the model on test images using `test_model.js`
- Transfer learning on a new model using the last model with `transfer_learning.js`

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)

## Introduction

Fashion MNIST is a dataset of Zalando's article images consisting of 70,000 28x28 grayscale images in 10 categories, with 7,000 images per category. The goal of this project is to classify these images into their respective categories using a neural network implemented in TensorFlow.js.

## Setup

### Prerequisites

- Node.js (v12 or later)
- npm (v6 or later)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/matin-ghorbani/Fashion-MNIST-training-TensorFlow-JS.git
    cd Fashion-MNIST-training-TensorFlow-JS
    ```

2. Create a new package.json:

    ```bash
    npm init -y
    ```

3. Install the dependencies:

    ```bash
    npm install @tensorflow/tfjs-node
    npm install jimp
    ```

4. Place the Fashion MNIST dataset CSV files (`fashion-mnist_train.csv` and `fashion-mnist_test.csv`) in the `data` directory. You can download the dataset from [here](https://dax-cdn.cdn.appdomain.cloud/dax-fashion-mnist/1.0.2/fashion-mnist.tar.gz).

## Usage

### Training the Model

To train the model, run the following command:

```bash
node build_model.js
```

### Testing the Model

To test the model, run the following command:

```bash
node test_model.js /path/to/image.jpg
```

### Using transfer learning to train the Model

To train the model by transfer learning, run the following command:

```bash
node transfer_learning.js
```

## Results

After training, the model achieves the following results on the test set:

- Test-set loss: ***0.0124***
- Test-set accuracy: ***0.9979***
