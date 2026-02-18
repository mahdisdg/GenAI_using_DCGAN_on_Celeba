# Photorealistic Face Generation using DCGAN

This repository implements a Deep Convolutional Generative Adversarial Network (DCGAN) to synthesize high-fidelity human face images. The model is trained on the CelebA (Celebrity Attributes) dataset, a large-scale face attributes dataset with more than 200,000 celebrity images.

The goal of this project was to leverage architectural constraints (strided convolutions and batch normalization) to achieve stable adversarial training and generate diverse, realistic human features from random noise.



## 🌟 Highlights
* Architectural Stability: Implements the core DCGAN principles—replacing pooling layers with strided convolutions and using Batch Normalization to stabilize the learning process.
* High-Fidelity Output: Generates $64 \times 64$ RGB images that capture complex facial features including hair texture, lighting, and skin tones.
* Visual Training Progress: Includes scripts to generate animated GIFs that visualize the "evolution" of the generator from random static to structured facial features.
* Efficient Pipeline: Optimized data loading and preprocessing for the 200k+ image CelebA dataset using PyTorch's DataLoader.

## 🛠 Tech Stack
* Deep Learning: PyTorch (Tensors, Autograd, NN Module)
* Data Processing: NumPy, PIL (Pillow)
* Visualization: Matplotlib, ImageIO (for GIF generation), tqdm

## 📐 Model Architecture

### Generator
The Generator takes a 100-dimensional latent vector $z$ and performs a series of fractionally-strided convolutions (transpose convolutions).
- Activations: ReLU for hidden layers, Tanh for the output layer to match normalized pixel values.
- Normalization: Batch Normalization after every layer to prevent the generator from collapsing to a single point.

### Discriminator
The Discriminator is a deep CNN that acts as a binary classifier.
- Activations: LeakyReLU (slope = 0.2) to prevent "dying neurons" and assist gradient flow back to the generator.
- Downsampling: Uses strided convolutions instead of Max Pooling to allow the network to learn its own spatial downsampling.



## 📈 Training Process
The training follows a minimax game where the Discriminator is trained to maximize the probability of assigning the correct label to both training examples and samples from G, while the Generator is trained to minimize $\log(1 - D(G(z)))$.

1.  Normalization: Input images are normalized to a $[-1, 1]$ range.
2.  Optimizers: Two separate Adam optimizers ($\beta_1 = 0.5, \beta_2 = 0.999$) are used to maintain the delicate balance between the two networks.
3.  Progression: The model undergoes several epochs, starting from abstract blobs and refining into recognizable human faces.

## 📁 Repository Structure
* Q4new.ipynb: Full implementation including data loading, model definition, training loops, and visualization.
* dcgan_final.gif: An animation showing the generator’s progress throughout the training iterations.
* samples/: Directory containing generated face samples at different stages of training.

## ⚙️ How to Run
1.  Clone the repo:
   
    git clone [https://github.com/your-username/CelebA-DCGAN-Synthesis.git](https://github.com/mahdisdg/GenAI_using_DCGAN_on_Celeba.git)
    
2.  Dataset: Download the CelebA dataset from Kaggle and place the images in the /data folder.
3.  Execute: Run the cells in main.ipynb to train the model or generate new faces using pre-saved weights.
