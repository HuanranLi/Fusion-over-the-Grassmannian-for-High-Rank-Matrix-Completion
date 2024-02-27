import torch
import numpy as np
from torchvision import datasets, transforms

def MNIST(classes, images_per_class):
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root='../datasets', train=True, download=True, transform=transform)

    images_class_array = []
    labels_array = []
    for classes_i in classes:
        print('Sampling Class', classes_i)
        # Filter the dataset for two classes, e.g., classes 0 and 1
        class_0 = [data for data, target in mnist_dataset if target == classes_i]

        # Sample 100 images from each class
        images_class_array.append( torch.stack(class_0[:100]) )

        labels = np.ones(100) * classes_i
        labels_array.append(labels)


    # Concatenate the samples from both classes
    concatenated_samples = torch.cat(images_class_array, dim=0)

    # Reshape (vectorize) the images: flatten each 28x28 image into a 784-dimensional vector
    vectorized_images = concatenated_samples.view(-1, 28*28)

    # Convert the tensor to a NumPy array
    images_np = vectorized_images.numpy().T

    labels_np = np.concatenate(labels_array)

    print("Images shape:", images_np.shape)
    print("Labels shape:", labels_np.shape)

    return images_np, labels_np
