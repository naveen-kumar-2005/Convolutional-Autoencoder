# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
Noise is a common issue in real-world image data, which affects performance in image analysis tasks. An autoencoder can be trained to remove noise from images, effectively learning compressed representations that help in reconstruction. The MNIST dataset (28x28 grayscale handwritten digits) will be used for this task. Gaussian noise will be added to simulate real-world noisy data.

## DESIGN STEPS
## STEP 1:
Import necessary libraries including PyTorch, torchvision, and matplotlib.

## STEP 2:
Load the MNIST dataset with transforms to convert images to tensors.

## STEP 3:
Add Gaussian noise to training and testing images using a custom function.

## STEP 4:
Define the architecture of a convolutional autoencoder:

Encoder: Conv2D layers with ReLU + MaxPool

Decoder: ConvTranspose2D layers with ReLU/Sigmoid

## STEP 5:
Initialize model, define loss function (MSE) and optimizer (Adam).

## STEP 6:
Train the model using noisy images as input and original images as target.

## STEP 7:
Visualize and compare original, noisy, and denoised images.


## PROGRAM
### Name: Naveen Kumar.R
### Register Number: 212223230139

```py
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid()
        )



    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```
```py
model = DenoisingAutoencoder().to(device)
criterion =nn.MSELoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)
```
```py
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader)}")

```
```py
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: Naveen Kumar.R")
    print("Register Number: 212223230139")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
```

## OUTPUT

### Model Summary

<img width="825" height="373" alt="image" src="https://github.com/user-attachments/assets/0a005544-4686-4a5d-9776-e8176dd07f3f" />



### Original vs Noisy Vs Reconstructed Image

<img width="1793" height="650" alt="image" src="https://github.com/user-attachments/assets/1751e3ae-8e72-4468-8be3-f438814c35cb" />





## RESULT
The convolutional autoencoder was successfully trained to denoise MNIST digit images. The model effectively reconstructed clean images from their noisy versions, demonstrating its capability in feature extraction and noise reduction.
