# TUFT Dental Image Segmentation using Unet-Resnet 
Biomedical image processing often involves tasks like localization, where each pixel in an image needs a class label assigned. Traditionally, this might involve training a network in a sliding-window fashion. However, this approach has limitations:

**1.Limited Localization:** Patch-based training with a sliding window might not capture long-range dependencies within the image, hindering precise localization.
**2.Data Imbalance:** Training with patches can lead to a much larger volume of training data compared to the number of whole images, which can skew the training process.

**U-Net** addresses these challenges by offering a convolutional neural network architecture specifically designed for image segmentation. Its key features include:
**1.U-Shaped Architecture:** This structure allows for efficient feature extraction through contracting paths (encoder) and precise localization through expansive paths (decoder) with skip connections.
**2.Skip Connections:** These connections directly copy information from the encoder to the decoder at corresponding resolutions. This preserves spatial details crucial for accurate localization, even after down-sampling in the encoder.

**ResNet**, specifically ResNet-34, is a frequently used backbone architecture for U-Net. It offers several advantages:

**1.Mitigating Vanishing Gradients** by utilizing skip connections that jump over some layers, ResNet helps address the vanishing gradient problem, a common challenge in training very deep networks.
**2.Deep Network Training** This architecture allows for training deeper networks compared to traditional architectures, leading to significant performance improvements in various image segmentation tasks.
**3.Simplicity and Scalability** ResNet-34 is known for its balanced design, offering both simplicity and the ability to be scaled to handle more complex tasks.

On this chance I will share my experiment when I using Unet-Resnet34 with :
1.Unet-Resnet34 (batch size: 32)
2.Unet-Resnet34 (batch size: 64)
Hyperparameter : 
-Input Size: 256 x 512
-Batch Size: 32 & 64
-Optimizer: Adam
-Learning Rate: 1e-3
-Scheduler: Reduce On Plateau
-Factor: 0.5
-Patience: 5
-Early Stopping: Patience 20

Then, change Hyperparameter to :
-Learning Rate: 5e-4
-Scheduler: Reduce On Plateau
-Factor: 0.2
-Patience: 10

Using same model 
3.Unet-Resnet34 (batch size: 32)
4.Unet-Resnet34 (batch size: 64)

**Metrics Evaluation**
Dice Coefficient: Measures the overlap between the predicted segmentation and the ground truth. It is particularly useful for imbalanced datasets.
Pixel Accuracy: Computes the percentage of correctly classified pixels in the entire image.
IoU (Intersection over Union): Measures the intersection between the predicted segmentation and the ground truth divided by their union, providing a robust evaluation of segmentation performance.

**Model Performance Summary**
This table summarizes the performance metrics (Dice Coefficient, IoU, Pixel Accuracy, and Inference Speed) on the test set for the U-Net model with different backbones. The models are ranked from highest to lowest average score based on Dice Coefficient, IoU, and Pixel Accuracy.
![experiment_indonesia_ai](https://github.com/AndhikaNugRah/DentalSegmentationUnetResnet/assets/158553151/7a82eb6f-cd49-40eb-a56d-547a5a9b369d)

Based on the results, the U-Net with ResNet34 and a batch size of 32 is recommended for its stability performance, making it highly efficient for real-time applications.

**Prediction Results**
![image](https://github.com/AndhikaNugRah/DentalSegmentationUnetResnet/assets/158553151/67845869-5dd1-4e24-a19c-0fbe6a4a7d56)

This section highlights the performance of the U-Net model with ResNet34 on the test dataset. Here, you can observe the original image, the ground truth mask, and the model's prediction. The model effectively captures details, achieving a Dice Coefficient of 0.907757, IoU of 0.83224, and Pixel Accuracy of 0.977755 on the test set.

This model can be embedded in digital X-ray machines to assist dentists in analyzing routine X-rays by providing real-time feedback during radiograph capture. This can potentially improve workflow efficiency and potentially reduce analysis time.
