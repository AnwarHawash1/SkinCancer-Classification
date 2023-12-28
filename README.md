# SkinCancer-Classification
## --------------------------------------------------------:-Anwar-:--------------------------------------------------------
This code is a comprehensive implementation of a skin cancer classification model using TensorFlow and Keras. The code includes functions for setting up the environment, preparing train-test data, defining the model architecture, training the model, testing and evaluating the model, and visualizing the training curves. Additionally, it includes functions for loading and processing image data, as well as data augmentation.

Here's a breakdown of the code:

1. **Environment Setup:**
   - Import necessary libraries and suppress warnings.
   - Import TensorFlow, NumPy, Pandas, and other relevant libraries for data manipulation and visualization.

2. **Train-Test Data Preparation Function:**
   - Function to split the data into training and testing sets.
   - Utilizes ImageDataGenerator for data augmentation.

3. **Model Architecture Function:**
   - Function to create a Convolutional Neural Network (CNN) model using TensorFlow/Keras.
   - The model consists of convolutional layers, max-pooling layers, and fully connected layers.

4. **Model Training Function:**
   - Function to train the created model using training data.
   - Implements early stopping and learning rate reduction as callbacks.

5. **Model Testing and Evaluation Function:**
   - Function to evaluate the model on the test set.
   - Displays accuracy, classification report, and sample predictions.

6. **Training Curve Plotting Function:**
   - Function to plot the training and validation curves for accuracy and loss over epochs.

7. **Image Loading and Resizing Function:**
   - Function to load and resize images.

8. **Data Loading and Initial Processing:**
   - Reads CSV files containing metadata and ground truth labels.
   - Processes and prepares the data for training.

9. **Constants and Dictionaries Setup:**
   - Defines constants and dictionaries for skin lesion types.

10. **Load Processed Data and Image Path Mapping:**
   - Loads processed data and maps image paths.

11. **Sample Image Display:**
   - Displays sample images for each skin lesion type.

12. **Label Mapping and Sorting:**
   - Maps labels to integers and sorts the data accordingly.

13. **Data Augmentation:**
   - Augments the data by replicating images for certain classes.

14. **Data Overview and Shape Comparison:**
   - Prints the shapes of the original and augmented data.

15. **Prepare Augmented Data for Model Training:**
   - Prepares the augmented data for training.

16. **One-Hot Encode Labels:**
   - Converts integer labels to one-hot encoded labels.

17. **Create Model Architecture:**
   - Creates the CNN model.

18. **Train-Test Data Split for Augmented Data:**
   - Splits the augmented data into training and testing sets.

19. **Model Training:**
   - Trains the model on the augmented data.

20. **Save Model:**
   - Saves the model weights, architecture, and a TensorFlow Lite version.

21. **Plot Model Training Curve:**
   - Plots the training and validation curves.

22. **Link to Download Model Weights and TensorFlow Lite Model:**
   - Provides links to download the saved model weights and TensorFlow Lite model.

23. **Multilabel Confusion Matrix and Visualization:**
   - Evaluates the model using a multilabel confusion matrix and visualizes the results.

Overall, the code provides a complete pipeline for building, training, and evaluating a skin cancer classification model using CNNs.
