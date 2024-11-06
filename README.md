# NLP-DA-PROJECT
This project utilizes DistilBERT to classify tweets as disaster-related or non-disaster-related, providing a lightweight and efficient NLP solution. Follow the steps below to set up the environment and execute the code.

📋 Requirements
Python: Version 3.6 or higher
Libraries: Install the required libraries by running:

bash
pip install pandas scikit-learn transformers torch

Dataset: Ensure you have train.csv and test.csv files for training and testing. Place these files in the working directory or update the file paths in the code.
🚀 Steps to Execute
1. Set Up Environment
Install the required libraries as outlined above.
Confirm that train.csv and test.csv are available in the directory you’re working in.

3. Load and Preprocess the Data
Start by loading the data using the specified paths.
Preprocess the text data by normalizing, removing special characters, and preparing it for tokenization.

4. Tokenize and Encode the Text
Use the DistilBERT tokenizer to transform the text data into model-compatible format.
Set max_length to truncate and pad sequences to a consistent length across all samples.

5. Initialize and Configure the Model
Load the DistilBERT model pre-trained for sequence classification.
Configure training parameters such as batch_size, num_train_epochs, and learning_rate.

6. Train the Model
Run the training code to fine-tune DistilBERT on the disaster tweet dataset.
Training progress, including loss values, will be displayed in the console or log.

7. Evaluate the Model
After training, evaluate the model on validation data to check performance metrics (e.g., F1 score, accuracy).
Review metrics to assess model accuracy and identify areas for potential improvement.

8. Make Predictions
Use the model to generate predictions on the test dataset.
Save the predictions to a file (e.g., predictions.csv) for analysis or potential submission.

📊 Additional Information
GPU Support: This project benefits from GPU acceleration. Use Google Colab, Kaggle, or a local machine with GPU support for faster training.
Configurable Parameters: Experiment with parameters like num_train_epochs, learning_rate, and max_length to optimize performance based on your dataset and resources.
