# NLP-DA-PROJECT
THIS IS THE REPOSITORY FOR THE NLP DA PROJECT FOR THE TITLE NATURAL DISASTER TWEET ANALYSIS
This project uses DistilBERT for classifying tweets as disaster-related or non-disaster-related. Follow the steps below to set up the environment and run the project.

Requirements
Python: Version 3.6 or higher

Libraries: Install required libraries by running:

bash
Copy code
pip install pandas scikit-learn transformers torch
Dataset: Ensure you have train.csv and test.csv files for training and testing. Place these files in your working directory or update the code with the correct file paths.

Steps to Execute
Set Up Environment:

Install the required libraries as mentioned above.
Confirm that train.csv and test.csv are available in the directory you’re working in.
Load and Preprocess the Data:

Start by loading the data using the paths provided in the code.
Preprocess the text data (e.g., convert to lowercase, remove special characters) for uniformity and improved model performance.
Tokenize and Encode the Text:

Use the DistilBERT tokenizer to convert text data into a format suitable for model input.
Set max_length for truncation/padding, ensuring uniform input sizes across all text samples.
Initialize and Configure the Model:

Load the DistilBERT model pre-trained for sequence classification.
Configure training parameters such as batch size, number of epochs, and learning rate.
Train the Model:

Run the training code to fine-tune the DistilBERT model on the dataset.
Training progress, including loss values, will be displayed in the console or log.
Evaluate the Model:

After training, evaluate the model on validation data to check its performance metrics (e.g., F1 score, accuracy).
Review metrics to understand model accuracy and identify potential improvements.
Make Predictions:

Use the model to make predictions on the test dataset.
Save the predictions to a file (e.g., predictions.csv) for analysis or submission if required.
Additional Notes
GPU Support: This project benefits from GPU acceleration. It is recommended to use Google Colab, Kaggle, or a local machine with GPU support for faster training.
Adjustable Parameters: Experiment with parameters such as num_train_epochs, learning_rate, and max_length based on your dataset and available resources.
