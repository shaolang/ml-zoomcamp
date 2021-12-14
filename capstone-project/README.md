# Machine Learning Zoomcamp Capstone Project
This project tackles the classification of facial expressions. The
dataset is downloaded from [Kaggle FER12][data-source]. The main task
is to classify the facial expression for any given 48x48 grayscale image.

This project contains the following artifacts:

* `/train`: contains the images for training
* `/test`: contains the images for testing
* `requirements.txt`: states the Python packages necessary for running
  this project.
* `notebook.ipynb`: contains Exploratory Data Analysis (EDA), feature
  transformation, multiple model training, model selection, and model
  hyperparameter-tuning.
* `train.py`: implements the script to train and export the model as `model.h5`.
* `predict.py`: runs a web service for the facial expression classification.

# Running the Project

To run the project:

* Download and install [Python][python] version 3.8 and above.
* Download the artifacts from this repository.
* Navigate to the directory that contains the artifacts downloaded and
  run `pip install -r requirements.txt` to install the necessary Python
  packages.
* Run `jupyter notebook` to start Jupyter Notebook web server.
* Open `notebook.ipynb` shown in the web browser.

# Create TensorFlow Model

Run `python train.py` to create the TensorFlow model. Once it's completed,
it'll create the file `model.h5` in the same directory as `train.py`.

# Testing the Web Service

To test the web service, create the model file first by following the
instructions in the Create TensorFlow model above. Then run `python predict.py`.
It'll start the web service, listening at 0.0.0.0:8080.

Then run the following command to send the data to interact with the service:

```
curl -X POST -H "Content-Type: text/plain" --data "https://cdn-media.threadless.com/cache/profile_photos/B7/1529098_1_profile_picture_48x48.jpg" http://localhost:8080/
```

[data-source]: https://www.kaggle.com/msambare/fer2013
[python]: https://www.python.org
