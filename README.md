# Automatic Music Genre Classification
This group project aims to classify a short music sample between 10 genres (rock, jazz, classic, blues…) using machine learning techniques. The first part is a state of the art in Automatic Music Genre Classification. The second part is our own implementation, using several classic classifiers (Logistic Reg., Decision Trees, SVMs…) with hand-crafted features.

## Requirements
You can install the requirements using `pip install -r requirements.txt`

## Data
The dataset used in our work is the GTZAN genre collection dataset, created by George Tzanetakis, available for download at [this page](http://marsyas.info/downloads/datasets.html). (Approximately 1.2GB) However, for the sake of time, we provided the [CSV file](https://github.com/luca-serra/automatic-music-genre-classification/blob/master/Data/extracted_features.csv) (obtained thanks to `feature_extraction.py` run on the dataset) containing the features we used for this project.
## Usage
 Run the python file `main_classification.py` or `main_classification_with_vote.py`.
 * This will run cross-validation across the training set, and print the accuracy score.
 * Confusion matrix over the whole cross-validation will be displayed.

## Report
The report of our work and of our researches upon this topic can be found [here](https://github.com/luca-serra/automatic-music-genre-classification/blob/master/Final_Project.pdf). Two other folders complete this report section ([resources](https://github.com/luca-serra/automatic-music-genre-classification/tree/master/resources) and [summaries](https://github.com/luca-serra/automatic-music-genre-classification/tree/master/summaries) we made of these documents).
