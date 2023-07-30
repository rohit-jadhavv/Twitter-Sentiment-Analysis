
# Twitter Sentiment Analysis

Twitter Sentiment Analysis is a project that automatically analyzes tweets to determine whether they express positive or negative sentiments. By using Natural Language Processing (NLP) techniques and machine learning algorithms, the project provides insights into public opinions and emotions on various topics, products, or events shared on Twitter. It helps businesses, researchers, and individuals better understand the sentiment of social media users and make data-driven decisions.


## Key Features

#### Data Collection: 
The dataset used for training and testing the sentiment analysis model is obtained from Kaggle. It includes labeled tweets with binary sentiment labels (positive or negative).

#### Text Preprocessing: 
Text preprocessing is a critical step in natural language processing (NLP) tasks, including Twitter sentiment analysis. It involves cleaning and transforming raw text data into a format that is suitable for analysis and modeling.Tweets undergo thorough text preprocessing, including tokenization, removing stop words, stemming or lemmatization, and handling special characters and emojis. 

#### Data Visualization using Word Cloud: 
The tweets in the dataset are segregated into positive and negative sentiment and then we use word cloud for representing positive and negative sentiment separately.


## Dataset
The Dataset used for this project is obtained from Kaggle
Link: https://www.kaggle.com/datasets/kazanova/sentiment140


## Text Representation
The Text Represenation Technique used in this model is TfidfVectorizer,TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer is a popular technique used in Natural Language Processing (NLP) to transform text data into numerical feature vectors. It is a critical step in many text-based machine learning tasks, such as text classification, clustering, and information retrieval.


## Model Information
The project uses following model to perform Sentiment Classification:
1.Logistic Regression:  Logistic Regression is a linear classification algorithm that models the probability of a binary outcome (positive or negative sentiment) based on input features. It is commonly used in text classification tasks due to its simplicity and efficiency.

2.Naive Bayes (MultinomialNB): Naive Bayes MultinomialNB is a probabilistic classification algorithm based on Bayes' theorem with an assumption of independence between the features. It is well-suited for text classification tasks and has been widely used in NLP applications.

In this Twitter Sentiment Analysis project, we have implemented and evaluated two different models for sentiment classification. The purpose of using multiple models is to compare their performance and identify the one that provides more accurate results for the sentiment analysis task.

## Notes:
I also provided the twitter pipeline where I created a pipeline for Data Preprocessing, Vectorization and Model Training

