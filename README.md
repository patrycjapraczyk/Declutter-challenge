#### This project aims to explore different machine learning approaches to design a method for classifying comments into useful and clutter.

It is based on the following competition:

https://dysdoc.github.io/docgen2/index.html

https://github.com/dysdoc/declutter

### Requirements for running the project

In order to run software created in this project, the following must be installed on the system: 

* Python 3.8+,
* Pip, 
* Git, 
* Anaconda 1.7.1+, 
* Jupyterlab (for Jupyter Notebooks),
* A text editor or an IDE for Python, PyCharm Professional Edition with Scientific Mode is recommended, as it has many libraries and tools pre-installed. 
 
The project relies on many Python libraries. They can be installed by running the following command in the project root folder: 

`pip install -r requirements.txt`

### GitHub repository 

The project is available at a public GitHub repository with the following link:  

https://github.com/patrycjapraczyk/Declutter-challenge 

### Running instructions

The main module with a few sample k-fold validation with ADASYN method and BOW applied can be run by executing in the root folder:

`python3 -m main`

Running this file generates a few warnings- SettingWithCopyWarning and ConvergenceWarning. They do not affect the correctness of the program and can be ignored.
The resulting scores from k-fold validation procedure will be printed in the console. 
Please note that k-fold procedure will take a few minutes.
In order to change the test options or run test against Python dataset, please follow instructions found in **main.py** file.

Jupyter notebook files can be run by executing the following commands assuming that the terminal is open in the root directory of the project:

`cd notebooks`
`jupyter notebook`

This opens a Jupyter notebook view in the browser at http://localhost:8888/

### Project structure and file overview 

* **const/** contains constants used across the project
    * **programming_keywords.py** contains two arrays, one with Python programming language keywords such as ‘self’ or ‘def’, and another with Java keywords such as ‘abstract’ or ‘class’
    * **java_tags.py** contains an array with Java tags such as ‘@author’
    
* **data/** contains csv and text files with datasets
    * **code_javadoc.txt** contains a file with longer code blocks relevant to javadoc comments 
    * **code_data.csv** contains code taken by extracting the next valid line of comments of all types 
    * **functional_types.csv** contains data comment data along with functional types assigned 
    * **python_data.csv contains** a dataset with Python comments, their label, relevant code and functional_type, 
    * **train_set_0520.csv** original dataset provided by DeClutter challenge organisers, contains data extracted from Java projects with the following columns: ID, type, path_to_file, begin_line, link_to_comment, comment, non-information 
    
* **helpers/** contain files with auxiliary functions used throughout the project 
    * **class_imbalance_sampling.py** contains implementations of data sampling methods for tackling class imbalance problem, a chosen data imbalance sampling algorithm can be applied on data by calling ‘ImbalanceSampling.get_sampled_data(algo, x_train, y_train)’ with ‘algo’ parameter being a string of values ‘SMOTE’, ‘ADASYN’, ‘RANDOM_OVERSAMPLE’, ‘RANDOM_UNDERSAMPLE’, ‘SMOTEEN’, 
    * **cochrans_exec.py** executes Cochran’s Q test on data from comparison of different models with different text vectorising methods using k-fold validation, 
    * **data_loader.py** loads data from csv or text files needed to train the models or do data analysis, the data is loaded into Pandas DataFrame format,
    * **data_preprocessing.py** - contains functions for data sanitation and regularisation 
    * **ensemble_weighing.py** contains functions for grid search for finding the best weights between text and no-text classifiers when putting them together to create an ensemble model 
    * **feature_helper.py** contains helper functions that are used when processing features that are input into models such as get_stop_words_num 
    * **github_scraper.py** contains functions for extracting code relevant to comments from GitHub and saving them into a csv file 
    * **mcnemar_test.py** executes McNemar test to check the variance between results from k-fold validation of different models 
    * **parameters_tuning.py** contains functions for tuning models’ hyperparameters based on grid search 
    * **score_metrics.py** contains functions for calculating models’ different score measures such as precision, accuracy, recall, balanced accuracy, f1 etc 
    * **text_similarity.py** calculates various text similarity measures such as Jaccard Index and Cosine similarity, a similarity score between two strings can be retrieved by calling “get_similarity_score(s1: str, s2: str, type: str)”, where type is ‘JACC’, ‘COSINE’ or ‘COSINE_TFIDF’ 
    * **textual_analysis.py** contains functions that aid in text analysis such as ‘create_word_cloud’ 

* **models/** contains files with different model implementations, model execution and model factory 
    * **abstract_model.py** contains a model interface that is inherited by all classes with model implementations 
    * **ada_boost.py** contains an implementation of Ada Boost classifier 
    * **decision_tree_classifier.py** contains an implementation of Decision Tree classifier 
    * **dummy.py** contains an implementation of Dummy classifier 
    * **ensemble.py** contains an implementation of ensemble of two models- one with comment text data and another with other features 
    * **gaussian_process.py** contains an implementation of Gaussian Process classifier 
    * **gradient_boosting.py** contains an implementation of Gradient Boosting classifier 
    * **K_neighbors_classifier.py** contains an implementation of K Neighbors classifier 
    * **logistic_regression.py** contains an implementation of Logistic Regression classifier
    * **MLP.py** contains an implementation of MLP classifier
    * **model_exec.py** contains all functions that implement different stages needed to execute models- preprocessing, k fold validation, creating and combining different features, splitting data into test and train, testing the dataset with Python data, comment vectorising 
    * **model_factory.py** - based on Factory Method Design Pattern, returns a classifier with a specified type and returns a list of all classifiers 
    * **naive_bayes.py** contains an implementation of Naïve Bayes classifier 
    * **quadratic_discriminant_analysis.py** contains an implementation of Quadratic Discriminant Analysis classifier 
    * **random_forest.py** contains an implementation of Random Forest classifier 
    * **SVM.py** contains an implementation of SVM classifier 
    
* **notebook/** contains Jupyter notebook files 
    * **dataAnalysis/** contains notebooks with illustrations about data that were used to gain better knowledge about the dataset and investigate features distinguishing informative and non-informative comments 
    * **feature_selection.ipynb** contains results from ANOVA f-test executed to select relevant features 
    * **randomForestAlgorithm.ipynb** contains a graph illustrating feature importance extracted from Random Forest algorithm 

* **test/** contains unit tests for “data_preprocessing.py”, “feature_helper.py”, “github_scraper.py”, “textual_analysis.py” 

* **text_representation/** contains files with different implementations of text vectorising methods and text vectorizer factory  
    * **abstract_text_representation.py** contains a text vectorizer interface that is inherited by all classes with text vectorizers 
    * **bag_of_ngrams.py** contains an implementation of Bag of NGrams text vectorising method 
    * **bag_of_words.py** contains an implementation of Bag of Words text vectorising method
    * **text_representation_factory.py** based on Factory Method Design Pattern, returns a text vectorising method of a specified type  
    * **tfidf.py** contains an implementation of TFIDF text vectorising method 
    * **word2vec.py** contains an implementation of Word2vec text vectorising method 

 