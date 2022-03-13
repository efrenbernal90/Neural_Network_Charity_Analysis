# Neural_Network_Charity_Analysis

## Overview

Using the features in the provided dataset, create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, there are more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

### Resources

#### Data

- charity_data.csv

#### Software Tools

- [Jupyter Notebook v6.4.6](https://jupyter-notebook.readthedocs.io/en/stable/index.html)
- [SciKit-Learn Library v 1.0.2](https://scikit-learn.org/stable/getting_started.html)
  - [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html?highlight=onehotencoder#sklearn.preprocessing.OneHotEncoder)
  - [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Tensor Flow](https://www.tensorflow.org/guide/)
  - [Sequential Model](https://www.tensorflow.org/guide/keras/sequential_model)
  - [Broweser 'playground' Model](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=spiral&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=8&seed=0.14370&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=true&xSquared=true&ySquared=true&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

## Results

### ***Data Preprocessing***

Once the `charity_data.csv` file was read into a dataframe, the data was split into features and target variables to train and test the `SequentialModel`:

  - The **IS_SUCCESSFUL** column was used as a target variable, since it measures successful use of money, and therefore an ideal candidate for the model.
  - The **EIN** and **NAME** columns are unique categorical identifiers dropped for this model.
  - The remaining columns were used as features for the model and preprocessed 
