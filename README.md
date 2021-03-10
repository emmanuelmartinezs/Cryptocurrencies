# Cryptocurrencies

## Overview of Project
You and Martha have done your research. You understand what unsupervised learning is used for, how to process data, how to cluster, how to reduce your dimensions, and how to reduce the principal components using PCA. It’s time to put all these skills to use by creating an analysis for your clients who are preparing to get into the cryptocurrency market.

Martha is a senior manager for the Advisory Services Team at Accountability Accounting, one of your most important clients. Accountability Accounting, a prominent investment bank, is interested in offering a new cryptocurrency investment portfolio for its customers. The company, however, is lost in the vast universe of cryptocurrencies. So, they’ve asked you to create a report that includes what cryptocurrencies are on the trading market and how they could be grouped to create a classification system for this new investment.

The data Martha will be working with is not ideal, so it will need to be processed to fit the machine learning models. Since there is no known output for what Martha is looking for, she has decided to use unsupervised learning. To group the cryptocurrencies, Martha decided on a clustering algorithm. She’ll use data visualizations to share her findings with the board.

## Deliverables:
This new assignment consists of three technical analysis deliverables and a written report.

1. ***Deliverable 1:*** Preprocessing the Data for PCA
2. ***Deliverable 2:*** Reducing Data Dimensions Using PCA
3. ***Deliverable 3:*** Clustering Cryptocurrencies Using K-means
4. ***Deliverable 4:*** Visualizing Cryptocurrencies Results


## Deliverables:
This new assignment consists of three technical analysis deliverables and a proposal for further statistical study:

* Data Source: `crypto_data.csv`
* Data Tools:  `crypto_clustering_starter_code.ipynb`.
* Software: `Python 3.9`, `Visual Studio Code 1.50.0`, `Anaconda 4.8.5`, `Jupyter Notebook 6.1.4` and `Pandas`


## Resources and Before Start Notes:

![logo](https://github.com/emmanuelmartinezs/Cryptocurrencies/blob/main/Resources/Images/Header.jpg?raw=true)


### Unsupervised Machine Learning and Cryptocurrencies
#### Using Unsupervised Learning to Discover Unknown Patterns

##### Challenges of Unsupervised Learning

**IMPORTANT**
> Unsupervised learning isn't the solution for every data analytic challenge. Just because supervised learning might not work for one situation doesn't mean unsupervised learning will work instead. Understanding the data and what can be done with it is an important first step before choosing an algorithm.

Recall that unsupervised learning does not take in any pairing of input and outcomes from the data—it only looks at the data as a whole. This can cause some challenges when running the algorithm. Since we won't know the outcome it's predicting, we might not know that the result is correct.

This can lead to issues where we're trying to decide if the model has provided any helpful information that we can use to make decisions in the real world. For example, our store owner might run a model that ends up grouping the type of people by how much they're buying. This could be useful in some contexts—for example, knowing who the top spenders are—but it might not help the store owner better organize the store for maximum purchases per person, or understand the differences in product preferences between top purchasers.

The only way to determine what an unsupervised algorithm did with the data is to go through it manually or create visualizations. Since there will be a manual aspect, unsupervised learning is great for when you want to explore the data. Sometimes you'll use the information provided to you by the unsupervised algorithm to transition to a more targeted, supervised model.

As with supervised learning, data should be preprocessed into a correct format with only numerical values, null value determination, and so forth. The only difference is unsupervised learning doesn't have a target variable—it only has input features that will be used to find patterns in the data. It's important to carefully select features that could help to find those patterns or create groups.

The next section will cover data preprocessing and data munging, and provide a refresher on Pandas and data cleaning. First, you'll need to install the necessary libraries for practice.

##### Install Your Tools
If you already have some libraries installed from previous modules, you may skip those parts of the installation instructions.

##### Scikit-learn
To install the Scikit-learn library, follow these steps:

1. Open your terminal and activate your PythonData environment.

2. Run the following command:

`conda install scikit-learn`

3. After installation, you're all set.

##### Plotly
To install the Python Plotly library, follow these steps:

1. Open your terminal and activate your PythonData environment.

2. Run the following command:

`conda install plotly`

3. After installation, you're all set.

##### hvPlot
To install the hvPlot visualization library, follow these steps:

1. Open your terminal and activate your PythonData environment.

2. Run the following command:

`conda install -c pyviz hvplot`

3. After installation, you're all set.

##### Steps for Preparing Data
**After** digging into unsupervised learning a bit, you realize that your first step in convincing Accountability Accountants to invest in cryptocurrency is to preprocess the data.

You and Martha open up the dataset to get started preprocessing it. Together, you will want to manage unnecessary columns, rows with null values, and mixed data types before turning your algorithm loose.

##### Data Selection
Before moving data to our unsupervised algorithms, complete the following steps for preparing data:

1. Data selection
2. Data processing
3. Data transformation

Data selection entails making good choices about which data will be used. Consider what data is available, what data is missing, and what data can be removed. For example, say we have a dataset on city weather that consists of temperature, population, latitude and longitude, date, snowfall, and income. After looking through the columns, we can readily see that population and income data don't affect weather. We might also notice some rows are missing temperature data. In the data selection process, we would remove the population and income columns as well as any rows that don't record temperatures.

##### Data Processing
Data processing involves organizing the data by formatting, cleaning, and sampling it. In our dataset on city weather, if the date column has two different formats—mm-dd-yyyy (e.g., 01-23-1980) and month-data-year (e.g., jan-23-1980)—we would convert all dates to the same format.

##### Data Transformation
Data transformation entails transforming our data into a simpler format for storage and future use, such as a CSV, spreadsheet, or database file. Once our weather data is cleaned and processed, we would export the final version of the data as a CSV file for future analysis.


Now we can begin our machine learning journey.

> Let's move on!

# Deliverable 1:  
## Use Resampling Models to Predict Credit Risk 
### Deliverable Requirements:

Using your knowledge of the `imbalanced-learn` and `scikit-learn` libraries, you’ll evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. First, you’ll use the oversampling `RandomOverSampler` and `SMOTE` algorithms, and then you’ll use the undersampling `ClusterCentroids` algorithm. Using these algorithms, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

> To Deliver. 

**Follow the instructions below:**

Follow the instructions below and use the `credit_risk_resampling_starter_code.ipynb` file to complete Deliverable 1.

Open the `credit_risk_resampling_starter_code.ipynb` file, rename it `credit_risk_resampling.ipynb`, and save it to your **Credit_Risk_Analysis** folder.

Using the information we’ve provided in the starter code, create your training and target variables by completing the following steps:

* Create the training variables by converting the string values into numerical ones using the `get_dummies()` method.
* Create the target variables.
* Check the balance of the target variables.

Next, begin resampling the training data. First, use the oversampling `RandomOverSampler` and `SMOTE` algorithms to resample the data, then use the undersampling `ClusterCentroids` algorithm to resample the data. For each resampling algorithm, do the following:

* Use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
* Calculate the accuracy score of the model.
* Generate a confusion matrix.
* Print out the imbalanced classification report.

Save your `credit_risk_resampling.ipynb` file to your **Credit_Risk_Analysis** folder.


#### Deliverable 1 Requirements

For all three algorithms, the following have been completed:
- An accuracy score for the model is calculated
- A confusion matrix has been generated
- An imbalanced classification report has been generated 


# Deliverable 2:  
## Use the SMOTEENN algorithm to Predict Credit Risk 
### Deliverable Requirements:

Using your knowledge of the `imbalanced-learn` and `scikit-learn` libraries, you’ll use a combinatorial approach of over and undersampling with the `SMOTEENN` algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from Deliverable 1. Using the `SMOTEENN` algorithm, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

> To Deliver. 

**Follow the instructions below:**

Follow the instructions below and use the information in the `credit_risk_resampling_starter_code.ipynb` file to complete Deliverable 2.

1. Continue using your `credit_risk_resampling.ipynb` file where you have already created your training and target variables.
2. Using the information we have provided in the starter code, resample the training data using the `SMOTEENN` algorithm.
3. After the data is resampled, use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
4. Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.

Save your `credit_risk_resampling.ipynb` file to your Credit_Risk_Analysis folder.


#### Deliverable 2 Requirements

The combinatorial SMOTEENN algorithm does the following:
- An accuracy score for the model is calculated
- A confusion matrix has been generated
- An imbalanced classification report has been generated  


# Deliverable 3:  
## Use Ensemble Classifiers to Predict Credit Risk 
### Deliverable Requirements:

Using your knowledge of the `imblearn.ensemble` library, you’ll train and compare two different ensemble classifiers, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, to predict credit risk and evaluate each model. Using both algorithms, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

> To Deliver. 

**Follow the instructions below:**

Follow the instructions below and use the information in the `credit_risk_resampling_starter_code.ipynb` file to complete Deliverable 3.

1. Open the `credit_risk_ensemble_starter_code.ipynb` file, rename it `credit_risk_ensemble.ipynb`, and save it to your **Credit_Risk_Analysis** folder.
2. Using the information we have provided in the starter code, create your training and target variables by completing the following:
    - Create the training variables by converting the string values into numerical ones using the `get_dummies()` method.
    - Create the target variables.
    - Check the balance of the target variables.
3. Resample the training data using the `BalancedRandomForestClassifier` algorithm with 100 estimators.
    - Consult the following [Random Forest documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html) for an example.
4. After the data is resampled, use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
5. Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
6. Print the feature importance sorted in descending order (from most to least important feature), along with the feature score.
7. Next, resample the training data using the `EasyEnsembleClassifier` algorithm with 100 estimators.
    - Consult the following [Easy Ensemble documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html) for an example.
8. After the data is resampled, use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
9. Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.

Save your `credit_risk_ensemble.ipynb` file to your **Credit_Risk_Analysis** folder.


#### Deliverable 3 Requirements

The `BalancedRandomForestClassifier` algorithm does the following:
- An accuracy score for the model is calculated 
- A confusion matrix has been generated 
- An imbalanced classification report has been generated 
- The features are sorted in descending order by feature importance

The `EasyEnsembleClassifier` algorithm does the following:
- An accuracy score of the model is calculated 
- A confusion matrix has been generated 
- An imbalanced classification report has been generated  


### DELIVERABLE RESULTS:
## Deliverable 1
### Preprocessing the Data for PCA
The following was done to preprocess the data:

1. Remove cryptocurrencies that are not being traded.
2. Keep all the cryptocurrencies that have a working algorithm; all have working algorithm
3. The ```IsTrading``` column is unnecesssary; it was dropped.
4. Rows that have at least one null value were removed.
5. Filter dataset to reflect only coins that have been mined.
6. Create a new DataFrame of cryptocurrency names ```CoinNames```, and use the index from the previous dataset as the index for this new DataFrame. DataFrame ```cc_names_df```   
![d1](https://github.com/emmanuelmartinezs/Cryptocurrencies/blob/main/Resources/Images/s1.png?raw=true)

7. Remove the ```CoinName``` column from the DataFrame since it's not necessary for the clustering algorithm. Dataframe: ```crypto_df```    
![d1](https://github.com/emmanuelmartinezs/Cryptocurrencies/blob/main/Resources/Images/s2.png)

Finally, using the ```get_dummies()``` method for columns ```Algorithm``` and ```ProofType``` and the StandardScaler ```fit_transform()``` function to standardize the features.  The resulting DataFrame has 98 columns, which cannot be shown fully here, but here is an excerpt:
![d1](https://github.com/emmanuelmartinezs/Cryptocurrencies/blob/main/Resources/Images/s3.png)


## Deliverable 2
### Reducing Data Dimensions Using PCA
The next steps involve, applying PCA to reduce the dimensions to 3 principal components.

The resulting DataFrame, ```pcs_df``` now includes columns ```PC 1```, ```PC 2```, and ```PC 3```, and uses the index of the ```crypto_df``` DataFrame as the index.  Please see below:  
![d1](https://github.com/emmanuelmartinezs/Cryptocurrencies/blob/main/Resources/Images/s4.png)


## Deliverable 3 
### Clustering Cryptocurrencies Using K-means
To determine an appropriate number of clusters for the dataset ```pcs_df```, start with plotting an elbow curve with hvplot to find the value for K.  
 
![d1](https://github.com/emmanuelmartinezs/Cryptocurrencies/blob/main/Resources/Images/s5.png)


Based on the above curve, it looks like **4 clusters** is the place to start!


Using **K=4** and applying the K-means algorithm, provides the following predictions:
![d1](https://github.com/emmanuelmartinezs/Cryptocurrencies/blob/main/Resources/Images/s6.png)

A new DataFrame, ```clustered_df``` is created by:

*  Concatenating the ```crypto_df``` and ```pcs_df``` DataFrames with the same index as the crypto_df DataFrame. 
*  Adding the ```CoinNames``` column from the ```cc_names_df``` dataset created earlier.  The resulting DataFrame:
![d1](https://github.com/emmanuelmartinezs/Cryptocurrencies/blob/main/Resources/Images/s7.png)

## Deliverable 4
### Visualizing Cryptocurrencies Results

With this new DataFrame, ```clustered_df``` , we start with a 3D Scatter plot using the Plotly Express **scatter_3d()** function to visualize the 4 Classes.  Each data point shows the ```CoinName``` and ```Algorithm``` data when hovering over.
![d1](https://github.com/emmanuelmartinezs/Cryptocurrencies/blob/main/Resources/Images/s8.png)


Next, a table is created featuring the tradable cryptocurrencies using the **hvplot.table()** function.
![d1](https://github.com/emmanuelmartinezs/Cryptocurrencies/blob/main/Resources/Images/s9.png)

The total number of tradable cryptocurrencies in the clustered_df DataFrame: 
```The Total Number of tradable cryptocurrencies is 532.```

For the last visualization, a new dataset is created using the  **MinMaxScaler().fit_transform** method to scale the ```TotalCoinSupply``` and ```TotalCoinsMined``` columns (range of zero and one) from the ```clustered_df```, adding the ```CoinName``` from ```cc_names_df``` and the ```Class``` column from ```clustered_df```.  The resulting DataFrame:
![d1](https://github.com/emmanuelmartinezs/Cryptocurrencies/blob/main/Resources/Images/s10.png)


Finally, a 2D **hvplot scatter plot** with x="TotalCoinsMined_scaled", y="TotalCoinSupply_scaled", and by="Class" with the ```CoinName``` displayed when you hover over the the data point.
![d1](https://github.com/emmanuelmartinezs/Cryptocurrencies/blob/main/Resources/Images/s11.png))

This plot highlights the fact that Class 2 has only 1 cryptocurrency, **BitTorrent**, quite the outlier!

So, what if we increase to 5 clusters, another acceptable option to test?  What does that look like?

Well, not terribly different...Class 2 is still only 1 coin, Class 3 is basically the same, and Classes 0 and 1 broke out into 3 Classes: 0, 1, 4.

![d1](https://github.com/emmanuelmartinezs/Cryptocurrencies/blob/main/Resources/Images/s12.png)

![d1](https://github.com/emmanuelmartinezs/Cryptocurrencies/blob/main/Resources/Images/s13.png)



##### Cryptocurrencies Completed by Emmanuel Martinez
