# IEEE-CIS-Fraud-Detection
Kaggle Fraud Transaction Detection competition

---

![alt text](https://github.com/emisaycheese/IEEE-CIS-Fraud-Detection/blob/master/GIF/head.png)


This is a **private** repo for winning the Kaggle Fraud Transaction Detection competition](https://www.kaggle.com/c/ieee-fraud-detection/) at 217 out of 6300 teams (3%) !


Screenshot place holder.

Account Id


In this repo, we organize our Kaggle core codes into 2 parts: feature engineering (data pipline) and model training.

## Feature Engineering

We did the minimal data cleaning work for this project, as LightGBM model can automatically handle quite a lot, e.g., `missing values`, `outliers`. The current optimal way would be the default choice of the model.

We have different methods for **continuous** and **categorical** data correspondingly.

### Continuous Data. 
We have done a variety of mathematical transformation based on the data distributions.

### Categorical Data

We spent the majority of our time on this.

* Risk Mapping and [feature embeding](https://arxiv.org/abs/1604.06737). We understood there was a risk of overfitting but we took extra care to deal with this potential issue.

* Label Transformation. For some categorical features with unknown meanings and less than their unique values was less than a threshold, as [this](https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931) suggested, we used the label transformation function directly from `sklearn`. We are aware of the bad implications of this but maybe for tree model, the adversarial impacts are trivial or not significant.

Domain Knowledge: this technique incoporates the common knowledge of our daily life, like a transaction made online is more dangerous than local store, or the hours of transaction made and so on. 

Interaction Terms: we used the interactions among features depending on the sole critieror that after the interaction, we could see the boundary of two classes more clearly

The most surprising features we found being useful was countings for the nulls of each row, which makes sense and simple, `"KISS"` as our law.

We used the H2O autoML tool to generate insights/reduce dimensions from some not so important features, e.g., V-type features, features with too many missing values, invarient features.


### Data Pipline 

It is coming soon!


## Model Training



### Model

For lightGBM, we had to choose a booster as the core model, we chose `DART` for the following reasons:

Graph place holder (from qishi ml journal club teng)

* Combine the advantages of random forest and gradient boosting tree models
* Our first few trees are not so important thus reducing the effects of overfitting.
* But really slow to train. 

(Not ready yet, T will update this)

### Model Segmentation


We have two types of modeling approaches, one is to separate the data via the most important feature - transaction product type and the other one is not. The reason we took these approaches are that we found the product type had dramatic impacts on other features' distributions, missing percentages. 

Please find more visualzation evidence in [notebook]. ( T will update this notebook)

### Model Training


For both approaches, we used the CV modules to address the overfiting issues.

* **Cross-validation module** - dealing with overfitting issues
* **Time-series module** - dealing with the time problem. Remember, the train and test dataset are sequential, which test dataset comes later in time

### Hyper-Parameter Optimization
* **Grid Search module** 
* **Random Search module** 
* **Bayes optimization module** - dealing with high dimensional hyper-parameter space searching

#### Why do we use different methods to search parameters? 

Because the search space for LightGBM is so huge and we try to use different methods to find diversified local optimal solutions. (T maybe updates this possibly)


### Model stacking
We built the models from a diversified background to reduce the results correlation and thus achieve a better performance in model stacking.

* **Model stacking module** - dealing with overfitting issues


In addition, after an in-depth research into the computational mechanisms of ROC AUC scores, we found a clever way to play with it and increase the performance. 

---
## What are the differentiating factors?

**Techonology** & **Experiences**

* We utilize the cutting-edge machine learning modeling framework and tracking system to greatly increase our collaborating efficiency. Featuretools and H2O that let us to focus on the most important features. In addition, we also used `mlflow` to track our ML experiments scientifically and **control** the submissions so that we do not repeat the experiments and avoid overfiting. 
* We are a diversified team of data scientist veterans from consulting, banking and hedgefund industry.



## Work for us?
Write separate functions in py file as `utils` and use notebook to call the necessary

* Model Stack module V 3
* Model Segmentation module V 2
* Parameter optimizations: 3 module E 1
* CV and times series E 1
* Data Visual T&E 1
* Show the fancy techs (gradually) T 4
