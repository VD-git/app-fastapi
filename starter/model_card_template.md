# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The acronym LGBMClassifier denotes Light Gradient Boosting Machine Classifier, employing decision tree algorithms for tasks like ranking and classification. It leverages Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) techniques, ensuring precision in handling substantial datasets. This unique approach enhances speed, while simultaneously reducing memory usage. More info can also be found in the link: https://lightgbm.readthedocs.io/en/latest/index.html

* lightgbm == 4.2.0

## Intended Use

The model is intend to be utilized in the following problem: The task is to predict whether a person makes over 50K a year.

## Training Data

Extraction was done by Barry Becker from the 1994 Census database.  A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

The data is public available in the following link: https://archive.ics.uci.edu/dataset/20/census+income

## Evaluation Data

In order to make the evaluation the data set was split in 80%/20% (train/test). Regarding the hyperparameter optimazation, with the train data was performed a GridSearchCV evaluation, where a cross-validation is done in order to have better use of the data avoiding any data leakage.

The pre-processing is composed of two main steps: 1°) OneHotEncoder and 2°) LabelBinarizer, futher than that the first step is consisted of an ETL to clean the string format (mainly applying a strip function)

## Metrics
* precision = 0.77;
* recall = 0.65;
* fbeta = 0.71.

For a more detail approach, in attachment in the folder "../starter/model/slice_output.txt" can be found a table with slice metrics for each of the categorical feature.

## Ethical Considerations

Points to be noted and take into consideration, is that race and sex are definied as features. Due to that, those are variables to pay attention to avoid bias.

## Caveats and Recommendations

* Since the data is a bit old, from 1994, and right now is 2024, it is interesting to recalculate the salary, maybe applying the inflation tax during this period, at least to recalculate the threshold, since the salary output is binary;
* One possibility is to treat the missing values, the method chosen here was to create a new variable for them.