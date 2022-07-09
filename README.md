# lime-detector
Tool for detecting leakage in training process of ML models.


## Objective
Given a scenario which makes use of a machine learning model, often a black box i.e., that which cannot
be interpreted by the user, the objective of this project is to decipher the characteristics of the model
using Local Interpretable Model-Agnostic Explanations (LIME), and then detect the presence of data
leakage on the basis of parameters deduced from the explanations.

## Overview
In this project, we have evaluated the Random Forest Classifier on different subsets of the “20
newsgroups dataset”. This dataset contains 18000 posts from discussion forums which can be categorized
into 20 topics. Since the dataset contains “text” form of data, we convert them into TF-IDF vectors so
that they become a compatible input to the classifier model. We train the model with the parameter -
maximum allowed number of trees as 500.
We use LIME (local interpretable model-agnostic explanations) to find the top 6 informative features.
LIME finds informative features by using a surrogate model to estimate the actual model. The surrogate
model is a linear regression model, and the features with higher weights correspond to more informative
features. We can verify this by removing the top 2 informative features, and using the original classifier to
predict again. The predictions are flipped, when certain features are ignored, thus indicating their
importance. However LIME is not aware that the classifier worked upon the TF-IDF vectorized
representation of the text, thus we create a pipeline from sklearn for vectorization. The LIME explainer
instance uses the class probability predictor function from the pipeline.
We make observations on the informative features.
We then consider a variation of the “20-newsgroups-dataset” which does not contain certain sections of
texts. We then apply TF-IDF vectorization and train on the new features and repeat the procedure.
We make observations on these new informative features.

