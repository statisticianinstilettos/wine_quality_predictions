# email_project

Gradient Boosting Machines (GBMs)

The purpose of this project is to provide an example/tutorial using Gradient Boosting Machines (GBMs). I created a spam email detection model using real datasets obtained from KAggle, and a Gradient Boosting Trees Model. The model is trained to predict "Spam" (class 1) versus "Not Spam" (class 0).

Note: The purpose of this project is to apply Gradient Boosting Machines. The training data is not optimal for training a true spam detection system, because it is not a representative sample of emails. This model is extreemly accurate because the training data was from two different sources, and thus the model had little trouble distinguishing between the two types of emails. It was difficult to find example email data that was a representative sample of true emails.I would not expect this model to generalize well to emails in my inbox right, now however, this model could be retrained on a better dataset, and would generalize well.  

<img src="https://media.giphy.com/media/YAnpMSHcurJVS/giphy.gif" width=400>

## Modeling approach
The data used to train this model contains conversational emails from the Enron dataset tagged as "Not Spam". https://www.kaggle.com/wcukierski/enron-email-dataset
An example is shown here:
example of class 0 email body


The data used to train this model also contains spam emails from the fradulent dataset tagged as "Spam". https://www.kaggle.com/rtatman/fraudulent-email-corpus
An example is shown here:
example of class 1 email body

* Data is cleaned and processed using typical methods to clean text data (see notebook)
* Feature engineering using TF-IDF and SVD to compress email body text into 25 features
* GBT model is used to fit the model
* AUC, Confusion Matrix, and Accuracy are used to evaluate the models performance

## Boosting Machines
Boosting Machines are an ensemble of weak learners. By combining multiple weak models, the result is an ensemble model that is quite effective, and does not overfit because all the models are weak.

Gradient Boosting Machines can be used for Regression or Classification tasks. They are typically applied to tree based models, but could in theory be applied to any type of weak learner.


AdaBoosting
AdaBoosting is the simplies effectibe Boosting algorithm for binary classifcation. It seqentially fits weak learners, and ensembles the predictions. As the weak learners are fit, each observation is weighted by it's missclassification weight, causing the next model to foucus on explaining patterns not detected by the previous models.  Predictions are made by majority vote.

Gradient Boosting
refresher on Gradient Descent. Show plots?

Gradient bossting can be used on both classifcation and regression problems. The weak learners are fit to predict the gradient of the loss function. Any diffentiable loss function used can be selected.Gradient boosting sequentially fits models to the gradient to explain the patterns missed by the previous model. An additive model is used to ensemble the weak learners, as output of the new tree model is added to the output of the previous tree model. It becomes a recursive equation, where each weak learner explains a pattern not picked up by the previous model.
<Show math>


The gradient works out to be the direction of the residuals. insert maths.
<Show math>


Difference between classification and regression loss.
"The generalization allowed arbitrary differentiable loss functions to be used, expanding the technique beyond binary classification problems to support regression, multi-class classification and more."

predictions are made by the resulting additive model. We end up with a big model with lots of terms for each feature, nudging it into different directions?

What is stocastic gradieng Boosting

What is difference between bagging and boosting?

GBMs in Python
sklearn

xgboost
uses regularization
