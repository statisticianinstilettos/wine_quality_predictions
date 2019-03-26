from __future__ import print_function
from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

import subprocess
import sys      
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'nltk']) 
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'xgboost']) 

import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import RegexpTokenizer
import xgboost as xgb

def ohe_features(df, feature, occurances):
    '''
    df: pandas data frame with feature to be encoded
    feature: str. feature name
    occurances: number of occurances to threshold feature at
    '''
    vc = df[feature].value_counts()
    keep_values = vc[vc > occurances].index.tolist()
    ohe_feature = pd.get_dummies(df[feature])

    feature_names = ohe_feature.columns
    keep_features = feature_names[feature_names.isin(keep_values)]
    return ohe_feature[keep_features]

def make_lower_case(text):
    return text.lower()

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

def feature_engineering(wine):
    
    ## Clean strings
    print("Cleaning document strings")
    wine["description"] = wine["description"].str.replace('\d+', '')
    wine["description"] = wine.description.apply(func=remove_punctuation)
    wine["description"] = wine.description.apply(func=make_lower_case)
    
    ## Make LSA features
    print("Making latent document features using LSA")
    tf = TfidfVectorizer(analyzer='word', 
                     min_df=10,
                     ngram_range=(1, 2),
                     stop_words='english')
    
    svd = TruncatedSVD(n_components=5)

    tfidf_matrix = tf.fit_transform(wine.description)
    lsa_features = pd.DataFrame(svd.fit_transform(tfidf_matrix))
    collist = map(str, range(0, 5))
    collist = ["latent_description_" + s for s in collist]
    lsa_features.columns = collist
    
    ## Make feature matrix
    X = pd.concat([lsa_features, wine["price"]], axis=1)
    
    ## save TF-IDF and SVD models
    joblib.dump(tf, os.path.join(args.model_dir, "tfidf.joblib"))
    print("TFIDF processor trained and saved!".format(os.path.join(args.model_dir, "tfidf.joblib")))
    
    joblib.dump(svd, os.path.join(args.model_dir, "svd.joblib"))
    print("SVD processor trained and saved!".format(os.path.join(args.model_dir, "svd.joblib")))
    
    return X


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #parse environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    args = parser.parse_args()

    #Load Data
    print("Loading wine data")
    wine = pd.read_csv( os.path.join(args.train, "winemag-data_first150k.csv"), index_col=0, encoding="utf-8")
    wine.dropna(subset=["price"], inplace=True)
    wine.reset_index(inplace=True, drop=True)
    
    ### Feature Engineering ###
    X = feature_engineering(wine)
    y = wine["points"]
    print(X.head(2))
    
    ### Train Model ###
    
    #Format data for xgboost
    print("Training xgboost model!")
    dtrain = xgb.DMatrix(X, label=y)
    
    #Set hyperparameters
    param = {'max_depth': 3, 'eta': 1, 'subsample':0.5, 'alpha':1}
    param['nthread'] = 4
    param['eval_metric'] = 'mae'
    param['objective'] = 'reg:linear'
    param['silent'] = 1
    evallist = [(dtrain, 'train')]
    num_round = 10
    
    #Train xgboost model
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
    
    #Save model
    joblib.dump(bst, os.path.join(args.model_dir, "model.joblib"))
    print("Model trained and saved!".format(os.path.join(args.model_dir, "model.joblib")))

    
#Model Loading
def model_fn(model_dir):
    """Deserialized and return fitted model
    """
    bst = joblib.load(os.path.join(model_dir, "model.joblib"))
    return bst


#Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type):
    """Parse input data payload and return object to make predictions on.
    """
    
    #TODO see if this can be a class var and inherit it from training class.
    model_dir = "/opt/ml/model"
    
    if content_type == "application/json":
        
        #TODO should not have to load these into memmory every time you make a call to the endpoint.
        #load tfidf and svd processors
        tf = joblib.load(os.path.join(model_dir, "tfidf.joblib"))
        svd = joblib.load(os.path.join(model_dir, "svd.joblib"))
        
        #transform input into features
        one_wine = json.loads(request_body)
        #should be a dataframe with 'description' and 'price'
        one_wine = pd.DataFrame(one_wine, index=[0])
        
        #clean strings
        one_wine["description"] = one_wine["description"].str.replace('\d+', '')
        one_wine["description"] = one_wine.description.apply(func=remove_punctuation)
        one_wine["description"] = one_wine.description.apply(func=make_lower_case)
    
        #transform features using LSA
        tfidf_x = tf.transform(one_wine.description)
        lsa_x = pd.DataFrame(svd.transform(tfidf_x))
        collist = map(str, range(0, 5))
        collist = ["latent_description_" + s for s in collist]
        lsa_x.columns = collist
        x = pd.concat([lsa_x, one_wine["price"]], axis=1)
        
        return x
    
    else:
        raise ValueError("{} not supported by script!".format(content_type))


#Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    """DIY predict method
    """
    
    #convert input to xgboost data object
    x = xgb.DMatrix(input_object)

    #predict wine price from xgboost model
    prediction = model.predict(x, ntree_limit=model.best_ntree_limit)
    prediction = np.round(prediction,0)[0]
 
    #personalise message 
    if prediction <= 74:
        message = "Not recommended."
        
    if (prediction >= 75) and (prediction <= 79): 
        message = "Mediocre: a drinkable wine that may have minor flaws."
        
    if (prediction >= 80) and (prediction <= 84): 
        message = "Good: a solid, well-made wine."
        
    if (prediction >= 85) and (prediction <= 89): 
        message = "Very good: a wine with special qualities."
        
    if (prediction >= 90) and (prediction <= 94): 
        message = "Outstanding: a wine of superior character and style."
        
    if prediction >= 95:
        message = "Classic: a great wine."
        
    output = {"predicted_points": str(prediction), "message": message}
    
    return output


#Serialize the prediction result into the desired response content type
def output_fn(prediction, accept):
    """Format prediction output
    """
    if accept == "application/json":
        return worker.Response(json.dumps(prediction), mimetype=accept)
    else:
        raise RuntimeException("{} content_type is not supported by this script.".format(accept))
