from __future__ import print_function

import subprocess
import sys      
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'nltk']) 
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'xgboost']) 

import os
os.system('pip install -r requirements.txt')

import argparse
import os
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    #parser.add_argument('--max_leaf_nodes', type=int, default=-1)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Load Data
    print("Loading wine data")
    wine = pd.read_csv( os.path.join(args.train, "winemag-data_first150k.csv"), index_col=0, encoding="utf-8")
    wine.dropna(subset=["price"], inplace=True)
    wine.reset_index(inplace=True, drop=True)
    
    # Feature Engineering
    
    ## Make categorical features
    print("Making categorical features")
    country = pd.get_dummies(wine.country)
    collist = country.columns.tolist()
    collist = ["country_" + s for s in collist]
    country.columns = collist
    print ("There are {} country categorical variables".format(country.shape[1]))

    designation = ohe_features(wine, "designation", 50)
    collist = designation.columns.tolist()
    collist = ["designation_" + s for s in collist]
    designation.columns = collist
    print ("There are {} designation categorical variables".format(designation.shape[1]))

    province = ohe_features(wine, "province", 50)
    collist = province.columns.tolist()
    collist = ["province_" + s for s in collist]
    province.columns = collist
    print ("There are {} province categorical variables".format(province.shape[1]))

    region = ohe_features(wine, "region_1", 50)
    collist = region.columns.tolist()
    collist = ["region_" + s for s in collist]
    region.columns = collist
    print ("There are {} region categorical variables".format(region.shape[1]))

    variety = ohe_features(wine, "variety", 50)
    collist = variety.columns.tolist()
    collist = ["variety_" + s for s in collist]
    variety.columns = collist
    print ("There are {} variety categorical variables".format(variety.shape[1]))

    winery = ohe_features(wine, "winery", 50)
    collist = winery.columns.tolist()
    collist = ["winery_" + s for s in collist]
    winery.columns = collist
    print ("There are {} winery categorical variables".format(winery.shape[1]))
    
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
    X = pd.concat([country, designation, region, variety, winery, lsa_features, wine["price"]], axis=1)
    y = wine["points"]
    
    # Train Model
    ## Format data for xgboost
    print("Training xgboost model!")
    dtrain = xgb.DMatrix(X, label=y)
    
    ## set hyperparameters
    param = {'max_depth': 3, 'eta': 1, 'subsample':0.5, 'alpha':1}
    param['nthread'] = 4
    param['eval_metric'] = 'mae'
    param['objective'] = 'reg:linear'
    param['silent'] = 1
    evallist = [(dtrain, 'train')]
    num_round = 10
    
    ## Train xgboost model
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
    
    ## Save model
    joblib.dump(bst, os.path.join(args.model_dir, "model.joblib"))
    print("Model trained and saved!")

def model_fn(model_dir):
    """Deserialized and return fitted model
    """
    bst = joblib.load(os.path.join(model_dir, "model.joblib"))
    return bst

def predict_fn(input_data, model):
    """DIY predict method
    """
    message_cheap = "Wow, what a steal!"
    message_expensive = "Yikes, that is a spendy wine!"

    #format payload as a dataframe
    x = pd.DataFrame(input_data, index=[0])
    #convert it to xgboost data object
    x = xgb.DMatrix(x, label=y_test)

    #predict wine price from xgboost model
    prediction = bst.predict(x, ntree_limit=bst.best_ntree_limit)
    prediction = np.round(prediction,2)[0]
 
    #personalise message 
    if prediction < 50:
        message = message_cheap
    else:
        message = message_expensive
        
    output = {"price": "$"+str(prediction), "message":message}
    
    return output
