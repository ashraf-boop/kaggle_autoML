import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV,LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.inspection import permutation_importance

class FeatureSelection:
    def __init__(self, Max_Features = 20):
        self.Max_Features = Max_Features
        self.Lasso_Selected_Features = []
        self.Permutation_Importance_Selected_Features = []
        self.Final_Selected_Features = []
        self.Negative_Permutation_Important_Features = []

        self.Selector_MetaData ={
        "Lasso_Features_Info" :{},
        "Permutation_Importance_Feature_Info" :{},
        "Negative_Permutation_Importance_Feature_Info":{},
        }

    def LassoVerification(self, X: pd.DataFrame , y: pd.Series):
        print("="*40,"LASSO BASED FEATURE SELECTION STARTED","="*40,sep="",end="\n")
        self.Lasso_Selected_Features=[]

        Is_Classification_Type = True if (y.dtype == "object" or str(y.dtype) == "bool" or y.unique() <= 10) else False

        if Is_Classification_Type:
            print("="*40,"DETECTED OBJECTIVE TYPE : CLASSIFICATION","="*40,sep="",end="\n")
            Classification_Type_Lasso_Model = LogisticRegressionCV(penalty="l1",solver="saga",max_iter=4000,random_state=69) # haha funny number
            Classification_Type_Lasso_Model.fit(X=X,y=y)
            
    






