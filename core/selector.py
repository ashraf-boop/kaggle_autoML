import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV,LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.inspection import permutation_importance

class FeatureSelection:
    def __init__(self):
        self.Lasso_Selected_Features = []
        self.Permutation_Importance_Selected_Features = []
        self.Final_Selected_Features = []
        self.Negative_Permutation_Important_Features = []
        self.Is_Small_Or_Medium = False

        self.Selector_MetaData ={
        "Lasso_Hyperparameters" :{},
        "Lasso_CrossValidation_splits" :int, 
        "Lasso_Features_Info" :{},
        "Permutation_Importance_Hyperparameters" :{},
        "Permutation_Importance_CrossValadation_Splits" :int,
        "Permutation_Importance_Feature_Info" :{},
        "Negative_Permutation_Importance_Feature_Info" :{},
        "Final_Selected_Features" :[]
        }


    def LassoVerification(self, X: pd.DataFrame , y: pd.Series):
        print("="*40,"LASSO BASED FEATURE SELECTION STARTED","="*40,sep="",end="\n")

        self.Is_Small_Or_Medium = len(X) < 10000 
        Is_Classification_Type = True if (y.dtype == "object" or str(y.dtype) == "bool" or y.nunique() <= 10) else False

        Splits = 8 if self.Is_Small_Or_Medium else 4
        Max_Itter = 6000 if self.Is_Small_Or_Medium else 3000

        if Is_Classification_Type:
            print("="*40,"DETECTED OBJECTIVE TYPE : CLASSIFICATION","="*40,sep="",end="\n")
            

            CV_Classification = StratifiedKFold(n_splits=Splits,random_state=69,shuffle=True)

            LogisticRegressionCV_Hyperparameters = {"penalty" : "l1",
                                                    "solver" :"saga",
                                                    "max_iter" :Max_Itter,
                                                    "random_state" :69,
                                                    "n_jobs":-3,
                                                    "cv" :CV_Classification,
                                                    "Cs" :100}
            
            self.Selector_MetaData["Lasso_CrossValidation_splits"] = Splits
            self.Selector_MetaData["Lasso_Hyperparameters"] = LogisticRegressionCV_Hyperparameters 
            
            Classification_Type_Lasso_Model = LogisticRegressionCV(**LogisticRegressionCV_Hyperparameters)
            Classification_Type_Lasso_Model.fit(X=X,y=y)

            if Classification_Type_Lasso_Model.coef_.shape[0] == 1:
                Rank = pd.Series(abs(Classification_Type_Lasso_Model.coef_[0]),index=X.columns)
            else:
                Rank = pd.Series(abs(Classification_Type_Lasso_Model.coef_).mean(axis=0),index=X.columns) #Gets the features and their importance as calculated by l1
              
        else:
            print("="*40,"DETECTED OBJECTIVE TYPE : REGRESSION","="*40,sep="",end="\n")
            N_Alpha = 150 if self.Is_Small_Or_Medium else 80

            CV_Regression = KFold(n_splits=Splits,shuffle=True,random_state=69) 

            LassoCV_Hyperparameters = {"max_iter" :Max_Itter,
                                       "random_state" :69,
                                       "n_jobs" :-3,           # Leaves out 2 cores for OS and other operations
                                       "cv" :CV_Regression,
                                       "alphas" :None,
                                       "n_alphas" :N_Alpha}
            
            self.Selector_MetaData["Lasso_CrossValidation_splits"] = Splits
            self.Selector_MetaData["Lasso_Hyperparameters"] = LassoCV_Hyperparameters

            Regression_Type_Lasso_Model = LassoCV(**LassoCV_Hyperparameters)
            Regression_Type_Lasso_Model.fit(X=X,y=y)

            Rank = pd.Series(abs(Regression_Type_Lasso_Model.coef_),index=X.columns)


        Rank = Rank[Rank>0].sort_values(ascending=False)
        self.Selector_MetaData["Lasso_Features_Info"] = Rank.to_dict()
        self.Lasso_Selected_Features = Rank.index.to_list() 
        print("="*40,"LASSO BASED FEATURE SELECTION FINISHED","="*40,sep="",end="\n") 
    


