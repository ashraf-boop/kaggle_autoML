import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV,ElasticNetCV
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.inspection import permutation_importance

class FeatureSelection:
    def __init__(self):
        self.ElasticNet_Selected_Features = []
        self.Permutation_Importance_Selected_Features = []
        self.Final_Selected_Features = []
        self.Negative_Permutation_Important_Features = []
        self.Is_Small_Or_Medium = False
        self.Is_Classification_Type = False
        self.Class_Imbalance = False

        self.Selector_MetaData ={
        "ElasticNet_Hyperparameters" :{},
        "ElasticNet_CrossValidation_splits" :int, 
        "ElasticNet_Features_Info" :{},
        "Permutation_Importance_Hyperparameters" :{},
        "Permutation_Importance_CrossValadation_Splits" :int,
        "Permutation_Importance_Feature_Info" :{},
        "Negative_Permutation_Importance_Feature_Info" :{},
        "Final_Selected_Features" :[]
        }


    def Prerequisites(self, X: pd.DataFrame , y: pd.Series):
        self.Is_Small_Or_Medium = len(X) < 10000 
        self.Is_Classification_Type = True if (y.dtype == "object" or str(y.dtype) == "bool" or y.nunique() <= 10) else False

        if self.Is_Classification_Type:
            Class_Count = y.value_counts()
            self.Class_Imbalance = True if (Class_Count.min()/Class_Count.max()) < 0.20 else False
            print("="*40,"DETECTED OBJECTIVE TYPE: CLASSIFICATION","="*40,sep="",end="\n\n")
        else:
            print("="*40,"DETECTED OBJECTIVE TYPE : REGRESSION","="*40,sep="",end="\n\n")

        print("-"*20,"PREREQUISITES HAVE BEEN SET",end="\n\n")



    def ElasticNetVerification(self, X: pd.DataFrame , y: pd.Series):
        print("="*40,"ELASTICNET BASED FEATURE SELECTION STARTED","="*40,sep="",end="\n\n")

        if self.Is_Small_Or_Medium:
            Splits = 8
            Max_Itter = 6000
            L1_Ratio = [0.7, 0.8, 0.85, 0.9, 0.95]
        else:
            Splits = 4
            Max_Itter = 3000
            L1_Ratio = [0.9, 0.95]

        if self.Is_Classification_Type:
            CV_Classification = StratifiedKFold(n_splits=Splits,random_state=69,shuffle=True)

            if self.Class_Imbalance:
                Class_Weight = "balanced"
                Scoring_Metric = "roc_auc" if y.nunique() == 2 else "f1_weighted"
            else:
                Class_Weight = None
                Scoring_Metric = "neg_log_loss" if y.nunique() == 2 else "accuracy"

            LogisticRegressionCV_Hyperparameters = {"penalty" : "elasticnet",
                                                    "solver" :"saga",
                                                    "max_iter" :Max_Itter,
                                                    "random_state" :69,
                                                    "n_jobs":-3,
                                                    "cv" :CV_Classification,
                                                    "Cs" :100,
                                                    "l1_ratios" : L1_Ratio,
                                                    "class_weight": Class_Weight,    
                                                    "scoring": Scoring_Metric
                                                    }
            
            self.Selector_MetaData["ElasticNet_CrossValidation_splits"] = Splits
            self.Selector_MetaData["ElasticNet_Hyperparameters"] = LogisticRegressionCV_Hyperparameters 
            
            Classification_Type_ElasticNet_Model = LogisticRegressionCV(**LogisticRegressionCV_Hyperparameters)
            Classification_Type_ElasticNet_Model.fit(X=X,y=y)

            if Classification_Type_ElasticNet_Model.coef_.shape[0] == 1:
                Rank = pd.Series(abs(Classification_Type_ElasticNet_Model.coef_[0]),index=X.columns)
            else:
                Rank = pd.Series(abs(Classification_Type_ElasticNet_Model.coef_).mean(axis=0),index=X.columns) #Gets the features and their importance as calculated by l1
              
        else:
            N_Alpha = 150 if self.Is_Small_Or_Medium else 80
            CV_Regression = KFold(n_splits=Splits,shuffle=True,random_state=69) 

            ElasticNetCV_Hyperparameters = {"max_iter" :Max_Itter,
                                            "random_state" :69,
                                            "n_jobs" :-3,           # Leaves out 2 cores for OS and other operations
                                            "cv" :CV_Regression,
                                            "alphas" :None,
                                            "n_alphas" :N_Alpha,
                                            "l1_ratios" : L1_Ratio}
            
            self.Selector_MetaData["ElasticNet_CrossValidation_splits"] = Splits
            self.Selector_MetaData["ElasticNet_Hyperparameters"] = ElasticNetCV_Hyperparameters

            Regression_Type_ElasticNet_Model = ElasticNetCV(**ElasticNetCV_Hyperparameters)
            Regression_Type_ElasticNet_Model.fit(X=X,y=y)

            Rank = pd.Series(abs(Regression_Type_ElasticNet_Model.coef_),index=X.columns)


        Rank = Rank[Rank>0].sort_values(ascending=False)
        self.Selector_MetaData["ElasticNet_Features_Info"] = Rank.to_dict()
        self.ElasticNet_Selected_Features = Rank.index.to_list() 
        print(f"ElasticNet selected features amount:         {len(Rank)}")
        print(f"ElasticNet selected features:                {Rank.index.to_list()}")
        print("-"*20,"ELASTICNET BASED FEATURE SELECTION FINISHED",sep="",end="\n\n") 

    def PermutationImportanceVerification(self , X : pd.DataFrame , y : pd.Series):
        print("="*40,"PERMUTATION IMPORTANCE BASED FEATURE SELECTION STARTED","="*40,sep="",end="\n\n")

        if self.Is_Small_Or_Medium:
            Splits = 8 if len(X) < 2000 else 5
            N_Repeats = 10                  
            N_Estimators = 175
            Max_Depth = 10                  
            Max_Samples = None              
        else:
            Splits = 4
            N_Repeats = 5
            N_Estimators = 225              
            Max_Depth = 15                  
            Max_Samples = 0.8
            
        if self.Is_Classification_Type:

            CV = StratifiedKFold(n_splits=Splits,shuffle=True,random_state=69)

            if self.Class_Imbalance:
                Class_Weight = "balanced_subsample"
                Scoring_Metric = "roc_auc" if y.nunique() == 2 else "roc_auc_ovr_weighted"
            else:
                Class_Weight = None
                Scoring_Metric = "neg_log_loss" if y.nunique() == 2 else "accuracy"

            RandomForestClassifier_Hyperparameters = {"n_estimators" : N_Estimators,
                                                      "criterion" : "gini",
                                                      "max_depth" : Max_Depth,
                                                      "class_weight" : Class_Weight,
                                                      "max_samples" : Max_Samples,
                                                      "n_jobs" : -3,
                                                      "random_state" : 69
                                                      }