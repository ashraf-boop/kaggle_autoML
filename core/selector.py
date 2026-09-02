import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV,ElasticNetCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.model_selection import StratifiedKFold,KFold
import shap

class FeatureSelection:
    def __init__(self):
        self.ElasticNet_Selected_Features = []
        self.LGBM_Selected_Features = []
        self.Union_Features = []
        self.Shap_Selected_Features = []
        self.Final_Selected_Features = []
        self.Is_Small_Or_Medium = False
        self.Is_Classification_Type = False
        self.Class_Imbalance = False

        self.Selector_MetaData ={
        "ElasticNet_Hyperparameters" :{},
        "ElasticNet_CrossValidation_splits" :int, 
        "ElasticNet_Features_Info" :{},
        "LGBM_Hyperparameters" :{},
        "LGBM_Features_Info" :{},
        "ElasticNet_Union_LGBM_Features" :[],
        "Shap_Hyperparameters" :{},
        "Shap_CrossValadation_Splits" :int,
        "Shap_Feature_Info" :{},
        "Final_Selected_Features" :[]
        }


    def Prerequisites(self, X: pd.DataFrame , y: pd.Series):
        ''' This is to calculate and catogrize the data into its length, if its classification
            or regression type and the class imbalance if classification type'''

        self.Is_Small_Or_Medium = len(X) < 10000 
        self.Is_Classification_Type = True if (y.dtype == "object" or str(y.dtype) == "bool" or y.nunique() <= 10) else False

        if self.Is_Classification_Type:
            Class_Count = y.value_counts()
            self.Class_Imbalance = True if (Class_Count.min()/Class_Count.max()) < 0.20 else False
            print("="*40,"DETECTED OBJECTIVE TYPE: CLASSIFICATION","="*40,sep="",end="\n\n")
        else:
            print("="*40,"DETECTED OBJECTIVE TYPE : REGRESSION","="*40,sep="",end="\n\n")

        print("-"*20,"PREREQUISITES HAVE BEEN SET",end="\n\n")



    def ElasticNetFeatureSelection(self, X: pd.DataFrame , y: pd.Series) ->list:
        ''' ElasticNet based feature selection, this is carried out by LogisticRegressionCV and ElasticNetCV
            in hand with StratifiedKFold and KFold considering variable L1 ratio values '''
        
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

            if Classification_Type_ElasticNet_Model.coef_.shape[0] == 1: # Gets the features and their importance as calculated by l1
                Rank = pd.Series(abs(Classification_Type_ElasticNet_Model.coef_[0]),index=X.columns)
            else:
                
                Rank = pd.Series(abs(Classification_Type_ElasticNet_Model.coef_).mean(axis=0),index=X.columns) 
              
        else:
            N_Alpha = 150 if self.Is_Small_Or_Medium else 80
            CV_Regression = KFold(n_splits=Splits,shuffle=True,random_state=69) 

            ElasticNetCV_Hyperparameters = {"max_iter" :Max_Itter,
                                            "cv" :CV_Regression,
                                            "alphas" :None,
                                            "n_alphas" :N_Alpha,
                                            "l1_ratios" : L1_Ratio,
                                            "random_state" :69,
                                            "n_jobs" :-3 }         # Leaves out 2 cores for OS and other operations
            
            self.Selector_MetaData["ElasticNet_CrossValidation_splits"] = Splits
            self.Selector_MetaData["ElasticNet_Hyperparameters"] = ElasticNetCV_Hyperparameters

            Regression_Type_ElasticNet_Model = ElasticNetCV(**ElasticNetCV_Hyperparameters)
            Regression_Type_ElasticNet_Model.fit(X=X,y=y)

            Rank = pd.Series(abs(Regression_Type_ElasticNet_Model.coef_),index=X.columns)

        Rank = Rank[Rank>0].sort_values(ascending=False)
        self.Selector_MetaData["ElasticNet_Features_Info"] = Rank.to_dict()
        self.ElasticNet_Selected_Features = Rank.index.to_list() 

        print(f"ElasticNet selected features amount:         {len(Rank)}")
        print(f"ElasticNet selected features(By rank):       {self.ElasticNet_Selected_Features}")
        print("-"*20,"ELASTICNET BASED FEATURE SELECTION FINISHED",sep="",end="\n\n") 

        return self.ElasticNet_Selected_Features

    def LightGBMFeatureSelector(self , X : pd.DataFrame , y : pd.Series) ->list:
        ''' LightGBM based feature selection, this is done by calculating the gain
            values when the LightGBM based is trained on the training data '''
        
        print("="*40,"LightGBM BASED FEATURE SELECTION STARTED","="*40,sep="",end="\n\n")

        if self.Is_Small_Or_Medium:                
            N_Estimators = 50
            Max_Depth = 4                
            Subsample = 1
            Subsample_Frequency = 0
            Learning_Rate = 0.06
        else:
            N_Estimators = 50             
            Max_Depth = 7                  
            Subsample = 0.8
            Subsample_Frequency = 1
            Learning_Rate = 0.05

        if self.Is_Classification_Type:

            if self.Class_Imbalance:
                Class_Weight = "balanced"
            else:
                Class_Weight = None

            LGBM_Hyperparameters = {"n_estimators" : N_Estimators,
                                    "learning_rate" : Learning_Rate,
                                    "max_depth" : Max_Depth,
                                    "subsample" : Subsample,
                                    "subsample_freq" : Subsample_Frequency,
                                    "class_weight" : Class_Weight,
                                    "random_state" : 69, 
                                    "n_jobs" : -3,
                                    "verbose" : -1} 
            
            LGBM_Model = LGBMClassifier(**LGBM_Hyperparameters)

        else:
            LGBM_Hyperparameters = {"n_estimators" : N_Estimators,
                                    "learning_rate" : Learning_Rate,
                                    "max_depth" : Max_Depth,
                                    "subsample" : Subsample,
                                    "subsample_freq" : Subsample_Frequency,
                                    "random_state" : 69, 
                                    "n_jobs" : -3,
                                    "verbose" : -1}

            LGBM_Model = LGBMRegressor(**LGBM_Hyperparameters)

        self.Selector_MetaData["LGBM_Hyperparameters"] = LGBM_Hyperparameters

        LGBM_Model.fit(X=X,y=y)
        LGBM_Gain = LGBM_Model.booster_.feature_importance(importance_type="gain")
        LGBM_Gain = pd.Series(LGBM_Gain,index = X.columns)
        LGBM_Gain = LGBM_Gain[LGBM_Gain > 0].sort_values(ascending=False)

        self.Selector_MetaData["LGBM_Features_Info"] = LGBM_Gain.to_dict()

        self.LGBM_Selected_Features = LGBM_Gain.index.to_list()

        print(f"LightGBM selected features amount:         {len(LGBM_Gain)}")
        print(f"LightGBM selected features(By rank):       {self.LGBM_Selected_Features}")
        print("-"*20,"LightGBM BASED FEATURE SELECTION FINISHED",sep="",end="\n\n") 

        return self.LGBM_Selected_Features

    

    def ShapFeatureSelection(self , X : pd.DataFrame , y : pd.Series):
        ''' SHAP based feature selection. This is paired up to Random
            Forest tree model '''

        print("="*40,"SHAP BASED FEATURE SELECTION STARTED","="*40,sep="",end="\n\n")

        if self.Is_Small_Or_Medium:                
            N_Estimators = 175
            Max_Depth = 10                  
            Max_Samples = None              
        else:
            Splits = 4
            N_Estimators = 225              
            Max_Depth = 15                  
            Max_Samples = 0.8
            CV = StratifiedKFold(n_splits=Splits,shuffle=True,random_state=69)

        if self.Is_Classification_Type:

            if self.Class_Imbalance:
                Class_Weight = "balanced_subsample"
                Scoring_Metric = "roc_auc" if y.nunique() == 2 else "roc_auc_ovr_weighted"
            else:
                Class_Weight = None
                Scoring_Metric = "neg_log_loss" if y.nunique() == 2 else "accuracy"

            RandomForest_Hyperparameters = {"n_estimators" : N_Estimators,
                                                      "criterion" : "gini",
                                                      "max_depth" : Max_Depth,
                                                      "class_weight" : Class_Weight,
                                                      "max_samples" : Max_Samples,
                                                      "n_jobs" : -3,
                                                      "random_state" : 69
                                                      }
            RandomForestModel = RandomForestClassifier(**RandomForest_Hyperparameters)
        else:

            RandomForest_Hyperparameters = {"n_estimators" : N_Estimators,
                                            "criterion" : "squared_error",
                                            "max_depth" : Max_Depth,
                                            "max_samples" : Max_Samples,
                                            "n_jobs" : -3,
                                            "random_state" : 69
                                            }
            RandomForestModel = RandomForestRegressor(**RandomForest_Hyperparameters)

        RandomForestModel.fit(X=X,y=y)
        

    def Removing_Orphan_Indicators(self , Selected_Features : list) ->list :
        ''' Removes the features (originally added as missing indicators by the imputer in cleaning class)
            where its parent feature (original feature) are removed in feature selection '''

        Surviving_Parents = {
            feature for feature in Selected_Features
            if not feature.startswith("missingindicator_") and not feature.endswith("_missing_value") 
        }              # checks for all the parents and then creates a dict of its name

        Clean_Features = []

        for feature in Selected_Features:

            if feature.startswith("missingindicator_"):
                parent_name = feature[len("missingindicator_"):] # strips the length of "missingindicator_" from the front of the name
            elif feature.endswith("_missing_value"):
                parent_name = feature[:-len("_missing_value")] # strips the length of "_missing_value" from the back of the name
            else:
                parent_name = None

            if parent_name is not None:                     # removes the orphan indicators
                if parent_name in Surviving_Parents:     
                    Clean_Features.append(feature)
            else:
                Clean_Features.append(feature)

        return Clean_Features 
            
                    
                    

            