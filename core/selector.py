import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV,ElasticNetCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.model_selection import StratifiedKFold,KFold,train_test_split
import shap

class FeatureSelection:
    def __init__(self):
        self.ElasticNet_Selected_Features = []
        self.LGBM_Selected_Features = []
        self.Union_Features = []
        self.SHAP_Selected_Features = []
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
        "SHAP_Hyperparameters" :{},
        "SHAP_CrossValadation_Splits" :int,
        "SHAP_Feature_Info" :{},
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
                                                    "n_jobs":-3,                     # Leaves out 2 cores for OS and other operations
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

            if Classification_Type_ElasticNet_Model.coef_.shape[0] == 1:     # Gets the features and their importance as calculated by l1
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
                                            "n_jobs" :-3 }        
            
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
            if self.Is_Classification_Type:
                CV = StratifiedKFold(n_splits=Splits,shuffle=True,random_state=69)
            else:
                CV = KFold(n_splits=Splits,shuffle=True,random_state=69)

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
        else:

            RandomForest_Hyperparameters = {"n_estimators" : N_Estimators,
                                            "criterion" : "squared_error",
                                            "max_depth" : Max_Depth,
                                            "max_samples" : Max_Samples,
                                            "n_jobs" : -3,
                                            "random_state" : 69
                                            }

        RandomForestModel = RandomForestClassifier(**RandomForest_Hyperparameters) if self.Is_Classification_Type else RandomForestRegressor(**RandomForest_Hyperparameters) 

        if self.Is_Small_Or_Medium:
            X_train,X_valid,y_train,y_valid = train_test_split(X,y,train_size=0.8,random_state=69,
                                                               stratify=True if self.Is_Classification_Type else False)
            RandomForestModel.fit(X=X_train,y=y_train)

            SHAP_Explainer = shap.TreeExplainer(RandomForestModel)
            SHAP_Values = SHAP_Explainer.shap_values(X=X_valid,check_additivity=False)  # guardrail to prevent tiny floating-point 
                                                                                        # imprecisions from throwing a hard ConvergenceError 

            if isinstance(SHAP_Values,list):                                                    # For multiclass output
                Mean_Shap = np.mean([np.abs(arr).mean(axis=0) for arr in SHAP_Values],axis=0) 
            elif len(SHAP_Values.shape) == 3 :                                                  # (Samples,Fatures,Classes)
                Mean_Shap = np.abs(SHAP_Values).mean(axis = (0,2))
            else:                                                                               # (Samples,Features)
                Mean_Shap = np.abs(SHAP_Values).mean(axis=0)

            Feature_Importance = pd.Series(Mean_Shap,index = X.columns)

        else:
            OOF_SHAP_Score = np.zeros(shape=X.shape[1])

            for fold,(train_idx,valid_idx) in enumerate(CV.split(X=X,y=y)): # CV is not unbound as CV gets initialized under the same 
                                                                            # condition when this else block gets called
                X_train_fold,X_valid_fold = X.iloc[train_idx],X.iloc[valid_idx]
                y_train_fold = y.iloc[train_idx]

                RandomForestModel.fit(X=X_train_fold,y=y_train_fold)

                SHAP_Explainer = shap.TreeExplainer(RandomForestModel)
                SHAP_Values = SHAP_Explainer.shap_values(X=X_valid_fold,check_additivity=False)

                if isinstance(SHAP_Values,list):                                                    
                    Mean_Shap = np.mean([np.abs(arr).mean(axis=0) for arr in SHAP_Values],axis=0) 
                elif len(SHAP_Values.shape) == 3 :                                                  
                    Mean_Shap = np.abs(SHAP_Values).mean(axis = (0,2))
                else:                                                                               
                    Mean_Shap = np.abs(SHAP_Values).mean(axis=0)    

                OOF_SHAP_Score += Mean_Shap / Splits # Splits is not unbound as it also gets initilized under the same condition
                                                     # under when this else block gets called     
            Feature_Importance = pd.Series(OOF_SHAP_Score,index = X.columns)

        SHAP_Selected_Features = Feature_Importance[Feature_Importance > 1e-5].sort_values(ascending=False).index.to_list()

        self.SHAP_Selected_Features = SHAP_Selected_Features
        self.Selector_MetaData["SHAP_Hyperparameters"] = RandomForest_Hyperparameters
        self.Selector_MetaData["SHAP_Feature_Info"] = Feature_Importance.to_dict()

        print(f"SHAP selected features amount:         {len(SHAP_Selected_Features)}")
        print(f"SHAP selected features(By rank):       {self.SHAP_Selected_Features}")
        print("-"*20,"SHAP BASED FEATURE SELECTION FINISHED",sep="",end="\n\n") 

        return SHAP_Selected_Features

    def Removing_Orphan_Indicators(self , Selected_Features : list) ->list :
        ''' Removes the features (originally added as missing indicators by the imputer in cleaning class)
            where it's parent feature (original feature) are removed in feature selection '''

        Surviving_Parents = {
            feature for feature in Selected_Features
            if not feature.startswith("missingindicator_") and not feature.endswith("_missing_value") 
        }              # checks for all the parents and then creates a dict of its name

        Clean_Features = []

        for feature in Selected_Features:

            if feature.startswith("missingindicator_"):
                parent_name = feature[len("missingindicator_"):]  # strips the length of "missingindicator_" from the front of the name
            elif feature.endswith("_missing_value"):
                parent_name = feature[:-len("_missing_value")]    # strips the length of "_missing_value" from the back of the name
            else:
                parent_name = None

            if parent_name is not None:                           # removes the orphan indicators
                if parent_name in Surviving_Parents:     
                    Clean_Features.append(feature)
            else:
                Clean_Features.append(feature)

        return Clean_Features            