import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold,train_test_split
from sklearn.linear_model import ElasticNet,LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost import XGBClassifier,XGBRegressor


optuna.logging.set_verbosity(optuna.logging.WARNING) # Silences optuna logs so that it dosen't flood the terminal

class ModelTrainer:
    def __init__(self, Is_Classification_Type : bool, Is_Small_Or_Medium : bool,
                  Class_Imbalance : bool) -> None:
        
        self.Is_Classification_Type = Is_Classification_Type
        self.Is_Small_Or_Medium = Is_Small_Or_Medium
        self.Class_Imbalance = Class_Imbalance
        self.Best_Score_Model = None
        
        

        self.TrainerMetadata = {
            "XGBoost_Best_Features" : {},
            "RandomForest_Best_Features" : {},
            "ElasticNet_Best_Features" : {},
            "XGBoost_Score" : float,
            "RandomForest_Score" : float,
            "ElasticNet_Score" : float,
            "Best_Score_Model" : None
        }

    def _Get_Model_Instance(self, Trial, Model_Name: str):
        """Constructs the targeted model with dynamically
           suggested hyperparameter ranges."""

        if Model_Name == "Linear":
            if self.Is_Classification_Type:
                Lin_Params = {
                    "penalty": "elasticnet",
                    "solver": "saga",
                    "C": Trial.suggest_float("lin_C", 1e-4, 3.0 if self.Is_Small_Or_Medium else 10.0, log=True),
                    "l1_ratio": Trial.suggest_float("lin_l1_ratio", 0.0, 1.0),
                    "max_iter": Trial.suggest_int("lin_max_iter", 300, 2000 if self.Is_Small_Or_Medium else 5000),
                    "random_state": 69,
                    "n_jobs": -3
                }
                if self.Class_Imbalance:
                    Lin_Params["class_weight"] = "balanced"
                return LogisticRegression(**Lin_Params)
            else:
                Lin_Params = {
                    "alpha": Trial.suggest_float("lin_alpha", 1e-4, 3.0 if self.Is_Small_Or_Medium else 10.0, log=True),
                    "l1_ratio": Trial.suggest_float("lin_l1_ratio", 0.0, 1.0),
                    "max_iter": Trial.suggest_int("lin_max_iter", 300, 2000 if self.Is_Small_Or_Medium else 5000),
                    "random_state": 69,
                    "n_jobs": -3
                }
                return ElasticNet(**Lin_Params)

        elif Model_Name == "XGBoost":
            XGB_Params = {
                "n_estimators": 3000,  # high ceiling; early stopping picks the real number
                "max_depth": Trial.suggest_int("xgb_max_depth", 2, 6 if self.Is_Small_Or_Medium else 10),
                "learning_rate": Trial.suggest_float("xgb_lr", 1e-3, 0.2 if self.Is_Small_Or_Medium else 0.1, log=True),
                "subsample": Trial.suggest_float("xgb_subsample", 0.7 if self.Is_Small_Or_Medium else 0.5, 1.0),
                "colsample_bytree": Trial.suggest_float("xgb_colsample_bytree", 0.7 if self.Is_Small_Or_Medium else 0.5, 1.0),
                "reg_alpha": Trial.suggest_float("xgb_reg_alpha", 0.1, 10.0 if self.Is_Small_Or_Medium else 25.0, log=True),
                "reg_lambda": Trial.suggest_float("xgb_reg_lambda", 1.0, 10.0 if self.Is_Small_Or_Medium else 25.0, log=True),
                "tree_method": "exact" if self.Is_Small_Or_Medium else "hist",
                "early_stopping_rounds": 100 if self.Is_Small_Or_Medium else 30,
                "random_state": 69,
                "n_jobs": -3
            }
            if self.Is_Classification_Type:
                return XGBClassifier(**XGB_Params)
            else:
                return XGBRegressor(**XGB_Params)

        else:  # RandomForest
            RF_Params = {
                "n_estimators": Trial.suggest_int("rf_n_estimators", 50, 250 if self.Is_Small_Or_Medium else 400),
                "max_depth": Trial.suggest_int("rf_max_depth", 5, 20 if self.Is_Small_Or_Medium else 25),
                "max_features": Trial.suggest_float("rf_max_features", 0.3, 1.0),
                "min_samples_split": Trial.suggest_int("rf_min_samples_split", 2, 20 if self.Is_Small_Or_Medium else 100),
                "min_samples_leaf": Trial.suggest_int("rf_min_samples_leaf", 1, 20 if self.Is_Small_Or_Medium else 50),
                "max_samples": Trial.suggest_float("rf_max_samples", 0.3, 0.8) if not self.Is_Small_Or_Medium else None,
                "random_state": 69,
                "n_jobs": -3
            }
            if self.Is_Classification_Type:
                RF_Params["criterion"] = "gini"
                if self.Class_Imbalance:
                    RF_Params["class_weight"] = "balanced_subsample"
                return RandomForestClassifier(**RF_Params)

            RF_Params["criterion"] = "squared_error"
            return RandomForestRegressor(**RF_Params)