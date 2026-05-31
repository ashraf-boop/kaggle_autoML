import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

class DataCleaner:

    def __init__(self):
        # For number based missing values filling and scaling
        self.NumFiller =Pipeline(steps=[
            ("impuder",SimpleImputer(strategy="median",add_indicator=True)),
            ("scaling",StandardScaler()) 
            ]) 
        
        # For Object based missing values filling and encoding
        self.ObjFiller = Pipeline(steps=[
            ("impuder",SimpleImputer(strategy="most_frequent",add_indicator=True)),
            ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False)) 
            ]) 
        
        self.Num_Columns = []  # Track the columns for numerical columns
        self.Obj_Columns = []  # Track the columns for object columns
        self.Columns_To_Drop = []

    def Transformation(self, df: pd.DataFrame, target_column: None):
         
        Unique_Ceiling = 0.90 # For deciding to remove a column based on its uniquness only if its an object/string
        self.Columns_To_Drop = []
        Total_Rows = len(df)
        Dataframe_Copy = df.copy()

        for col in df.columns:  # Identifying the columns to be removed (indentification info) 
            if col == target_column:
                continue

            if col.lower() in ["id","uuid","index"]:
                self.Columns_To_Drop.append(col)
                continue

            if df[col].dtype == "object" or str(df[col].dtype) == "category": 
                uniqueness = df[col].nunique() / Total_Rows
                if uniqueness >= Unique_Ceiling :
                    self.Columns_To_Drop.append(col) 
            
        Modified_Data_Copy = Dataframe_Copy.drop(self.Columns_To_Drop,axis=1) # removing the indentifier columns

        X = Modified_Data_Copy.drop(target_column,axis=1) # Feature Dataset
        y = Modified_Data_Copy[target_column] # target Values

        self.Num_Columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.Obj_Columns = X.select_dtypes(include=["object","category"]).columns.tolist()

        if self.Num_Columns:
            X_Num_Clean = self.NumFiller.fit_transform(X[self.Num_Columns])
            # gets the names of the new features created and athe old ones 
            X_Num_Columns =self.NumFiller.get_feature_names_out(self.Num_Columns) if hasattr(self.NumFiller,"get_feature_names_out") else self.Num_Columns
            X_Num = pd.DataFrame(X_Num_Clean,columns=X_Num_Columns,index=X.index) # Stitches back the data into a Dataframe based on the index
        else:
            X_Num = pd.DataFrame(index=X.index)

        if self.Obj_Columns:
            X[self.Obj_Columns] = X[self.Obj_Columns].replace(r"^\s*$",np.nan,regex=True) # replacing empty space with NaN
            X[self.Obj_Columns] = X[self.Obj_Columns].replace(["nan","NaN","null","NULL","N/A","n/a","-"],np.nan)

            X_Obj_Clean = self.ObjFiller.fit_transform(X[self.Obj_Columns])
            X_Obj_Columns = self.ObjFiller.get_feature_names_out(self.Obj_Columns)
            X_Obj = pd.DataFrame(X_Obj_Clean,columns=X_Obj_Columns,index=X.index)
        else:
            X_Obj = pd.DataFrame(index=X.index)
        
        X_Final = pd.concat([X_Num,X_Obj],axis=1)
        # The first replace function replaces all the illegal characters and replaces them, the second one replaces spaces
        X_Final.columns = X_Final.columns.str.replace(r"[^\w\s\-]","",regex=True).str.replace(" ","_")

        return ((X_Final,y) if target_column else y)