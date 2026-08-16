import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings

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
        self.Date_Columns = [] # Tracks the columns for Date type columns
        self.Columns_To_Drop = [] # Tracks the universal columns to drop
        self.Is_Fitted = False  # Checks if the Data has already fitted once
        self.Expected_Type = {} # Dict of column datatypes
        self.Is_Small_Or_Medium = False
        self.Is_Sampled = False
        self.Date_Features_To_Keep = []

    def Validating_Data(self,Data:pd.DataFrame):
        if len(Data)==0:
            raise ValueError("The data is empty")
        
        if self.Is_Fitted:
            All_Expected = self.Obj_Columns + self.Num_Columns + self.Date_Columns
            Missing_Columns = [col for col in All_Expected if col not in Data.columns]
            if(Missing_Columns):
                raise KeyError(f"Critical columns missing from the data \n Missing columns:{Missing_Columns} \nCAUGHT IN 'Validating_Data'. \nSTOPPED BEFORE TRANSFORMATION")
                
            for col in All_Expected:
                if col in self.Date_Columns:
                    continue
                Current_Col_Type = "numeric" if np.issubdtype(Data[col].dtype,np.number) else "categorical" #type:ignore
                Expected_Type = self.Expected_Type[col]
                if Current_Col_Type != Expected_Type:
                    warnings.warn(F"DATA DRIFT OCCURED, AFFECTED COLUMN: {col}\nCAUGHT IN 'Validating_Data'. \nSTOPPED BEFORE TRANSFORMATION")
                    
        else:            
            print("VALIDATING DATA DONE, MOVING ON FOR FITTING\n")



    def Fitting(self,df:pd.DataFrame,Target_Column:None):
        self.Validating_Data(df)
        Unique_Ceiling = 0.90 # For deciding to remove a column based on its uniquness only if its an object/string
        self.Columns_To_Drop = []
        self.Date_Columns = []
        Total_Rows = len(df)
        Dataframe_Copy = df.copy()
        self.Is_Small_Or_Medium = True if Total_Rows <= 10000 else False
        
        for col in Dataframe_Copy.columns:  # Identifying the columns to be removed (indentification info) 
            if Target_Column and col == Target_Column:
                continue

            if col.lower() in ["id","uuid","index"]:
                self.Columns_To_Drop.append(col)
                continue
            
            # Checks if the column is in DateTime format structurally or looks like parsable date string 
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.Date_Columns.append(col)
                continue

            if df[col].dtype == "object" or str(df[col].dtype) == "category": 

                # Checks if the non null first cell is present , if all the cells are empty then First_Val is ""
                First_Val = df[col].dropna().iloc[0] if not (df[col].dropna().empty) else ""

                # Checks if the First_Val is a str and of length >=8
                if isinstance(First_Val,str) and len(First_Val) >= 8:
                    try:
                        pd.to_datetime(First_Val,errors="raise")
                        self.Date_Columns.append(col)
                        continue
                    except(ValueError,TypeError):
                        pass

                uniqueness = df[col].nunique() / Total_Rows
                if uniqueness >= Unique_Ceiling :
                    self.Columns_To_Drop.append(col) 
        
        Columns_To_Remove= self.Columns_To_Drop + self.Date_Columns + ([Target_Column] if Target_Column and Target_Column in df.columns else [])
        X = Dataframe_Copy.drop(Columns_To_Remove,axis=1)

        for col in X.columns:
           
            converted_series = pd.to_numeric(X[col], errors='coerce')  # Check failure rate of numeric conversion
            failed_count = converted_series.isna().sum() - X[col].isna().sum()
            failure_rate = failed_count / len(X) if len(X) > 0 else 0

            
            if pd.api.types.is_numeric_dtype(X[col]) or (failure_rate > 0 and failure_rate < 0.05):  # If < 5% fail conversion (or already numeric), treat as numeric column
                self.Num_Columns.append(col)
            else:
                self.Obj_Columns.append(col)

        for col in self.Num_Columns: self.Expected_Type[col] = "numeric"
        for col in self.Obj_Columns: self.Expected_Type[col] = "categorical"

        if self.Num_Columns:
            X_Num_Clean = X[self.Num_Columns].apply(pd.to_numeric,errors = "coerce")
            self.NumFiller.fit(X_Num_Clean)

        if self.Obj_Columns:
            # Any cell having blank/empty value or contains only space and/or tabs gets replaced by NaN 
            X_Obj_Clean = X[self.Obj_Columns].astype(str).replace(r"^\s*$",np.nan,regex=True)
            X_Obj_Clean = X_Obj_Clean.replace(["nan","NaN","null","NULL","N/A","n/a","-"],np.nan)
            self.ObjFiller.fit(X_Obj_Clean)

        Date_Features = pd.DataFrame(index=df.index)
        for col in self.Date_Columns:
            if col in df.columns:
                Datetime_Series = pd.to_datetime(df[col], errors="coerce")
                Date_Features[f"{col}_hour"] = Datetime_Series.dt.hour
                Date_Features[f"{col}_Date"] = Datetime_Series.dt.dayofweek
                Date_Features[f"{col}_Is_Weekend"] = Datetime_Series.dt.dayofweek.isin([5,6]).astype(int)
                Date_Features[f"{col}_month"] = Datetime_Series.dt.month
                Date_Features[f"{col}_year"] = Datetime_Series.dt.year

        # Store only non-constant features learned from training data
        self.Date_Features_To_Keep = [feat for feat in Date_Features.columns if Date_Features[feat].nunique() > 1]

        self.Is_Fitted = True
        print("FITTING DATA CLEANED,VALIDATED AND FITTED INTO THE STANDARD SCALER...")
        return self


    def Transformation(self, df:pd.DataFrame, Target_Column:None):
        if not self.Is_Fitted:
            raise RuntimeError("THE DATA HAS NOT BEEN FITTED, U MUST CALL .Fitting() BEFORE .Transformation()")
        Dataframe_Copy = df.copy()

        # Handles the conversion of DateTime data into 5 catogries of features
        Date_Features = pd.DataFrame(index=Dataframe_Copy.index)
        for col in self.Date_Columns:
            if col in Dataframe_Copy.columns:
                Datetime_Series = pd.to_datetime(Dataframe_Copy[col], errors="coerce")
                
                Date_Features[f"{col}_hour"] = Datetime_Series.dt.hour
                Date_Features[f"{col}_Date"] = Datetime_Series.dt.dayofweek
                Date_Features[f"{col}_Is_Weekend"] = Datetime_Series.dt.dayofweek.isin([5,6]).astype(int)
                Date_Features[f"{col}_month"] = Datetime_Series.dt.month
                Date_Features[f"{col}_year"] = Datetime_Series.dt.year
                
        # If there exists a column from the above with only 0 or singular value it gets dropped
        if not Date_Features.empty:
            modes = Date_Features.mode()
            fill_val = modes.iloc[0] if not modes.empty else 0
            Date_Features = Date_Features.fillna(fill_val)

        # Enforce the exact features learned during Fitting()
        X_Date = Date_Features.reindex(columns=self.Date_Features_To_Keep, fill_value=0)
        Columns_To_Remove = self.Columns_To_Drop + self.Date_Columns + ([Target_Column] if Target_Column and Target_Column in df.columns else [])

        X = Dataframe_Copy.drop(Columns_To_Remove,axis=1)  # Feature Dataset

        if Target_Column and Target_Column in Dataframe_Copy.columns:
            y_Datatype = Dataframe_Copy[Target_Column].dtype
            if y_Datatype == "bool":
                y = Dataframe_Copy[Target_Column].astype(int)
            else:
                y = Dataframe_Copy[Target_Column]
        else:
            y = None

        if self.Num_Columns:
            X_Num_Data = X[self.Num_Columns].apply(pd.to_numeric, errors="coerce") # Keeps the data numeric
            X_Num_Clean = self.NumFiller.transform(X_Num_Data)
            # gets the names of the new features created and the old ones 
            X_Num_Columns = self.NumFiller.get_feature_names_out(self.Num_Columns) if hasattr(self.NumFiller,"get_feature_names_out") else self.Num_Columns
            X_Num = pd.DataFrame(X_Num_Clean,columns=X_Num_Columns,index=X.index) # Stitches back the data into a Dataframe based on the index
        else:
            X_Num = pd.DataFrame(index=X.index)

        if self.Obj_Columns:
            # Any cell having blank/empty value or contains only space and/or tabs gets replaced by NaN
            X_Obj_Data= X[self.Obj_Columns].astype(str).replace(r"^\s*$",np.nan,regex=True) 
            X_Obj_Data = X_Obj_Data.replace(["nan","NaN","null","NULL","N/A","n/a","-"],np.nan)
            X_Obj_Clean = self.ObjFiller.transform(X_Obj_Data)
            X_Obj_Columns = self.ObjFiller.get_feature_names_out(self.Obj_Columns)
            X_Obj = pd.DataFrame(X_Obj_Clean,columns=X_Obj_Columns,index=X.index) # Stitches back the dataframe for object datatype
        else:
            X_Obj = pd.DataFrame(index=X.index)
        
        X_Final = pd.concat([X_Num,X_Obj,X_Date],axis=1)
        # The first replace function replaces all the illegal characters and replaces them, the second one replaces spaces
        X_Final.columns = X_Final.columns.str.replace(r"[^\w\s\-]","",regex=True).str.replace(" ","_")
        print("TRANSFORMATION DATA IS CLEANED AND TRANSFORMED...\n")
        
        return ((X_Final,y) if y is not None else X_Final)
    
    def Sampling(self, X: pd.DataFrame, y: pd.Series):
        if self.Is_Small_Or_Medium:
            print()