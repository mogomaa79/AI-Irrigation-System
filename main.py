import sys
from sklearn.impute import SimpleImputer
from models.models import RandomForestModel, KNNModel, XGBoostModel

path = "data/raw/TARP.csv"

# Create an imputer object for handling missing values if needed
imputer = SimpleImputer() if "-impute=1" in sys.argv else None

# K-Nearest Neighbors (KNN)
knn = KNNModel(path, imputer=imputer, n_neighbors=84)
acc_knn = knn.train_model()
knn.show_params()

# Random Forest
rf = RandomForestModel(path, imputer=imputer, n_estimators=290, random_state=42)
acc_rf = rf.train_model()
rf.show_params()

# XGBoost
xgb = XGBoostModel(path, imputer=imputer)
acc_xgb = xgb.train_model()
xgb.show_params()