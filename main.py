import sys
from sklearn.impute import SimpleImputer
from siri.models import RandomForestModel, KNNModel, XGBoostModel


def main():
    # K-Nearest Neighbors (KNN)
    knn = KNNModel(PATH, imputer=imputer, n_neighbors=84)
    knn.train_model()
    knn.show_params()

    # Random Forest
    rf = RandomForestModel(PATH, imputer=imputer, n_estimators=290, random_state=42)
    rf.train_model()
    rf.show_params()

    # XGBoost
    xgb = XGBoostModel(PATH, imputer=imputer)
    xgb.train_model()
    xgb.show_params()

if __name__ == "__main__":
    PATH = "data/raw/TARP.csv" if "-path=" not in sys.argv else sys.argv[sys.argv.index("-path=") + 1]
    imputer = SimpleImputer() if "-impute=1" in sys.argv else None
    main()