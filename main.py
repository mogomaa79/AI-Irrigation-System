import sys
from sklearn.impute import SimpleImputer
from siri.models import RandomForestModel, KNNModel, XGBoostModel


def main():
    # K-Nearest Neighbors (KNN)
    knn = KNNModel(PATH, imputer=imputer, 
                   n_neighbors=84, leaf_size=1,
                   weights='distance', p=1)
    knn.train_model()
    knn.show_params()

    # Random Forest
    rf = RandomForestModel(PATH, imputer=imputer, 
                           n_estimators=290, random_state=42,
                           max_depth=22)
    rf.train_model()
    rf.show_params()

    # XGBoost
    xgb = XGBoostModel(PATH, 
                       imputer=imputer, alpha=0.005, 
                       reg_lambda=0.001, colsample_bytree=0.8, 
                       gamma=0.1, min_child_weight=1.0,
                       learning_rate=0.3, max_depth=6, 
                       n_estimators=54, subsample=1.0)
    xgb.train_model()
    xgb.show_params()


if __name__ == "__main__":
    PATH = "data/raw/TARP.csv" if "-path=" not in sys.argv else sys.argv[sys.argv.index("-path=") + 1]
    imputer = SimpleImputer() if "-impute=1" in sys.argv else None
    main()
