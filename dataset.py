import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
imputer = SimpleImputer(strategy='mean')

NAMES = ["Soil Moisture", "Temperature", " Soil Humidity", "Time", 
         "Air temperature (C)", "Wind speed (Km/h)", "Air humidity (%)",
         "Wind gust (Km/h)", "Pressure (KPa)", "ph", "rainfall", "N", "P", "K",
         "status"
        ]

path = "data/raw/TARP.csv"

def load_data(path, imputer=None):
    """Load data from path, Make Necessary Cleaning & Drop NULL Values"""
    dataset = pd.read_csv(path, names=NAMES, header=0)
    dataset["status"] = dataset["status"].apply(lambda x: 1 if x == "ON" else 0)

    # Deal with data accordingly imputers or without
    if imputer:
        dataset = imputer.fit_transform(dataset)
        return pd.DataFrame(dataset, columns=NAMES)
    else:
        dataset.dropna(inplace=True)
        return dataset
    

def prepare_data(path, imputer=None):
    """Divide Features & Outputs, creates train and test subsets, and scales the values"""
    dataset = load_data(path, imputer)
    X = dataset[["Soil Moisture", "Temperature", " Soil Humidity", "Time", 
            "Air temperature (C)", "Wind speed (Km/h)", "Air humidity (%)",
            "Wind gust (Km/h)", "Pressure (KPa)", "ph", "rainfall", "N", "P", "K"]]

    y = dataset["status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Load Data With Imputer
    data = load_data(path, imputer)
    print(data.head())
    print(data.shape)
    print(data.describe())
    