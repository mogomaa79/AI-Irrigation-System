import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, path, imputer=None):
        self.path = path
        
        self.imputer = imputer
        
        self.scaler = StandardScaler()

        self.names = ["Soil Moisture", "Temperature", " Soil Humidity", "Time", 
                      "Air temperature (C)", "Wind speed (Km/h)", "Air humidity (%)",
                      "Wind gust (Km/h)", "Pressure (KPa)", "ph", "rainfall", "N", "P", "K",
                      "status"]

    def load_data(self):
        """Load data from path, make necessary cleaning, and handle NULL values"""
        dataset = pd.read_csv(self.path, names=self.names, header=0)
        dataset["status"] = dataset["status"].apply(lambda x: 1 if x == "ON" else 0)

        if self.imputer:
            dataset = self.imputer.fit_transform(dataset)
            return pd.DataFrame(dataset, columns=self.names)
        else:
            dataset.dropna(inplace=True)
            return dataset

    def prepare_data(self):
        """Divide features and outputs, create train and test subsets, and scale the values"""
        dataset = self.load_data()
        X = dataset[["Soil Moisture", "Temperature", " Soil Humidity", "Time", 
                     "Air temperature (C)", "Wind speed (Km/h)", "Air humidity (%)",
                     "Wind gust (Km/h)", "Pressure (KPa)", "ph", "rainfall", "N", "P", "K"]]

        y = dataset["status"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Create DataLoader instance
    path = "../../data/raw/TARP.csv"
    data_loader = DataLoader(path)

    # Load Data With Imputer
    data = data_loader.load_data()
    print(data.head())
    print(data.shape)
    print(data.describe())
