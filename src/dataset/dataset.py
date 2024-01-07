"""
Dataset Module includes all classes required for loading, cleaning, and imputing data
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """
    DataLoader class for loading and preparing data.
    """

    def __init__(self, path, imputer=None):
        """
        Initialize DataLoader with the given path and optional imputer.

        Params:
        - path: Path to the dataset.
        - imputer: Imputer object for handling missing values.
        """
        self.path = path
        self.imputer = imputer
        self.scaler = StandardScaler()
        self.names = ["Soil Moisture", "Temperature", "Soil Humidity", "Time",
                      "Air temperature (C)", "Wind speed (Km/h)", "Air humidity (%)",
                      "Wind gust (Km/h)", "Pressure (KPa)", "ph", "rainfall", "N", "P", "K",
                      "status"]

    def load_data(self):
        """
        Load data from the specified path, perform necessary cleaning, and handle NULL values.

        Returns:
        - pd.DataFrame: Loaded and processed DataFrame.
        """
        dataset = pd.read_csv(self.path, names=self.names, header=0)
        dataset["status"] = dataset["status"].apply(lambda x: 1 if x == "ON" else 0)

        if self.imputer:
            dataset = self.imputer.fit_transform(dataset)
            return pd.DataFrame(dataset, columns=self.names)

        dataset.dropna(inplace=True)
        return dataset

    def prepare_data(self):
        """
        Divide features and outputs, create train and test subsets, and scale the values.

        Returns:
        - tuple: (x_train_scaled, x_test_scaled, y_train, y_test)
        """
        dataset = self.load_data()
        x = dataset[["Soil Moisture", "Temperature", "Soil Humidity", "Time",
                     "Air temperature (C)", "Wind speed (Km/h)", "Air humidity (%)",
                     "Wind gust (Km/h)", "Pressure (KPa)", "ph", "rainfall", "N", "P", "K"]]

        y = dataset["status"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=46)

        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        return x_train_scaled, x_test_scaled, y_train, y_test
