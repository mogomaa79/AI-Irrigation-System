import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dataset import prepare_data, path

def random_forest(X_train, X_test, y_train, y_test, n_estimators=290, random_state=42, *args, **kwargs):
    """
    Train a Random Forest classifier and evaluate its accuracy.

    Params:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training labels
    - y_test: Testing labels
    - n_estimators: Number of trees in the forest (Measured Best: 290)
    - random_state: Seed for random number generation (Measured Best: 42)

    Ret:
    - random_forest: Trained Random Forest model
    - accuracy: Accuracy of the model on the test set
    """
    random_forest_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    random_forest_model.fit(X_train, y_train)
    y_pred = random_forest_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return random_forest_model, accuracy

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data(path)

    n_estimators = 290
    random_state = 42 
    
    random_forest_model, accuracy = random_forest(X_train, X_test, y_train, y_test, n_estimators, random_state)
    
    print("Random Forest Model Parameters:")
    for param, value in random_forest_model.get_params().items():
        print(f"{param}: {value}")
    
    print(f'Random Forest Model Accuracy: {accuracy * 100:.2f}%')

    # Save the model as 'model.pkl'
    joblib.dump(random_forest_model, 'models/random_forest_model.pkl')
