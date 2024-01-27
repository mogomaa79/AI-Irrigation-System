"""
This module contains utility functions for the SIRI project.
"""


import matplotlib.pyplot as plt
from siri.models import Factory


def find_best_parameter(model_type, path, param_name, param_values):
    """Find the best parameter value for the given model type and parameter name.

    Keyword arguments:
    model_type -- the type of model to use
    path -- the path to the dataset
    param_name -- the name of the parameter to find the best value for
    param_values -- the values to try for the parameter

    Returns:
    best_score -- the best score achieved
    """

    best_score = float('-inf')
    best_value = None
    scores = []

    for value in param_values:
        model = model = Factory.create_model(model_type, path, **{param_name: value})
        score = model.train_model()
        scores.append(score)

        if score > best_score:
            best_score = score
            best_value = value

    # Plotting the graph
    plt.plot(param_values, scores)
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title('Best Parameter Search')
    plt.grid(True)

    # Saving the graph as an image
    plt.savefig(f'img/knn_{param_name}_graph.png')

    return best_score, best_value


# def main():
#     print(find_best_parameter('knn', 'data/raw/TARP.csv',
#                               'n_neighbors', range(1, 100)))
#     print(find_best_parameter('knn', 'data/raw/TARP.csv',
#                               'leaf_size', range(1, 100)))
#     print(find_best_parameter('knn', 'data/raw/TARP.csv',
#                                'algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']))
#     print(find_best_parameter('random_forest', 'data/raw/TARP.csv',
#                               'max_depth', range(15, 30)))
#     print(find_best_parameter('random_forest', 'data/raw/TARP.csv',
#                               'random_state', range(1, 101)))
#     print(find_best_parameter('xgboost', 'data/raw/TARP.csv',
#                                'n_estimators', range(1, 1001)))
#     print(find_best_parameter('xgboost', 'data/raw/TARP.csv',
#                               "gamma", [x / 100 for x in range(1, 101)]))
#     print(find_best_parameter('xgboost', 'data/raw/TARP.csv',
#                               "colsample_bytree", [x / 100 for x in range(1, 101)]))
#     print(find_best_parameter('xgboost', 'data/raw/TARP.csv',
#                               "min_child_weight", [x for x in range(1, 101)]))
#     print(find_best_parameter('xgboost', 'data/raw/TARP.csv',
#                               "learning_rate", [x / 100 for x in range(1, 101)]))
#     print(find_best_parameter('xgboost', 'data/raw/TARP.csv',
#                               "subsample", [x / 100 for x in range(1, 101)]))


# if __name__ == '__main__':
#     main()
