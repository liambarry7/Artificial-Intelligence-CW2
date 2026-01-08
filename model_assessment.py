# -*- coding: utf-8 -*-
"""
Practical assignment on supervised and unsupervised learning
Coursework 002 for: CMP-6058A Artificial Intelligence

Script containing the analysis of different classification models

@author: Liam
@date:   05/01/2026

"""
import pandas as pd

from data_handling import dataset_split
from models import *
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def test_model(model):
    """
        A function used to test specified models to get their best performing hyperparameters

        ------
        inputs:
            model: name of the model that will be tested

        ------
        returns: n/a
        """
    print(model)

    #https://www.geeksforgeeks.org/machine-learning/cross-validation-machine-learning/
    #https://www.geeksforgeeks.org/machine-learning/k-fold-cross-validation-in-machine-learning/

    # split dataset into 5 fold
    ff = StratifiedKFold(n_splits=5, shuffle=True, random_state=41) # use StratifiedKFold to avoid imbalanced class distribution
    # ff = KFold(n_splits=5, shuffle=True, random_state=41)
    # ff = KFold(n_splits=5, shuffle=False)

    # switch case to select model
    match model:
        case 'knn':
            knn_fine_tuning(ff)
        case 'dt':
            decision_tree_fine_tuning(ff)
        case 'mlp':
            mlp_fine_tuning(ff)
        case 'results':
            print("Showing comparison between models...")
        case _:
            print("Model not available.")

    # create summary graphs of hyperparameter comparison
    model_results = ['data_exports/dt_gridsearch_rs.csv', 'data_exports/mlp_gridsearch_rs.csv']
    # model_results = ['mlp_gridsearch_rs.csv']

    # plot graphs to compare performance of top 50 combinations
    # x = mean, y = std
    # colours for each hidden layer size, activation, learning rate etc
    # can be done by reading the csv file
    # include default hyperparam settings for baseline comparison

    # look at box plots to compare mean, median and std of accuracy scores across multiple configurations

    # parallel coordinates plot? - https://plotly.com/python/parallel-coordinates-plot/

    for results in model_results:
        results_df = pd.read_csv(results)
        print(results_df.head())
        print(results_df.columns)

        x_values = results_df['mean_test_score'].to_numpy()
        y_values = results_df['std_test_score'].to_numpy()

        plt.scatter(x_values, y_values)

    plt.show()

    import seaborn as sns
    colours = ["flare", "crest"]
    for results in range(len(model_results)):
        results_df = pd.read_csv(model_results[results])

        sns.scatterplot(
            data=results_df,
            x='mean_test_score',
            y='std_test_score',
            hue='rank_test_score',
            palette=sns.color_palette(colours[results], as_cmap=True)
        )

    plt.title("Test")
    plt.show()






def knn_fine_tuning(ff):
    """
        A function used to test k Nearest Neighbours using 5 fold cross validation to
        find its best hyperparameters

        ------
        inputs:
            five_fold: provides the train/test indices to split into 5 train/test sets

        ------
        returns: N/A
    """

    # get dataset
    training_set, test_set = dataset_split()

    # split training set
    x_train = training_set.drop(['Encoded_sign'], axis=1).to_numpy()
    y_train = training_set['Encoded_sign'].to_numpy()

    # base knn model
    knn = KNeighborsClassifier()

    # baseline CV score (before tuning)
    cv_accuracy_scores = cross_val_score(knn, x_train, y_train, cv=ff)
    print(f"CV accuracy scores: {cv_accuracy_scores}")
    print(f"Mean CV accuracy: {cv_accuracy_scores.mean()}")
    print(f"Standard deviation: {cv_accuracy_scores.std()}")
    
    # parameter grid
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # grid search
    knn_gridsearch = GridSearchCV(
        knn,
        param_grid,
        cv=ff,
        scoring='accuracy',
        refit=True,
        n_jobs=-1,
        verbose=2
    )

    knn_gridsearch.fit(x_train, y_train)

    print(f"Best kNN params: {knn_gridsearch.best_params_}")
    print(f"Best kNN Accuracy Score: {knn_gridsearch.best_score_}")

    # export results
    results_df = pd.DataFrame(knn_gridsearch.cv_results_)
    results_df = results_df[
        ['param_n_neighbors', 'param_weights', 'param_metric',
         'mean_test_score', 'std_test_score', 'rank_test_score']
    ].sort_values(by='rank_test_score')

    results_df.to_csv('data_exports/knn_gridsearch_rs.csv', index=False)

def mlp_fine_tuning(five_fold):
    """
        A function used to test MLPClassifiers using 5 fold cross validation to
        find its best hyperparameters

        ------
        inputs:
            five_fold: provides the train/test indices to split into 5 train/test sets

        ------
        returns: n/a
        """

    # get training and test datasets
    training_set, test_set = dataset_split()

    # split training set into x (landmarks) and y (labels)
    x_train = training_set.drop(['Encoded_sign'], axis=1).to_numpy()  # axis = 0  -> operate along rows, axis = 1  -> operate along columns
    y_train = training_set['Encoded_sign'].to_numpy()

    # create new mlp
    mlp = MLPClassifier(max_iter=1000, random_state=41)

    # train and evaluate mlp using 5-fold cross validation
    cv_accuracy_scores = cross_val_score(mlp, x_train, y_train, cv=five_fold)
    mean_cv = cv_accuracy_scores.mean()
    std_cv = cv_accuracy_scores.std()
    print(f"CV accuracy scores: {cv_accuracy_scores}")
    print(f"Mean CV accuracy: {mean_cv}")
    print(f"Standard deviation: {std_cv}")

    # --> look at using gridsearch https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    # https://drbeane.github.io/python_dsci/pages/grid_search.html
    print(f"MLP Params : {mlp.get_params()}")

    param_grid = [{
        'hidden_layer_sizes': [(64,32), (32,32), (75,50), (48,16)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'], # default = relu
        'learning_rate': ['constant', 'invscaling', 'adaptive'], # default = constant
        'solver': ['sgd', 'adam'] # default = adam
        # 'alpha': [0.0001, 0.001, 0.005, 0.0005] # default = 0.0001
    }]

    # mlp_gridsearch = GridSearchCV(mlp, param_grid, cv=five_fold, scoring='accuracy', refit=True) #param grid = list of dict of parameter settins to try as values

    mlp_gridsearch = GridSearchCV(mlp, param_grid, cv=five_fold, scoring='accuracy', refit=True, n_jobs=-1, verbose=2)
    # n_jobs = -1 : run on all available cores
    # verbose = 2 : gives update each time fold finishes

    mlp_gridsearch.fit(x_train, y_train)

    print(f"Best MLP params: {mlp_gridsearch.best_params_}")
    # print(mlp_gridsearch.scoring)
    print(f"Best MLP Object: {mlp_gridsearch.best_estimator_}")
    print(f"Best MLP Accuracy Score: {mlp_gridsearch.best_score_}")
    # print(mlp_gridsearch.cv_results_)

    cv_res = mlp_gridsearch.cv_results_
    print(cv_res.keys()) # dict_keys(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'param_activation', 'param_hidden_layer_sizes', 'param_learning_rate', 'param_solver', 'params', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'mean_test_score', 'std_test_score', 'rank_test_score'])


    results_df = pd.DataFrame(mlp_gridsearch.cv_results_)
    # # params = params used, mean_test_score = avg score over 5 folds, std_test_score =
    # results_df = results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
    results_df = results_df[['param_activation', 'param_hidden_layer_sizes', 'param_learning_rate', 'param_solver', 'mean_test_score', 'std_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
    print(results_df.head())

    results_df.to_csv('data_exports/mlp_gridsearch_rs.csv', index=False)


def decision_tree_fine_tuning(ff):
    """
        A function used to test DecisionTreeClassifiers using 5 fold cross validation to
        find its best hyperparameters

        ------
        inputs:
            five_fold: provides the train/test indices to split into 5 train/test sets

        ------
        returns: n/a
        """
    # https://www.geeksforgeeks.org/machine-learning/building-and-implementing-decision-tree-classifiers-with-scikit-learn-a-comprehensive-guide/

    #get appropriate data
    training_set, test_set = dataset_split()

    # split training set into x (landmarks) and y (labels)
    x_train = training_set.drop(['Encoded_sign'], axis=1).to_numpy()  # axis = 0  -> operate along rows, axis = 1  -> operate along columns
    y_train = training_set['Encoded_sign'].to_numpy()
    
    #create decision tree
    dtree = tree.DecisionTreeClassifier(random_state=7107)
    
    #evaluated method using 5-fold cv
    cv_accuracy_scores = cross_val_score(dtree, x_train, y_train, cv=ff)
    mean_cv = cv_accuracy_scores.mean()
    std_cv = cv_accuracy_scores.std()
    print(f"CV accuracy scores: {cv_accuracy_scores}")
    print(f"Mean CV accuracy: {mean_cv}")
    print(f"Standard deviation: {std_cv}")

    print(f"dtree Params : {dtree.get_params()}")

    param_grid = [{
        'max_depth': [5, 10, 20, None], # controls max depth to which tree can grow to, default = None
        'min_samples_leaf': range(1, 10, 2), # minimum number of samples required to be at a leaf node
        'min_samples_split': range(2, 10, 2), # minimal number of samples that are needed to split a node
        'criterion': ["entropy", "gini"] # quality of the split in the decision tree, default = gini
    }]

    dt_gridsearch = GridSearchCV(dtree, param_grid, cv=ff, scoring='accuracy', refit=True, n_jobs=-1, verbose=2)
    # n_jobs = -1 : run on all available cores
    # verbose = 2 : gives update each time fold finishes

    dt_gridsearch.fit(x_train, y_train)

    print(f"Best DT params: {dt_gridsearch.best_params_}")
    # print(dt_gridsearch.scoring)
    print(f"Best DT Object: {dt_gridsearch.best_estimator_}")
    print(f"Best DT Accuracy Score: {dt_gridsearch.best_score_}")
    # print(dt_gridsearch.cv_results_)


    cv_res = dt_gridsearch.cv_results_
    print(cv_res.keys())

    results_df = pd.DataFrame(dt_gridsearch.cv_results_)
    # params = params used, mean_test_score = avg score over 5 folds, std_test_score =
    # results_df = results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
    results_df = results_df[['param_criterion', 'param_max_depth', 'param_min_samples_leaf', 'param_min_samples_split', 'mean_test_score', 'std_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
    print(results_df.head())

    results_df.to_csv('data_exports/dt_gridsearch_rs.csv', index=False)


def compare_best_models():
    """
        A function used to compare each classifier model with their best performing settings.


        ------
        inputs: n/a

        ------
        returns: n/a
        """
    print("\nComparing Classifier Models...")
    # PART 2C : 3,4
    # - retrain best models (best kNN, DT and MLP) on entire training set
    # then compare each one against each other
    # return the best model with their parameters

    # get best params from csv files - first line as ranked
    # knn?
    knn_results = pd.read_csv("data_exports/knn_gridsearch_rs.csv")
    mlp_results = pd.read_csv("data_exports/mlp_gridsearch_rs.csv")
    dt_results = pd.read_csv("data_exports/dt_gridsearch_rs.csv")

    # get list of kNN params
    knn_optimal_params = knn_results[['param_n_neighbors', 'param_weights', 'param_metric']].iloc[0]

    knn_params = knn_optimal_params.to_list()
    knn_params[0] = int(knn_params[0])  # ensure k is int

    print(f"kNN Params: {knn_params}")

    # get list of MLP params
    mlp_optimal_params = mlp_results[['param_activation', 'param_hidden_layer_sizes', 'param_learning_rate', 'param_solver']].iloc[0]
    # print(f"MLP Best Params: {mlp_optimal_params}")
    mlp_params = mlp_optimal_params.to_list()
    import ast # https://www.geeksforgeeks.org/python/difference-between-eval-and-ast-literal-eval-in-python/
    mlp_params[1] = ast.literal_eval(mlp_params[1]) # convert hidden_layer_sizes from string back into tuple
    print(f"MLP Params: {mlp_params}")

    # get list of DT params
    dt_optimal_params = dt_results[['param_criterion', 'param_max_depth', 'param_min_samples_leaf', 'param_min_samples_split']].iloc[0]
    # print(f"DT Best Params: {dt_optimal_params}")
    dt_params = dt_optimal_params.to_list()
    dt_params[1] = int(dt_params[1]) # convert max depth to int
    print(f"DT Params: {dt_params}")

    # get training and test datasets
    training_set, test_set = dataset_split()

    # split training set into x (landmarks) and y (labels)
    x_train = training_set.drop(['Encoded_sign'], axis=1).to_numpy()
    y_train = training_set['Encoded_sign'].to_numpy() # labels
    x_test = test_set.drop(['Encoded_sign'], axis=1).to_numpy()
    y_test = test_set['Encoded_sign'].to_numpy() # labels

    # get kNN object
    #...

    # train models on whole training set
    mlp = multilayer_perceptron(x_train, y_train, mlp_params)
    dt = decision_tree_create(x_train, y_train, 7107, dt_params)

    # get model predictions from test set
    knn_y_pred = kNN_predict_batch(x_test, knn_params[0])
    mlp_y_pred = mlp_predict(mlp, x_test)
    dt_y_pred = decision_tree_decision(dt, x_test)

    # get model accuracies
    knn_accuracy = metrics.accuracy_score(y_test, knn_y_pred)
    mlp_accuracy = metrics.accuracy_score(y_test, mlp_y_pred)
    dt_accuracy = metrics.accuracy_score(y_test, dt_y_pred)
    print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")
    print(f"MLP Accuracy: {mlp_accuracy * 100:.2f}%")
    print(f"DT Accuracy: {dt_accuracy * 100:.2f}%")

    # get model precision
    knn_precision = metrics.precision_score(y_test, knn_y_pred, average="weighted")
    mlp_precision = metrics.precision_score(y_test, mlp_y_pred, average="weighted")
    dt_precision = metrics.precision_score(y_test, dt_y_pred, average="weighted")
    print(f"KNN Precision: {knn_precision * 100:.2f}%")
    print(f"MLP Precision: {mlp_precision * 100:.2f}%")
    print(f"DT Precision: {dt_precision * 100:.2f}%")

    # get model recall
    knn_recall = metrics.recall_score(y_test, knn_y_pred, average="weighted")
    mlp_recall = metrics.recall_score(y_test, mlp_y_pred, average="weighted")
    dt_recall = metrics.recall_score(y_test, dt_y_pred, average="weighted")
    print(f"KNN Recall: {knn_recall * 100:.2f}%")
    print(f"MLP Recall: {mlp_recall * 100:.2f}%")
    print(f"DT Recall: {dt_recall * 100:.2f}%")

    # get model f1 score
    knn_f1 = metrics.f1_score(y_test, knn_y_pred, average="weighted")
    mlp_f1 = metrics.f1_score(y_test, mlp_y_pred, average="weighted")
    dt_f1 = metrics.f1_score(y_test, dt_y_pred, average="weighted")
    print(f"KNN F1 score: {knn_f1 * 100:.2f}%")
    print(f"MLP F1 score: {mlp_f1 * 100:.2f}%")
    print(f"DT F1 score: {dt_f1 * 100:.2f}%")

    # https://scikit-learn.org/stable/api/sklearn.metrics.html
    # Metrics to calculate and compare against:
    # - Accuracy
    # - Precision
    # - Recall (sensitivity)
    # - F1 score
    # - Confusion Matrix
    # - ROC Curve?
    # - Absolute Mean Error?



def test_harness():
    print("TO DO IN THIS SCRIPT:"
          "\n - test each model to fine tune two hyperparameters"
          "\n - use 5-fold cross validation to do this"
          "\n - assess each model/test against performance metrics"
          "\n - then retrain each model with their best performing metrics with the main training set"
          "\n - compare this to others for classifier performance comparison"
          "\n - create visuals of performance metrics etc"
          "\n\n Use this script to test each classifier to work out their best hyperparameters, then use the ones in "
          "models to create a fine-tuned classifier")
          # "\n OR"
          # "\n Adapt model functions to first do 5fold tests, then use .fit to train on actual datasets etc like in labsheet ")

    # next to do:
    # - kNN hyperparam tests
    # - create graphs to compare settings
    # - retrain best models (best kNN, DT and MLP) on on entire training set
    # - then compare each classifier

    # test_model("knn")
    # test_model("mlp")
    # test_model("dt")
    # test_model("results")


    compare_best_models()


if __name__ == '__main__':
    test_harness()