from Decision_Tree import decision_tree as dt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def accuracy(prediction, actual):
    """
    Simple function to get accuracy score
    """
    correct_count = 0
    prediction_len = len(prediction) # prediction length
    for index in range(prediction_len): # for # in prediction length
        if int(prediction[index]) == actual[index]: # if the indexth prediction was correct
            correct_count += 1 # add  1 to th counter
    return correct_count/prediction_len # return raw accuracy, which is correct predictions divided by # of predictions
def main():
    """
    Main function that actually does all of the work to test and score my model, and its sklearn counterpart for comparison
    """
    # Loading/organizing Data
    wine_data = load_wine() # load dataset from sklearn for testing
    x = wine_data.data[:,:2] # isolate features
    y = wine_data.target # isolate target
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state= 42) # split data for train/test set
    
    # Our decision tree
    decision_tree_model =  dt.DecisionTree(_max_depth = 2, _min_splits = 30)
    decision_tree_model.fit(X_train, y_train)
    prediction  = decision_tree_model.predict(X_test)

    # Decision tree from sk learn
    sk_dt_model = DecisionTreeClassifier(max_depth= 2, min_samples_split= 30)
    sk_dt_model.fit(X_train, y_train)
    sk_dt_prediction = sk_dt_model.predict(X_test)
    print(prediction)
    # Printing results
    print("My Model's Accuracy: {0}".format(accuracy(prediction, y_test)))
    print("Sklearn Accuracy: {0}".format(accuracy(sk_dt_prediction, y_test)))


if  __name__ == "__main__":
    main()

