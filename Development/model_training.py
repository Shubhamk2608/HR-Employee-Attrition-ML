from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_models(X_train,y_train):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000,
    random_state=42),

        "Decision Tree": DecisionTreeClassifier(),

        "Random Forest": RandomForestClassifier(),

        "SVM": SVC()
    }



    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models