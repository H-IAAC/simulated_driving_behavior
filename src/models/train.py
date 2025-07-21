from sklearn.metrics import classification_report, accuracy_score


def grid_search(model, x_train, y_train, X_test, Y_test, param_grid, cv=3):
    from sklearn.model_selection import GridSearchCV

    # Initialize the model
    md = model()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=md, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=1)

    # Fit the model
    grid_search.fit(x_train, y_train)

    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    grid_search_predictions = grid_search.predict(X_test)

    print("Grid Search Classification Report:")
    print(classification_report(Y_test, grid_search_predictions))
    acc = accuracy_score(Y_test, grid_search_predictions)
    print(f"Grid Search Accuracy: {acc:.4f}")

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

    return best_params, best_score, classification_report(Y_test, grid_search_predictions, output_dict=True), acc


def train_model(model, x_train, y_train, X_test, Y_test, params=None):
    # Initialize the model
    md = model(**params) if params else model()

    # Fit the model
    md.fit(x_train, y_train)

    # Make predictions
    predictions = md.predict(X_test)

    print("Classification Report:")
    print(classification_report(Y_test, predictions))
    acc = accuracy_score(Y_test, predictions)
    print(f"Accuracy: {acc:.4f}")

    return classification_report(Y_test, predictions, output_dict=True), acc
