from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib


def train_models(X_train, y_train):

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }

    trained_models = {}

    for name, model in models.items():

        # Train model
        model.fit(X_train, y_train)

        # Save model
        file_name = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, f'models/{file_name}')

        trained_models[name] = model

    return trained_models