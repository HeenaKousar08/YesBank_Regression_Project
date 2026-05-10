from src.preprocessing import load_and_preprocess_data
from src.eda import perform_eda
from src.train_models import train_models
from src.evaluate import evaluate_models
from src.feature_importance import plot_feature_importance
from src.utils import create_folders


# Create folders
create_folders()

# Load dataset
X_train, X_test, y_train, y_test, df = load_and_preprocess_data(
    'data/data_YesBank_StockPrices.csv'
)

# Perform EDA
perform_eda(df)

# Train models
models = train_models(X_train, y_train)

# Evaluate models
results_df = evaluate_models(models, X_test, y_test)

# Feature importance using Random Forest
rf_model = models['Random Forest']

plot_feature_importance(
    rf_model,
    ['Open', 'High', 'Low']
)