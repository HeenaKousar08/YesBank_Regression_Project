import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(file_path):

    # Read dataset
    df = pd.read_csv(file_path, sep='	')

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove missing values
    df.dropna(inplace=True)

    # Features and target
    X = df[['Open', 'High', 'Low']]
    y = df['Close']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, df