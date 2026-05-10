import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model, feature_names):

    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })

    feature_df = feature_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(feature_df['Feature'], feature_df['Importance'])
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')

    plt.savefig('outputs/graphs/feature_importance.png')
    plt.close()