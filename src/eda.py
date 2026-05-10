import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(df):

    print(df.head())
    print(df.describe())

    # Heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(df[['Open', 'High', 'Low', 'Close']].corr(), annot=True)
    plt.title('Correlation Heatmap')
    plt.savefig('outputs/graphs/heatmap.png')
    plt.close()

    # Closing Price Trend
    plt.figure(figsize=(10, 5))
    plt.plot(df['Close'])
    plt.title('Closing Price Trend')
    plt.xlabel('Index')
    plt.ylabel('Close Price')
    plt.savefig('outputs/graphs/closing_price_trend.png')
    plt.close()