import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_data, split_data
import json

data_path = 'data/processed/clean_housing_data.csv'
model_path = 'models/linear-regression/model.pkl'

def evaluate():
    data = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(data,"price")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("MSE: ", mse)
    print("R2 score: ", r2)

    metrics={
        "MSE": float(mse),
        "R2 Score": float(r2)
    }
    with open("results/metrics/linear_regression.json", 'w') as f:
        json.dump(metrics, f, indent = 4)

    #plot
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")

    plt.savefig("results/plots/linear_regression.png")
    print("Figure saved")
    plt.show()

if __name__ == "__main__":
    evaluate()  