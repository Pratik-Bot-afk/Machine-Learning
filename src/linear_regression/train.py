from sklearn.linear_model import LinearRegression
import pickle
from utils import load_data, split_data

file_path = 'data/processed/clean_housing_data.csv'
target = 'price'
model_path = "models/linear-regression/model.pkl"


def train_model():
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(data, target)
   
    model = LinearRegression()

    model.fit(X_train, y_train)

    with open(model_path,"wb") as f:
        pickle.dump(model,f)

    print("Model trained and saved!")    

if __name__ == "__main__":
    train_model()        


