import pickle

model = pickle.load(open("models/saved/ids_model.pkl", "rb"))

def predict(data):
    result = model.predict([data])
    return "Attack" if result[0] == 1 else "Normal"