def save_dict(obj, path):
    import pickle
    with open(path, 'wb') as fout:
        pickle.dump(obj, fout)

def load_dict(path):
    import pickle
    with open(path, 'rb') as fout:
        return pickle.load(fout)