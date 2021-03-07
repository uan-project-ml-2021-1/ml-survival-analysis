import pandas


# load the dataset

def load_dataset(url: str, header_idx: int):
    dataset = pandas.read_csv(url, header=header_idx, delimiter=',')
    # array of data
    data = dataset.values
    # separation into input and output columns
    X = data[:, :-1].astype(str)
    y = data[:, -1].astype(str)

    # summarize
    print('Input', X.shape)
    print('Output', y.shape)
