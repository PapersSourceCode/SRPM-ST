from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder


def get_dataset_id(name):
    if name == 'mnist':
        data_id = 554
    elif name == 'usps':
        data_id = 41082
    elif name == 'skin':
        data_id = 1502
    elif name == 'har':
        data_id = 1478
    elif name == 'churn':
        data_id = 40701
    elif name == 'texture':
        data_id = 40499
    elif name == 'wine':
        data_id = 40498
    elif name == 'kdd':
        data_id = 44035
    return data_id


def get_dataset(name):
    data_id = get_dataset_id(name)
    labelencoder = LabelEncoder()

    if name == 'kdd':
        dataset = fetch_openml(as_frame=True, data_id=data_id, parser='pandas')

        df = dataset.data
        labels = dataset.target

        for col_name in df.columns:
            if str(df[col_name].dtype) == 'category':
                df[col_name] = df[col_name].cat.codes

            data = df.to_numpy()

    else:
        dataset = fetch_openml(as_frame=False, data_id=data_id, parser="pandas")

        data = dataset.data
        labels = dataset.target

    labels = labelencoder.fit_transform(labels)

    return data, labels
