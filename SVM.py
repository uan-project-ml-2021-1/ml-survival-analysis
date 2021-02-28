import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Location of dataset
url = "https://raw.githubusercontent.com/uan-project-ml-2021-1/documentation/master/S1DataPreprocessed.csv"

# Assign colum names to the dataset
headers = ['TIME', 'BP', 'Anaemia', 'Age', 'EF', 'Creatinine', 'Death']

# Read dataset to pandas dataframe
clinicdata = pd.read_csv(url, names=headers, header=0, delimiter=';')

print(clinicdata.head())

# assign data from first five columns to X variable (Independents)
X = clinicdata.iloc[:, 0:6]

clinicdata['Death'] = clinicdata['Death'].replace([1.0, 0], ['y', 'n'])
y = clinicdata.select_dtypes(include=[object])

# preprocessing

le = preprocessing.LabelEncoder()

y = y.apply(le.fit_transform)
# end fitting data

# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# adjust on Scalar values
scalar = StandardScaler()
scalar.fit(X_train)

X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

# Classifier
# 2 layers of 2 nodes each
# MultiLayer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(2, 2), random_state=17, max_iter=1500)
# On train
mlp.fit(X_train, y_train.values.ravel())

predictions = mlp.predict(X_test)

# evaluating the algorithm

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, zero_division="warn"))
