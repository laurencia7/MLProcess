# -*- coding: utf-8 -*-
"""ML Process: Insurance Claim.ipynb
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('insurance.csv')

"""# Data Preprocessing"""

df.info()

df.head()

"""### Problem data: *Outlier*"""

# The data frame contains several numerical variables.
df.boxplot(column=['bmi', 'charges', 'children', 'age', 'steps'])

plt.title('Box Plot dari 5 Variabel Numerik')

plt.show()

# Boxplot for bmi
sns.boxplot(x='bmi', data=df)
plt.title('Boxplot of BMI')
plt.show()

# Boxplot for charges
sns.boxplot(x='charges', data=df)
plt.title('Boxplot of Charges')
plt.show()

'''BMI'''
# calculating Q1 dan Q3
q1_bmi = df['bmi'].quantile(0.25)
q3_bmi = df['bmi'].quantile(0.75)

# calculating IQR
iqr_bmi = q3_bmi - q1_bmi

# Setting the upper and lower bounds.
upper_limit_bmi = q3_bmi + 1.5 * iqr_bmi
lower_limit_bmi = q1_bmi - 1.5 * iqr_bmi
print(f"Batas atas bmi : {upper_limit_bmi}")
print(f"batas bawah bmi : {lower_limit_bmi}\n")

# Checking for outliers in the 'bmi' column.
outliers_bmi = df[(df['bmi'] > upper_limit_bmi) | (df['bmi'] < lower_limit_bmi)]

# showing outlier
print(outliers_bmi)

'''charges'''
# calculating Q1 dan Q3
q1_charges = df['charges'].quantile(0.25)
q3_charges = df['charges'].quantile(0.75)

# calculating IQR
iqr_charges = q3_charges - q1_charges

# Setting the upper and lower bounds.
upper_limit_charges = q3_charges + 1.5 * iqr_charges
lower_limit_charges = q1_charges - 1.5 * iqr_charges
print(f"Batas atas charges : {upper_limit_charges}")
print(f"batas bawah charges : {lower_limit_charges}\n")

# Checking for outliers in the 'charges' column.
outliers_charges = df[(df['charges'] > upper_limit_charges) | (df['charges'] < lower_limit_charges)]

# showing outlier
print(outliers_charges)

# Menghitung median dari kolom bmi
bmi_median = df['bmi'].median()

# Mengganti data outliers pada kolom bmi
df.loc[df['bmi'] > upper_limit_bmi, 'bmi'] = bmi_median
df.loc[df['bmi'] < lower_limit_bmi, 'bmi'] = bmi_median

# Menghitung median dari kolom charges
charges_median = df['charges'].median()

# Mengganti data outliers pada kolom charges
df.loc[df['charges'] > upper_limit_charges, 'charges'] = charges_median
df.loc[df['charges'] < lower_limit_charges, 'charges'] = charges_median

"""================================================================================

# Data Exploration & Data Splitting

there are no null values present in any variable.
"""

print(df.isnull().sum())

"""After checking, there are no duplicate data entries."""

print(df.duplicated().sum())

print(df.describe())

"""### Data Visualization"""

import matplotlib.pyplot as plt
# Histogram for bmi
plt.hist(df['bmi'], bins=10)
plt.xlabel('BMI')
plt.ylabel('Count')
plt.title('Histogram of BMI')
plt.show()

# Histogram for age,
plt.hist(df['age'], bins=10)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Histogram of Age')
plt.show()

# Hitung jumlah male dan female
male_count = df['sex'].value_counts()[1]
female_count = df['sex'].value_counts()[0]

# Buat tabel visualisasi
data = {'Category': ['Male', 'Female'],
        'Count': [male_count, female_count],
        'Representation': ['1', '0']}
df_vis = pd.DataFrame(data)

# Ubah nama kolom agar sejajar saat di-print
df_vis = df_vis.rename(columns={'Category': 'Category', 'Count': 'Count', 'Representation': 'Representation'})

# Print tabel visualisasi
print(df_vis.to_string(index=False))

# Creating a bar plot from the "region" variable.
sns.countplot(x='region', data=df)

# Adding labels to the x and y axes.
plt.xlabel('Region')
plt.ylabel('Count')

# Adding labels to the x-axis according to the region codes.
plt.xticks(ticks=[0,1,2,3], labels=['Northeast', 'Northwest', 'Southeast', 'Southwest'])

plt.show()

# Calculating the Number of Records with Insurance Claims and Without tidak
yes_claim = len(df[df['insuranceclaim'] == 1])
no_claim = len(df[df['insuranceclaim'] == 0])

# pie chart
labels = ['Yes (' + str(yes_claim) + ')', 'No (' + str(no_claim) + ')']
sizes = [yes_claim, no_claim]
colors = ['green', 'red']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Insurance Claim')
plt.show()

# Scatter plot for charges dan age
plt.scatter(df['smoker'], df['charges'])
plt.xlabel('Smoker')
plt.ylabel('Charges')
plt.title('Scatter plot of Smoker and Charges')
plt.show()

# Heatmap for Visualizing Variable Correlations
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.title('Heatmap of Correlation Matrix')
plt.show()

"""**Scaling Data**"""

from sklearn.preprocessing import MinMaxScaler

# Selecting Columns for Scaling
cols_to_scale = ['age', 'bmi', 'steps', 'children', 'charges']

# Creating a Scaler Object
scaler = MinMaxScaler()

# Scaling Selected Columns
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

"""### Dataset Splitting"""

import keras

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# memisahkan feature dan target
X = df.drop('insuranceclaim', axis=1)
y = df['insuranceclaim']

# melakukan encoding one-hot pada label
y = to_categorical(y)

# memisahkan data menjadi train dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# memisahkan data test menjadi test dan validation set
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# menampilkan jumlah data pada masing-masing set
print("Jumlah data pada train set:", len(X_train))
print("Jumlah data pada validation set:", len(X_val))
print("Jumlah data pada test set:", len(X_test))

"""================================================================================"""

pip install tensorflow

pip install scikeras[tensorflow]

"""### Before Tuning"""

from tensorflow.keras.models import Sequential
from keras.regularizers import l2

import tensorflow as tf
model = keras.models.Sequential()

model.add(tf.keras.Input(shape=(8,)))

'''Sebelum tuning'''
model.add(keras.layers.core.Dense(16, activation='relu'))
model.add(keras.layers.core.Dense(16,   activation='relu'))
model.add(keras.layers.core.Dense(2,   activation='softmax'))
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum = 0.5), loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

history=model.fit(X_train, y_train, epochs=20, verbose=2)

# plot accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy Before Tuning')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes)
recall = recall_score(y_true, y_pred_classes)
f1score = f1_score(y_true, y_pred_classes)

print("Sebelum Tuning")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1score)

"""### Hyperparameter Tuning: GridSearch

"""

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import Dense, Dropout
from sklearn.metrics import make_scorer

def create_model(optimizer='sgd', learn_rate=0.01, momentum=0.5, dropout_rate=0.0, weight_decay=0.0):
    model = Sequential()
    model.add(Dense(16, input_dim=8, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(weight_decay)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(weight_decay)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='softmax'))

    if optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learn_rate, momentum=momentum)
    elif optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learn_rate)
    else:
        raise ValueError('Invalid optimizer')

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, verbose=2)

# Grid search parameters
batch_size = [16, 32]
epochs = [10, 20]
optimizer = ['sgd', 'adam']
learn_rate = [0.001, 0.01, 0.1]
momentum = [0.3, 0.5, 0.9]
dropout_rate = [0.0, 0.2]
weight_decay = [0.0, 0.001]

param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, learn_rate=learn_rate, momentum=momentum, dropout_rate=dropout_rate, weight_decay=weight_decay)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# summarize
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

"""### Architectural Modification"""

model = keras.models.Sequential()

model.add(tf.keras.Input(shape=(8,)))

'''After tuning'''
model.add(keras.layers.core.Dense(16, input_dim=8, activation='relu'))
model.add(keras.layers.core.Dense(16, activation='relu'))
model.add(keras.layers.core.Dense(16, activation='relu'))
model.add(keras.layers.core.Dense(2, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

history=model.fit(X_train, y_train, epochs=20, verbose=2)

# plot accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy After Tuning')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes)
recall = recall_score(y_true, y_pred_classes)
f1score = f1_score(y_true, y_pred_classes)

print("After Tuning")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1score)
