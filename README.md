## **Dataset & Features:**

This report details the model's architecture, training, evaluation, and insights gained from applying it to healthcare insurance data in the "insurance.csv" file. 

Here's the sample of the dataset:
![image](https://github.com/laurencia7/IntroToML/assets/91892470/dbb9d215-8091-4a3e-930f-55abad4ab3d4)

### About Dataset
```
- **age:** Age of the policyholder.
- **sex:** Gender of the policyholder (female=0, male=1).
- **bmi:** Body mass index (BMI), providing an understanding of body weight relative to height. It is an objective index of body weight calculated as weight in kilograms divided by the square of height in meters (kg / m^2). Ideally, the BMI falls within the range of 18.5 to 25.
- **steps:** Average number of walking steps per day taken by the policyholder.
- **children:** Number of children or dependents of the policyholder.
- **smoker:** Smoking status of the policyholder (non-smoker=0; smoker=1).
- **region:** The residential area of the policyholder in the United States, categorized as follows: 
    - Northeast=0
    - Northwest=1
    - Southeast=2
    - Southwest=3
- **charges:** Individual medical costs billed by health insurance.
- **insuranceclaim:** Binary variable indicating whether an insurance claim has been made, with "yes" represented as 1 and "no" as 0.
```
The dataset has been divided into three sets: training, validation, and testing, with an 80% training, 10% validation, and 10% testing distribution. The data split is as follows:
```
- Training Set: 1070 data records
- Validation Set: 134 data records
- Test Set: 134 data records
```
The dataset consists of 9 variables with 1338 data records in two data types: integer and float. Notably, no null values were identified in the dataset.

### Preprocessing Steps

1. **Categorical Encoding**: Categorical variables in the dataset have been converted to numerical data types, enabling their utilization in the model. Consequently, there is no requirement for categorical encoding.

2. **Outlier Handling**:
    
    ![image](https://github.com/laurencia7/IntroToML/assets/91892470/8d2845ed-0dc9-4e65-b40c-99129acfae1e)
    
    ![image](https://github.com/laurencia7/IntroToML/assets/91892470/c471747c-0a6a-4572-9f72-e7b2bf1030a0)
    
    Examination of boxplots revealed outliers in the 'bmi' (9 outliers) and 'charges' (193 outliers) variables. To address this, a detailed visualization of the boxplots for these two variables was conducted.
    It was decided to replace outlier values with the median value rather than removing outlier data. This approach was chosen to preserve valuable information and maintain a better representation of the data. Additionally, replacing outliers with the median helps mitigate their impact on the mean and standard deviation of the data, which can otherwise result in an unrepresentative mean.

5. **Data Scaling (Normalization)**:

   ![image](https://github.com/laurencia7/IntroToML/assets/91892470/d105e21f-e930-4e78-bd98-dd9aefe7ad80)


    As seen in descriptive statistics for each variable above is in different scales in the dataset. 
    To ensure that the numerical variables with different scales in the dataset, such as age, bmi, steps, children, and charges, do not influence the model's performance, data scaling was performed. Normalization (min-max scaling) was chosen as the data scaling technique. This transformation ensures that all features have a uniform value range between 0 and 1, effectively addressing the potential sensitivity of neural networks to input variable scale differences.
    Normalization is particularly crucial for neural networks, as failing to scale input variables can lead to suboptimal model performance. It ensures that the model can learn effectively from features with varying scales without bias.

## **Method:**

The model used is a neural network model built using TensorFlow/Keras. It appears to be a feedforward neural network with the following layers:

    Input Layer: The input layer has 8 units, indicating that the data has 8 input features.

    Hidden Layer 1: This layer has 16 units (neurons) with the ReLU (Rectified Linear Unit) activation function. ReLU is commonly used for hidden layers in neural networks.

    Hidden Layer 2: Similar to the first hidden layer, this layer also has 16 units with ReLU activation.

    Output Layer: The output layer consists of 2 units with the softmax activation function. This suggests that the model is designed for classification tasks with two classes. The softmax activation function is often used in multi-class classification problems to produce probability distributions over the classes.

The model is compiled using the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01 and momentum of 0.5 and will be tuned using Grid Search. For classification tasks like this one, 'categorical_crossentropy' is a common choice for the loss function, and 'accuracy' is used as a metric to measure model performance.

## **Experiment:**

Hyperparameter Tuning GridSearch

I employed the grid search method to discover optimal hyperparameter values for the machine learning model.

The reason behind my choice of approach is that grid search systematically evaluates various combinations of hyperparameter values and selects the combination that yields the best model performance. This helps conserve time and resources, as manually searching for optimal hyperparameter values can be time-consuming.

Furthermore, the grid search technique is straightforward to implement, especially for machine learning models that are not overly complex.

```
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
```
![image](https://github.com/laurencia7/IntroToML/assets/91892470/e38fa25c-126a-4428-905a-935551d05ad6)



From the results above, it was found that the best parameters that produced optimal accuracy are:

    'epochs': 20,
    'optimizer': 'adam' with 'learning rate': 0.01,
    Addition of 1 hidden layer.

The addition of Dropout layers and L2 Regularization: L2 Regularization turned out to be not necessary as it did not contribute anything to the model's loss (best parameters for dropout_rate and weight_decay are both 0).

    'dropout_rate': 0.0,
    'weight_decay': 0.0.

## **Result:**

![image](https://github.com/laurencia7/IntroToML/assets/91892470/1567f1ed-09d7-4c73-ae4a-ebdc9a4db1b3)

The plotted heatmap illustrates the correlations among variables. Negative values on the heatmap indicate a negative correlation between two variables. This implies that as one variable's value increases, the value of the other variable tends to decrease. The larger the negative value, the stronger the negative correlation between the two variables.

Conversely, positive values on the heatmap signify a positive correlation between the two variables, where an increase in the value of one variable is associated with an increase in the value of the other variable. The higher the positive value, the stronger the positive correlation between the two variables. A value of 0 on the heatmap indicates no correlation between the two variables.

It's noticeable that within the insurance.csv dataset, there are correlations between variables, both positive and negative in nature.

Accuracy after tuning plot

![image](https://github.com/laurencia7/IntroToML/assets/91892470/a21d35bf-2492-4fda-ba4c-eaea43b60172)

_Before tuning_
```
Accuracy: 0.746268656716418
Precision: 0.7674418604651163
Recall: 0.825
F1-score: 0.7951807228915662
```

_After Tuning:_

```
Accuracy: 0.8805970149253731
Precision: 0.9102564102564102
Recall: 0.8875
F1-score: 0.8987341772151898
```

It is evident that the accuracy, precision, recall, and F1-score outcomes after tuning are higher compared to those before tuning. This indicates that the optimized model (achieved by seeking the best hyperparameters through Grid Search) performs better than the model prior to tuning.

This enhancement in performance can be interpreted as the optimized model being capable of classifying with greater accuracy and precision on the same dataset as compared to the model before tuning. Consequently, hyperparameter tuning can aid in elevating the model's quality and maximizing its performance on the provided data.

