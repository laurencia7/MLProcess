**Introduction:**

In this report, I tackle a critical issue: creating a predictive model for determining whether a health insurance claim should be accepted or declined—a binary classification task.

Why it matters:
- Precise cost estimation: Accurate predictions help set premiums and coverage rates, maintaining financial equilibrium.
- Efficient resource allocation: Accurate predictions guide resource allocation, ensuring healthcare providers are prepared for demand.
- Improved patient outcomes: Insights from the model personalize treatment plans, enhancing patient well-being.

My approach:
- I've developed a neural network using TensorFlow/Keras.
- The model has an input layer, two hidden layers with ReLU activations, and an output layer with softmax activation.
- It's optimized with SGD (learning rate: 0.01, momentum: 0.5), using 'categorical_crossentropy' loss and 'accuracy' metric.

The aim: Equip the company with a tool to make informed insurance claim decisions, ensuring client well-being and the sustainability of our services.


**Dataset & Features:**

This report details the model's architecture, training, evaluation, and insights gained from applying it to healthcare insurance data in the "insurance.csv" file. 

Here's the sample of the dataset:
![image](https://github.com/laurencia7/IntroToML/assets/91892470/dbb9d215-8091-4a3e-930f-55abad4ab3d4)
```
_About Dataset_

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

_Preprocessing Steps_

1. **Categorical Encoding**: Categorical variables in the dataset have been converted to numerical data types, enabling their utilization in the model. Consequently, there is no requirement for categorical encoding.

2. **Outlier Handling**:
    
    ![image](https://github.com/laurencia7/IntroToML/assets/91892470/8d2845ed-0dc9-4e65-b40c-99129acfae1e)
    
    ![image](https://github.com/laurencia7/IntroToML/assets/91892470/c471747c-0a6a-4572-9f72-e7b2bf1030a0)
    
    Examination of boxplots revealed outliers in the 'bmi' (9 outliers) and 'charges' (193 outliers) variables. To address this, a detailed visualization of the boxplots for these two variables was conducted.
    It was decided to replace outlier values with the median value rather than removing outlier data. This approach was chosen to preserve valuable information and maintain a better representation of the data. Additionally, replacing outliers with the median helps mitigate their impact on the mean and standard deviation of the data, which can otherwise result in an unrepresentative mean.

5. **Data Scaling (Normalization)**: To ensure that the numerical variables with different scales in the dataset, such as age, bmi, steps, children, and charges, do not influence the model's performance, data scaling was performed. Normalization (min-max scaling) was chosen as the data scaling technique. This transformation ensures that all features have a uniform value range between 0 and 1, effectively addressing the potential sensitivity of neural networks to input variable scale differences.
Normalization is particularly crucial for neural networks, as failing to scale input variables can lead to suboptimal model performance. It ensures that the model can learn effectively from features with varying scales without bias.

**EDA:**

![Untitled](https://github.com/laurencia7/IntroToML/assets/91892470/5f05ded9-81be-4cdb-ad3f-8dfd6a8b4673)

In the plot above, it is evident that the distribution of policyholders is highest around a BMI of approximately 30.

![Untitled](https://github.com/laurencia7/IntroToML/assets/91892470/a26afca2-5779-4c9d-b1e5-a442278bb9b3)

From the plot above, it can be observed that the majority of policyholders are in their twenties.

```
Category  Count Representation
    Male    676              1
  Female    662              0
```

From the table above, it can be observed that the gender distribution of insurance policyholders is almost balanced between females and males. The number of males is slightly higher than females, but the difference is relatively small.

![image](https://github.com/laurencia7/IntroToML/assets/91892470/447f2f51-f071-47cd-a9ed-54c2c15febb7)

From the bar plot above, it is evident that according to the dataset from insurance.csv, the majority of insurance policyholders reside in the southeastern region of the USA.

![Untitled](https://github.com/laurencia7/IntroToML/assets/91892470/9f304c26-631c-42a3-8c46-d3aaa9d49a6c)

From the above pie plot, it is evident that out of 1338 individuals, 783 insurance policyholders had their claims accepted, while 555 insurance policyholders had their claims denied.

![Untitled](https://github.com/laurencia7/IntroToML/assets/91892470/6658dae8-536f-47d8-bdb3-2c418faa0cd9)

From the scatter plot above, it can be observed that policyholders who smoke incur a wide range of insurance charges. However, it is evident that the minimum insurance charge for policyholders who smoke is approximately $10,000. In contrast, for non-smokers, the minimum insurance charge is around $1,000.

**Method:**

The model used is a neural network model built using TensorFlow/Keras. It appears to be a feedforward neural network with the following layers:

    Input Layer: The input layer has 8 units, indicating that the data has 8 input features.

    Hidden Layer 1: This layer has 16 units (neurons) with the ReLU (Rectified Linear Unit) activation function. ReLU is commonly used for hidden layers in neural networks.

    Hidden Layer 2: Similar to the first hidden layer, this layer also has 16 units with ReLU activation.

    Output Layer: The output layer consists of 2 units with the softmax activation function. This suggests that the model is designed for classification tasks with two classes. The softmax activation function is often used in multi-class classification problems to produce probability distributions over the classes.

The model is compiled using the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01 and momentum of 0.5 and will be tuned using Grid Search. For classification tasks like this one, 'categorical_crossentropy' is a common choice for the loss function, and 'accuracy' is used as a metric to measure model performance.

**Experiment:**

Hyperparameter Tuning GridSearch

I employed the grid search method to discover optimal hyperparameter values for the machine learning model.

The reason behind my choice of approach is that grid search systematically evaluates various combinations of hyperparameter values and selects the combination that yields the best model performance. This helps conserve time and resources, as manually searching for optimal hyperparameter values can be time-consuming.

Furthermore, the grid search technique is straightforward to implement, especially for machine learning models that are not overly complex.

To improve the model's performance, several approaches can be undertaken, such as:

1. Adding or reducing layers in the model.
2. Modifying the number of neurons in each layer.
3. Adjusting the learning rate in the optimizer.
4. Employing regularization techniques like dropout or L2 regularization.
5. Choosing a more appropriate evaluation metric.

Here are some changes that can be made to the baseline architecture to enhance its performance:

1. Adding Dropout layer: A Dropout layer can aid in preventing overfitting by randomly deactivating neurons during training. This helps the model learn more generalized features from the data.

2. Adding Dense layer: Incorporating a Dense layer can help the model grasp more intricate features from the data.

3. Switching to the Adam optimizer: The Adam optimizer is a widely used optimizer in deep learning and often yields better results compared to the SGD optimizer.

4. Reducing the number of neurons in Dense layers: In some cases, an excessive number of neurons in Dense layers can lead to overfitting. Thus, reducing the number of neurons in Dense layers could help prevent overfitting.

5. Applying L2 Regularization: L2 Regularization is a technique that assists in preventing overfitting by adding a penalty to large weights.

Since we have a dataset with a relatively small number of features and a sufficient number of samples, adding layers and neurons might help the model grasp more complex features from the data. Meanwhile, dropout and L2 regularization could aid in preventing overfitting. Changing the optimizer to Adam might also improve the model's performance. However, these changes will be combined with grid search techniques to obtain optimal parameters.

![image](https://github.com/laurencia7/IntroToML/assets/91892470/e38fa25c-126a-4428-905a-935551d05ad6)



From the results above, it was found that the best parameters that produced optimal accuracy are:

    'epochs': 20,
    'optimizer': 'adam' with 'learning rate': 0.01,
    Addition of 1 hidden layer.

The addition of Dropout layers and L2 Regularization: L2 Regularization turned out to be not necessary as it did not contribute anything to the model's loss (best parameters for dropout_rate and weight_decay are both 0).

    'dropout_rate': 0.0,
    'weight_decay': 0.0.

**Result:**

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

**Conclusion:**

It is evident that the accuracy, precision, recall, and F1-score outcomes after tuning are higher compared to those before tuning. This indicates that the optimized model (achieved by seeking the best hyperparameters through Grid Search) performs better than the model prior to tuning.

This enhancement in performance can be interpreted as the optimized model being capable of classifying with greater accuracy and precision on the same dataset as compared to the model before tuning. Consequently, hyperparameter tuning can aid in elevating the model's quality and maximizing its performance on the provided data.

