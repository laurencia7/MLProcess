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

The dataset has been divided into three sets: training, validation, and testing, with an 80% training, 10% validation, and 10% testing distribution. The data split is as follows:

- Training Set: 1070 data records
- Validation Set: 134 data records
- Test Set: 134 data records

The dataset consists of 9 variables with 1338 data records in two data types: integer and float. Notably, no null values were identified in the dataset.

_Preprocessing Steps_

1. **Categorical Encoding**: Categorical variables in the dataset have been converted to numerical data types, enabling their utilization in the model. Consequently, there is no requirement for categorical encoding.

2. **Outlier Handling**: Examination of boxplots revealed outliers in the 'bmi' (9 outliers) and 'charges' (193 outliers) variables. To address this, a detailed visualization of the boxplots for these two variables was conducted.
It was decided to replace outlier values with the median value rather than removing outlier data. This approach was chosen to preserve valuable information and maintain a better representation of the data. Additionally, replacing outliers with the median helps mitigate their impact on the mean and standard deviation of the data, which can otherwise result in an unrepresentative mean.

3. **Data Scaling (Normalization)**: To ensure that the numerical variables with different scales in the dataset, such as age, bmi, steps, children, and charges, do not influence the model's performance, data scaling was performed. Normalization (min-max scaling) was chosen as the data scaling technique. This transformation ensures that all features have a uniform value range between 0 and 1, effectively addressing the potential sensitivity of neural networks to input variable scale differences.
Normalization is particularly crucial for neural networks, as failing to scale input variables can lead to suboptimal model performance. It ensures that the model can learn effectively from features with varying scales without bias.


**Method:**
The model used is a neural network model built using TensorFlow/Keras. It appears to be a feedforward neural network with the following layers:

    Input Layer: The input layer has 8 units, indicating that the data has 8 input features.

    Hidden Layer 1: This layer has 16 units (neurons) with the ReLU (Rectified Linear Unit) activation function. ReLU is commonly used for hidden layers in neural networks.

    Hidden Layer 2: Similar to the first hidden layer, this layer also has 16 units with ReLU activation.

    Output Layer: The output layer consists of 2 units with the softmax activation function. This suggests that the model is designed for classification tasks with two classes. The softmax activation function is often used in multi-class classification problems to produce probability distributions over the classes.

The model is compiled using the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01 and momentum of 0.5 and will be tuned using Grid Search. For classification tasks like this one, 'categorical_crossentropy' is a common choice for the loss function, and 'accuracy' is used as a metric to measure model performance.

