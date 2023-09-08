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

About Dataset
age : age of policyholder
sex: gender of policy holder (female=0, male=1)
bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 25
steps: average walking steps per day of policyholder
children: number of children / dependents of policyholder
smoker: smoking state of policyholder (non-smoke=0;smoker=1)
region: the residential area of policyholder in the US (northeast=0, northwest=1, southeast=2, southwest=3)
charges: individual medical costs billed by health insurance
insuranceclaim: yes=1, no=0

**Dataset & Features:**

In this section, we will describe the dataset and the preprocessing steps undertaken for our healthcare insurance claim prediction model.

**Dataset Description:**

The dataset has been divided into three sets: training, validation, and testing, with an 80% training, 10% validation, and 10% testing distribution. The data split is as follows:

- Training Set: 1070 data records
- Validation Set: 134 data records
- Test Set: 134 data records

The dataset consists of 9 variables with 1338 data records in two data types: integer and float. Notably, no null values were identified in the dataset.

**Preprocessing Steps:**

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

The model is compiled using the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01 and momentum of 0.5. For classification tasks like this one, 'categorical_crossentropy' is a common choice for the loss function, and 'accuracy' is used as a metric to measure model performance.

