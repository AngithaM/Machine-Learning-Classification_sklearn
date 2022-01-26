# Machine Learning
The goal is to learn the basics of using a machine learning package for a classification task, and to prepare a short report demonstrating how you have applied it.
Details :
1. Research what open-source machine learning packages are available, and select one that you think will be appropriate for this task. Your report should include a brief justification for your choice and an overview of the main features of the package you have chosen. 

2. In the ML package, select two different classification algorithms that you will apply to the dataset to learn two different classification models. In the  report, include a clear description of both algorithms. 

3. In the ML package, train each of your chosen algorithms using the training set provided in wildfires_training.txt. You should then test your trained models using the test set provided in wildfires_test.txt. Report on the results, including the classification accuracy for each model on the training set and on the test set. 

4. Discuss in your report whether the two models give very similar or significantly different results, and why. 

The training dataset is in a file called wildfires_training.txt. and the test dataset is in a file called wildfires_test.txt. Columns are separated by tabs and rows are separated by newlines. Each row describes one instance in the dataset. The attributes are in columns in the following order: fire, year, temp, humidity, rainfall, drought_code, buildup_index, day, month, wind_speed.

The goal of your classifier is to predict fire (which may be one of two classes: yes or no) based on the other attributes.

### 1.
There are many open-source machine learning packages available. The following are the libraries considered for the assignment:
•	Numpy	•	Pandas	•	SciPy	•	Scikit-learn	•	Keras	•	TensorFlow
Of all the above library, Scikit-learn build on top of Numpy and Scipy is the one I personally feel is the most appropriate. This library includes a lot of learning algorithm, both supervised and unsupervised. It also takes into consideration ease of code, documentation and performance. It even uses c-libraries and cython to boost performance.
The primary focus of this library is data modelling; not loading, manipulating and summarizing data. Some of the few models available are Clustering, Dimensionality Reduction, Feature selection and extraction, and Supervised Models. All these are very useful in a classification task and hence this is the most appropriate model.

### 2.
The data set given is a txt file with values in each row separated by whitespaces. Since it is not single whitespace separated, we use train_data.split() which by default splits based on white spaces(regardless of the number of white spaces). The end result will be a comma separated list of lists of the training data. Then, all the values in the table are converted from string to float.For this particular data set, this is all the pre-processing required, but the null value handling was still considered:
•	Handling Null Values
Most of the data will contain some null values that the algorithms won’t be able to handle. We can easily find such rows and drop it or we can define our own customized method to change the null value. The approach will vary based on the data set.
train_data.isnull().sum() gave all values as zero indicating the absence of null values  

### 3.
### Decision Tree Classifier:
The algorithm for it is clearly described in the lecture slides and is as follows:
Step 1: Iterate through all the attributes and calculate the entropy or gini and find the note with the highest information gain. This node is selected as the root note.
Step 2: Split the node into child node (create subsets) based on this attribute and calculate entropy/gini index again.
Step 3: Recursively repeat these steps until no subsets can be created as there are no attributes to add or every element belongs to the same class or if all the elements do not belong to same class and we need to take a majority vote. 
### Logistic Regression

A simple logistic regression model can be used as the target variable(dependant) is categorical. Since the target is (yes/no), we can use a simple binary logistic regression model (when the categorical value has only two values).
Let’s say the attributes are names x₁, x₂..xn, and the target variable is Y. This model is used to find the hyperplane that separates the data into two classes based on the data points. The coefficient of this plane is found by minimizing log loss fun. 
The log loss function is as follows:

![image](https://user-images.githubusercontent.com/74540513/149666445-2fa4c656-e818-43cf-a50d-a9c08fc9f55b.png)

The function will be passed the vector X for all the attributes (as part of Pi), Y for the target value and Wt for the co-efficient of the hyperplane. The first possible value of Wt will be picked almost randomly and then it’ll be optimized using the gradient dissent. The objective of the algorithm is to find the value of W for which the log loss (error) is minimum. 
The gradient descent function for it is as follows:
![image](https://user-images.githubusercontent.com/74540513/149666457-677d2b22-0bd6-4d97-8380-122a2e61d4fa.png)

The log loss function is optimized using gradient dissent. After one iteration through the entire training set, we get a value for W called first epoch. This is further optimized by more iterations until we get more epochs and then a final optimized value, W_T. The Pi is calculated with this W_T value and this probability value allows us to predict the class for the test set.

The algorithm can be summarized as:

1.	Initialize W
2.	Give the data for training
3.	Compute the target
4.	Calculate the error
5.	Update W based on the error and the gradient descent 
6.	Update the weights and repeat from 2 to 6.

### 4.
Both the models were able to classify the test data and training data sufficiently. Various metric used to determine the same are accuracy, f1 score and confusion matrix.
Decision Tree:
Below screenshots are from the decision tree and we can make the following conclusions from it. The training data was classified with 100 % accuracy and the test data with 86 %.
The F1 score which is a weighted average of precision and recall is calculated as well. The score which is maximum at 1 is exhibited for training data. This doesn’t necessarily be a good thing as it could indicate overfitting.

![image](https://user-images.githubusercontent.com/74540513/149666496-9a372098-dbb5-4a54-a46a-a8fc3cb565b2.png)![image](https://user-images.githubusercontent.com/74540513/149666503-3d09d521-dfe2-4b60-826f-80ac23d3d75c.png)

The confusion matrix is also plotted for the same:

![image](https://user-images.githubusercontent.com/74540513/149666516-4831831d-26bc-4faf-9f66-40a47df69663.png)![image](https://user-images.githubusercontent.com/74540513/149666521-9c85a23d-3a38-4feb-b439-e027495f7dbf.png)

The decision tree created using the above algorithm for the given dataset is as follows:

![image](https://user-images.githubusercontent.com/74540513/149666527-09a44502-8fe3-44cb-84ef-3c9ef65e09cc.png)

### Logistic Regression:

In contrast to the decision tree, logistic regression has lesser accuracy on the training data and higher on the test data. This indicated a more generalized classification and is hence a little better than decision tree for this particular data.
![image](https://user-images.githubusercontent.com/74540513/149666543-9bb0bad1-d36c-4626-902b-7ba06bdab58d.png)![image](https://user-images.githubusercontent.com/74540513/149666550-ff896ab9-bacf-445c-9e8b-a0687eb776b9.png)

1.Accuracy comparison of training and test data		    2.F1 score comparison

![image](https://user-images.githubusercontent.com/74540513/149666563-8b332e90-1399-4d4d-9420-ef36db54ce14.png)![image](https://user-images.githubusercontent.com/74540513/149666567-48949150-ed34-41cf-9f7b-06b305c0dc0b.png)
    1.Confusion Matrix of Test data				     2.Confusion Matrix of Train Data

5.
I have used DecisionTreeClassifier and LogisticRegression to classify the data. Both algorithms performed really well with an accuracy score of .86 and .9 respectively. LogisticRegression seemed to be able to classify the data quite well and this is because the two classes ‘yes’ and ‘no’ are linearly separable. The same can be seen in the below sample depictions. We can see from the decision tree model above that drought_code and rainfall have the most information gain. In the below diagrams we take these two features and observe that a line can be drawn to separate the target to some extend and with the addition of the rest of the features, the linear seperability will increase and hence we are able to get an accuracy of about 90% on the test data.
  
![image](https://user-images.githubusercontent.com/74540513/149666585-19cdd0d3-cf27-4297-8f32-e531365239ec.png)![image](https://user-images.githubusercontent.com/74540513/149666592-b06ad73a-7c02-4ea4-90e4-26d02ca6a474.png)






