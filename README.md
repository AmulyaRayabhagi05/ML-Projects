Here are the ML projects that I have worked on recently: 
<h1> Linear Regression: </h1> <br>
<h2><b>Objective:</b> </h2>
The objective of this project is to predict students’ final grades (G3) using a linear regression model trained via gradient descent, based on student-related attributes such as prior grades, study time, and family background. The model aims to maximize prediction accuracy (measured by R² and MSE) while using only the most relevant features determined through correlation analysis.
<h2><b>Code Summary </b> </h2>
The project began by loading the UCI Student Performance dataset using the ucimlrepo library. To ensure data quality, duplicate and missing rows were removed. Next, categorical variables were converted into numerical form through one-hot encoding, and only features with a correlation greater than 0.15 with the target variable G3 were retained for modeling. The preprocessed data was then split into training and testing sets, standardized using 
StandardScaler, and augmented with a bias term to enable intercept learning. A linear regression model was implemented from scratch using gradient descent, and a grid search was conducted over various learning rates and iteration counts to identify the optimal Learning Rate and Iterations. Model performance was evaluated using both the R² score and Mean Squared Error (MSE), with the best model having a Learning Rate of 0.05, Iterations: 3000, R² Score: 0.8555 and MSE: 1.4094. G1, G2, studytime, and failures emerged as the most predictive features. To further analyze the model and dataset, a correlation heatmap of selected features was generated, along with scatter plots comparing G1 and G2 against G3, and a final plot illustrating predicted versus actual values for G3.

<h2><b>Dataset:</b> </h2> https://archive.ics.uci.edu/dataset/320/student+performance

<h2><b>Python Libraries used:</b> </h2> 
This project uses the following Python libraries:

- pandas 
- numpy
- sklearn.linear_model.SGDRegressor
- sklearn.metrics
- sklearn.model_selection
- sklearn.preprocessing.StandardScaler
- seaborn
- matplotlib.pyplot
- ucimlrepo

<h2><b>How to run the code: </b> </h2> 
- Install Python locally
    - Download the latest version from the Python website https://www.python.org/downloads/
    - Follow the installation instructions for the operating system


- Download the public dataset
    - Go to this dataset link: https://archive.ics.uci.edu/dataset/320/student+performance
    - Look for the “Download” link, and download the zip file containing the dataset.
    - Extract the contents and place it into the same folder as the part1.py and make sure that they are in the same directory.
- Install required libraries
    - Open the terminal and run this command:
     <b>pip install numpy pandas scikit-learn matplotlib seaborn ucimlrepo</b>
    - This command allows you to download and install all the necessary libraries 
- Run the code(.py files)
    - Use the cd command in the terminal to run this code
    - To run LinearRegression.py:
        - python LinearRegression.py

<h2><b>Conclusion </b> </h2>
This code demonstrates how a simple yet interpretable model like linear regression can yield high predictive performance when guided by feature selection and proper normalization, especially in educational data.

<hr>

<h1> Neural Networks: </h1> <br>
<h2><b>Objective:</b> </h2>
The objective of this project is to implement and evaluate a feedforward neural network from scratch to predict a target variable (overall rating) using a dataset from the UCI Machine Learning Repository (travel reviews). The network is trained using different combinations of hyperparameters—such as learning rate, number of hidden units, number of epochs, and activation functions—with the goal of optimizing prediction accuracy while minimizing training and test loss. The project emphasizes standard preprocessing steps, thorough hyperparameter tuning, and performance visualization through model history plots.

<h2><b>Code Summary:</b> </h2>
The program begins by loading the dataset through a parameterized file path, ensuring flexibility across environments. In the preprocess() method, the dataset is cleaned, standardized, and optionally converted from categorical to numerical format. The cleaned data is split into training and test sets, then fed into the neural network.
<br><br>The train_evaluate() method iterates over all combinations of specified hyperparameters, including different learning rates, hidden layer sizes, epochs, and activation functions (e.g., sigmoid, tanh). For each configuration, a neural network is trained, and both training and test metrics—accuracy and loss (mean squared error)—are computed.
<br><br>All models’ performance metrics are stored and summarized in a results table. Additionally, the training history (accuracy or loss over epochs) is plotted for each configuration to visualize convergence behavior. The final report identifies the best-performing model and discusses which hyperparameters most influenced performance and why.

<h2><b>Dataset:</b> </h2> https://archive.ics.uci.edu/dataset/484/travel+reviews
<h2><b>Python Libraries used:</b> </h2> 
This project uses the following Python libraries:

- pandas 
- numpy
- sklearn.model_selection.train_test_split
- sklearn.preprocessing.StandardScaler
- ssklearnpreprocessing.LabelEncoder
- sklearn.preprocessing.StandardScaler
- seaborn
- matplotlib.pyplot
- ucimlrepo

<h2><b>How to run the code: </b> </h2> 
- Download the public dataset
    - Go to this dataset link: https://archive.ics.uci.edu/dataset/320/student+performance
    - Look for the “Download” link, and download the zip file containing the dataset.
    - Extract the contents and place it into the same folder as the part1.py and make sure that they are in the same directory.
- Install required libraries
    - Open the terminal and run this command:
     <b>pip install numpy pandas scikit-learn matplotlib seaborn ucimlrepo</b>
    - This command allows you to download and install all the necessary libraries 
- Run the code(.py files)
    - Use the cd command in the terminal to run this code
    - To run LinearRegression.py:
        - python LinearRegression.py
     
<h2><b>Conclusion </b> </h2>
This code demonstrates how a basic neural network, when combined with systematic hyperparameter tuning and standard preprocessing, can achieve strong predictive performance even with a small architecture. By exploring different activation functions, learning rates, and hidden layer sizes, the project highlights the importance of model selection in achieving low test error.

