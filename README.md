IPL Match Winner Prediction

This project predicts the winner of an Indian Premier League (IPL) cricket match using machine learning. The prediction is based on inputs such as Team 1, Team 2, and the match venue, which are provided through a simple graphical user interface built using Python’s Tkinter library.

The dataset used for training contains IPL match details from 2008 to 2019. The data is cleaned by handling missing values, standardizing team names, and encoding categorical variables such as teams, venues, and toss information. This preprocessing ensures the data is suitable for machine learning models.

Several machine learning algorithms are trained and evaluated in this project, including Logistic Regression, Random Forest, K-Nearest Neighbors, Naive Bayes, Decision Tree, Support Vector Machine, and XGBoost. Among these, XGBoost is used as the final model due to its better performance on the test data.

The trained model is integrated with a Tkinter-based GUI where the user can enter the two teams and venue. Once the submit button is clicked, the model processes the input, applies the same encoding used during training, and predicts the winning team, which is displayed on the screen.

This project demonstrates an end-to-end machine learning workflow, starting from data preprocessing and model training to real-time prediction using a desktop application. It showcases practical use of machine learning in sports analytics and basic deployment using a graphical interface.
