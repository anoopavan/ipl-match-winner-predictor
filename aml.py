import tkinter as tk
from tkinter import *
#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import seaborn as sns

def submit_action():
    team1 = varT1.get()
    team2 = varT2.get()
    v = varV.get()

    # Process the values or perform any other actions here
    print("TEAM 1:", t1)
    print("TEAM 2:", t2)
    print("VENUE:", v)

    
    train_data = pd.read_csv('Training Matches IPL 2008-2019.csv')
    train_data

    train_data.shape

    train_data.head()

    train_data.isnull().sum()

    train_data['city'].fillna('Abu Dhabi',inplace=True)
    train_data['winner'].fillna('Draw', inplace = True)


    #Both Rising Pue Supergiant and Rising Pune Supergiants represents same team similarly Delhi Capitals and Delhi Daredevils,
    #Deccan Chargers and Sunrisers Hyderabad
    train_data.replace("Rising Pune Supergiant","Rising Pune Supergiants", inplace=True)
    train_data.replace('Deccan Chargers', 'Sunrisers Hyderabad', inplace=True)
    train_data.replace('Delhi Daredevils', 'Delhi Capitals', inplace=True)

    plt.subplots(figsize = (15,5))
    sns.countplot(x = 'season' , data = train_data, palette='dark')
    plt.title('Total number of matches played in each season')
    plt.show()

    plt.subplots(figsize=(15,10))
    ax = train_data['winner'].value_counts().sort_values(ascending=True).plot.barh(width=.9,color=sns.color_palette("husl", 9))
    ax.set_xlabel('count')
    ax.set_ylabel('team')
    plt.show()


    train_data.replace({"Mumbai Indians":"MI", "Delhi Capitals":"DC", 
                   "Sunrisers Hyderabad":"SRH", "Rajasthan Royals":"RR", 
                   "Kolkata Knight Riders":"KKR", "Kings XI Punjab":"KXIP", 
                   "Chennai Super Kings":"CSK", "Royal Challengers Bangalore":"RCB",
                  "Kochi Tuskers Kerala":"KTK", "Rising Pune Supergiants":"RPS",
                  "Gujarat Lions":"GL", "Pune Warriors":"PW"}, inplace=True)

    #'team1': {'KKR':0,'CSK':1,'RR':2,'MI':3,'SRH':4,'KXIP':5,'RCB':6,'DC':7,'KTK':8,'RPS':9,'GL':10,'PW':11},
    #         'team2': {'KKR':0,'CSK':1,'RR':2,'MI':3,'SRH':4,'KXIP':5,'RCB':6,'DC':7,'KTK':8,'RPS':9,'GL':10,'PW':11},

    from sklearn.preprocessing import LabelEncoder

    lesomething = LabelEncoder()
    lesomething.fit(['KKR','CSK','RR','MI','SRH','KXIP','RCB','DC','KTK','RPS','GL','PW'])

    encode = {'toss_winner': {'KKR':0,'CSK':1,'RR':2,'MI':3,'SRH':4,'KXIP':5,'RCB':6,'DC':7,'KTK':8,'RPS':9,'GL':10,'PW':11},
              'winner': {'KKR':0,'CSK':1,'RR':2,'MI':3,'SRH':4,'KXIP':5,'RCB':6,'DC':7,'KTK':8,'RPS':9,'GL':10,'PW':11,'Draw':12}}

    print(train_data["team1"])
    train_data['team1'] = lesomething.transform(train_data['team1'])
    train_data['team2'] = lesomething.transform(train_data['team2'])

    train_data.replace(encode, inplace=True)
    train_data.head(5)

    dicVal = encode['winner']
    train = train_data[['team1','team2','city','toss_decision','toss_winner','venue','winner']]
    train.head(10)

    from sklearn.preprocessing import LabelEncoder
    df = pd.DataFrame(train)
    var_mod = ['city','toss_decision']
    le = LabelEncoder()
    for i in var_mod:
        df[i] = le.fit_transform(df[i])


    levenue = LabelEncoder()
    levenue.fit(df["venue"])

    df['venue'] = levenue.transform(df['venue'])


    df.info()

    df

    from sklearn.preprocessing import StandardScaler
    X = df[['team1', 'team2', 'venue']]
    y = df[['winner']]
    sc = StandardScaler()
    X = sc.fit_transform(X)

    import numpy as np
    import pandas as pd

    # Assuming you have already read the 'Training Matches IPL 2008-2019.csv' into 'train_data'
    # If not, include the read_csv line here

    # Select numerical features for outlier detection
    numerical_features = ['team1', 'team2', 'toss_winner']

    # Calculate Z-scores for each numerical feature
    z_scores = np.abs((train_data[numerical_features] - train_data[numerical_features].mean()) / train_data[numerical_features].std())

    # Define a threshold for outlier detection (e.g., 3, which is a common value)
    threshold = 3

    # Detect outliers
    outliers = train_data[z_scores > threshold]

    print("Outliers:")
    print(outliers)


    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    from xgboost import XGBClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import svm
    logistic_model = LogisticRegression()
    logistic_model.fit(X,y)
    print("Logistic Regression accuracy: ",(logistic_model.score(X,y))*100)
    Random_model = RandomForestClassifier()
    Random_model.fit(X,y)
    print("Random Forest accuracy: ", (Random_model.score(X,y))*100)
    knn_model = KNeighborsClassifier()
    knn_model.fit(X,y)
    print("KNeighbor Classifier accuracy: ", (knn_model.score(X,y))*100)
    NB_model = GaussianNB()
    NB_model.fit(X,y)
    print("Gaussion Navie Bayes accuracy: " ,(NB_model.score(X,y))*100)
    decision_model = DecisionTreeClassifier()
    decision_model.fit(X,y)
    print("Decision Tree Classifier accuracy: ", (decision_model.score(X,y))*100)
    svm_model=svm.SVC(kernel='linear')
    svm_model.fit(X,y)
    print("SVM accuracy: ", (svm_model.score(X,y))*100)

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    # Assuming you have already read the 'Training Matches IPL 2008-2019.csv' into 'train_data'
    # If not, include the read_csv line here

    # ... (The rest of your preprocessing code remains the same) ...

    # Split the data into features (X) and target (y)
    X = df.drop('winner', axis=1)
    y = df['winner']



    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an XGBoost classifier and train it on the training data
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)


    # Evaluate the model on the testing data
    accuracy = xgb_model.score(X_test, y_test)
    print("XGBoost accuracy: ", accuracy * 100)



    # Encode the input data in the same way as the training data
    team1_encoded = lesomething.transform([team1])[0]
    print(team1_encoded)
    team2_encoded = lesomething.transform([team2])[0]

    print(team2_encoded)
    venue_encoded = levenue.transform([venue])[0]

    print(venue_encoded)

    # Create a DataFrame with the input data and predict the winner
    input_data = pd.DataFrame({
        'team1': [team1_encoded],
        'team2': [team2_encoded],
        'city': [venue_encoded],
        'toss_decision': [0],  # Assuming toss decision is always 'bat' (0)
        'toss_winner': [team1_encoded],
        'venue': [venue_encoded]
    })

    predicted_winner = xgb_model.predict(input_data)[0]
    winner = list(dicVal.keys())[list(dicVal.values()).index(predicted_winner)]



        # Update the output label to display the winner
    output_label.config(text="Winner: " + winner)














# Main window
frame = tk.Tk()
frame.attributes('-fullscreen', True)
frame.title("TextBox Input")

# TEAM 1 label
l = tk.Label(frame, text="TEAM 1", bg="#E0EEEE", font=250)
l.pack()
l.place(x=710, y=330)
varT1 = StringVar()

# Entering values for TEAM 1
inputstr = Entry(frame, width=30, font=200, textvariable=varT1)
inputstr.pack()
inputstr.place(x=580, y=360)

# TEAM 2 label
d = tk.Label(frame, text="TEAM 2", bg="#E0EEEE", font=350)
d.pack()
d.place(x=710, y=390)
varT2 = StringVar()

# Entering value for TEAM2
inputdes = Entry(frame, width=30, font=200, textvariable=varT2)
inputdes.pack()
inputdes.place(x=580, y=421)

# VENUE label
d = tk.Label(frame, text="VENUE", bg="#E0EEEE", font=350)
d.pack()
d.place(x=710, y=450)
varV = StringVar()

# Entering value for VENUE
inputdes = Entry(frame, width=30, font=200, textvariable=varV)
inputdes.pack()
inputdes.place(x=580, y=480)

# Submit button
printButton = tk.Button(frame, text="SUBMIT", font=300, bg='#00CDCD', command=submit_action)
printButton.pack()
printButton.place(x=700, y=520)

# Output label to display the winner
output_label = Label(frame, text="", font=300)
output_label.pack()
output_label.place(x=650, y=600)

frame.mainloop()
