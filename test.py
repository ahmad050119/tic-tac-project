#hhh
# ADJGPOSJGSL
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


#variabbles
df = None
model = None
features = None
classes = None

def main_menu():
    while True:
        print ("1. load data")
        print("2. train model")
        print("3. evaluate model")
        print("4. real simulate")
        print("5. exist")
        choice = int(input("what do yo want to do? "))

        if choice == 1:
            load_data()
        if choice == 2:
            train_model()
        if choice == 3:
            evaluate_model()
        if choice == 4:
            simulate()
        if choice == 5:
            break


def load_data():
    #call the varaibles
    global df, features, classes
    
    print ("iris csv is available")
    print("tic is available")
    #ask for dataset
    data_name = input("enter the dataset name! ").strip()

    if data_name == 'iris':
        df = pd.read_csv("iris.csv")   

    elif data_name == 'tic':
        # since the dataset has no headers it has to be identified 
        df = pd.read_csv("tic-tac-toe.data", header=None)
        # manually assingned column names since the data set has none 
        df.columns = [
            "top-left", "top-middle", "top-right",
            "middle-left", "middle-middle", "middle-right",
            "bottom-left", "bottom-middle", "bottom-right",
            "class"
        ]

    else:
       return print(" try again!!")
        

# seperates the target class
    classes = df["class"]
#seperates the futures
    features = df.drop("class", axis=1)

    print(df.head(10))
    print (df.describe())

def train_model():
    global df, model, features, classes
    if df is None:
        print ("load a data!!")
        return
    
    # encode the strings to numbers for KNN to work
    #creates a coder object 
    coder = LabelEncoder()
    # apply it to all the features (aka "X", "o","b") into ints, (1,2,3)
    coder_features = features.apply(lambda col: LabelEncoder().fit_transform(col))
    # now apply it to the class too (aka "positive", "negative")
    coder_classes = LabelEncoder().fit_transform(classes)
    
    # Split the data into STRATIFIED train/test sets:
    strat_feat_train, strat_feat_test, strat_classes_train, strat_classes_test = train_test_split(
     coder_features, coder_classes, test_size=0.4, random_state=10, stratify= coder_classes )
    #added coder. to features and classes so the the encoded version of them gets trained
    print ("choose a model\n""1. for KNN\n""2. for Decision Tree\n")

    model_choice = int(input("enter your choice "))

    if model_choice == 1:
    # Creates a Knn Classifier Object with k=1
       k = int(input("enter the desired K value (recommended is 6) "))
       model = KNeighborsClassifier(n_neighbors=k) 

    # Trains this Knn Classifier with the training set obtained previously:
       model.fit(strat_feat_train, strat_classes_train)

    elif model_choice == 2:
     max_depth = input("Enter max depth (press Enter for default): ").strip()
     if max_depth:
        max_depth = int(max_depth)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=10)
     else:
        model = DecisionTreeClassifier(random_state=10)
    
    else:
        print("Invalid choice!")
    return


    
    
# Obtains the predictions from the kNN classifier:
    predictions = model.predict(strat_feat_test)

# printss
    input("enter to con")
    print("------ model evaluations ------")
    print("Accuracy:", accuracy_score(strat_classes_test, predictions))
#Prints the classification report:
    print ("\n------classification report------")
    print(classification_report(strat_classes_test, predictions))



def evaluate_model():
    pass

def simulate():
    global df, model, features, classes
    if model is None:
        print ("train a model first!!")
        return
    if model == "tic-tac-toe":
        print("enter x for player X and o for player O and b for blank")
    else:
        print("enter the feature values")
    input_features = [] 
    for col in features.columns:
        value = input(f"enter value for {col}: ")
        input_features.append(value)
    # Convert input features to DataFrame
    input_df = pd.DataFrame([input_features], columns=features.columns) 
    # Encode input features
    coder = LabelEncoder()
    coder_input = input_df.apply(lambda col: LabelEncoder().fit_transform(col))
    # Make prediction
    prediction = model.predict(coder_input)
    print("Predicted class:", prediction[0])
    

    

if __name__ == "__main__":
    main_menu()