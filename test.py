import abc  as ABC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

class app:
    def __init__(self):
        self.df = None
        self.model = None
        self.features = None
        self.classes = None

    def main_menu(self):
        while True:
            print ("1. load data")
            print("2. train model")
            print("3. evaluate model")
            print("4. real simulate")
            print("5. exist")
            choice = int(input("what do yo want to do? "))

            if choice == 1:
                self.load_data()
            if choice == 2:
                self.train_model()
            if choice == 3:
                self.evaluate_model()
            if choice == 4:
                self.simulate()
            if choice == 5:
                break
            else:
               print ('try again with (1/5)')

    def load_data(self):
    #call the varaibles
      print ("1: for iris dataset (already available)")
      print("2: for tic-tac-toe dataset (already available)")
      print("3: for your own dataset")
    #ask for dataset
      data_choice = int(input("enter the dataset choice! "))

      if data_choice == 1:
        self.df = pd.read_csv("iris.csv")   

      elif data_choice == 2:
        # since the dataset has no columns it has to be identified 
        self.df = pd.read_csv("tic-tac-toe.data", header=None)
        # manually assingned column names since the data set has none 
        self.df.columns = [
            "top-left", "top-middle", "top-right",
            "middle-left", "middle-middle", "middle-right",
            "bottom-left", "bottom-middle", "bottom-right",
            "class"
        ]
      elif data_choice == 3:
       own_data = input ('paste in your dataset-path ')
       #convert path to be readeable
       own_data = own_data.strip().strip('"').strip("'")
       own_data = own_data.replace("\\", "/")    

       try:
         self.df = pd.read_csv(own_data)
         self.df.rename(columns={self.df.columns[-1]: "class"}, inplace = True)
       except Exception as e:
        print(" your dataset could not be loaded, try again!!")
        print('error', e)
        return
      else:
          print('invalid choice!! ')
          return
# seperates the target class
      self.classes = self.df["class"]
#seperates the futures
      self.features = self.df.drop("class", axis=1)

      print(self.df.head(10))
      print (self.df.describe())

    def train_model(self):
     
     if self.df is None:
        print ("load a data!!")
        return
         # encode the strings to numbers for KNN to work
    #creates a coder object 
     coder = LabelEncoder()
    # apply it to all the features (aka "X", "o","b") into ints, (1,2,3)
     coder_features = self.features.apply(lambda col: LabelEncoder().fit_transform(col))
    # now apply it to the class too (aka "positive", "negative")
     coder_classes = LabelEncoder().fit_transform(self.classes)
    
    # make sure every class appears rwice, otherwise abort "stratify"
     class_count = pd.Series(coder_classes).value_counts() # this variable counts how many samples each class has
     if (class_count < 2).any(): # this checks if any of the classes has less than 2 sampables 
        stratify_parm = None  # disable stratification 
     else:
        stratify_parm = coder_classes 
        
    # Split the data into STRATIFIED train/test sets:
     strat_feat_train, strat_feat_test, strat_classes_train, strat_classes_test = train_test_split(
      coder_features, coder_classes, test_size=0.4, random_state=10, stratify= stratify_parm )
    #added coder. to features and classes so the the encoded version of them gets trained

     print ("choose a model\n""1. for KNN\n""2. for Decision Tree\n")
     model_choice = int(input("enter your choice "))

     if model_choice == 1:
     # Creates a Knn Classifier Object with k=1
       k = int(input("enter the desired K value (recommended is 6) "))
       self.model = KNeighborsClassifier(n_neighbors=k) 
    # creates a Decision Tree Classifier Object
     elif model_choice == 2:
        max_depth = input("Enter max depth (press Enter for default): ").strip()
        
        if max_depth:
            max_depth = int(max_depth)
            self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=10)
        else:
            self.model = DecisionTreeClassifier(random_state=10)
     else:
        print("Invalid choice!")
        return
    
    # Trains model assifier with the training set obtained previously:
     self.model.fit(strat_feat_train, strat_classes_train)

       #Obtains the predictions from the model classifier:
     predictions = self.model.predict(strat_feat_test)
     self.acc = accuracy_score(strat_classes_test, predictions)
     self.report = classification_report(strat_classes_test, predictions)
     self.matrix = confusion_matrix(strat_classes_test, predictions)

     #Prints accuracy
     print("Accuracy:", self.acc)
    
    def evaluate_model(self):
     if self.model is None:
        print("Train a model first!!")
        return
     
     #prints
     print("------ model evaluations ------")
     print("Accuracy:", self.acc)
     print ("\n------classification report------")
     print(self.report)
     print ("\n------Confusion Matrix------")
     print(self.matrix)

     save_choice = input("\nDo you want to save these results to a file? (y/n): ").strip().lower()
     if save_choice == "y":
        file_name = input("Enter file name ").strip()
        if not file_name.endswith(".txt"):
         file_name += ".txt"
        with open(file_name, "w") as f:
            f.write("------ Model Evaluation Results ------\n")
            f.write(f"Accuracy: {self.acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(self.report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(self.matrix) + "\n")
        print(f"Results saved successfully to '{file_name}'")
     else:
        print("Results not saved.")

    def simulate(self):
      if self.model is None:
        print ("train a model first!!")
        return
      if self.model == "tic-tac-toe":
        print("enter x for player X and o for player O and b for blank")
      else:
        print("enter the feature values")
      input_features = [] 
      for col in self.features.columns:
        value = input(f"enter value for {col}: ")
        input_features.append(value)
    # Convert input features to DataFrame
      input_df = pd.DataFrame([input_features], columns=self.features.columns) 
    # Encode input features
      self.coder = LabelEncoder()
      coder_input = input_df.apply(lambda col: LabelEncoder().fit_transform(col))
    # Make prediction
      prediction = self.model.predict(coder_input)
      print("Predicted class:", prediction[0])

if __name__ == "__main__":
    app_instance = app()
    app_instance.main_menu()