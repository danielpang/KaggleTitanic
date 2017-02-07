import pandas
# Import the logistic regression class
from sklearn.linear_model import LogisticRegression

# Read in data
data = pandas.read_csv("train.csv")
test_data = pandas.read_csv("test.csv")

# Print summary of each column data
#print(data.describe())

# We've noticed age is missing in some row values
# We can replace NULL values with the median of the other ages
# Use .fillna(val) pandas function
data['Age'] = data['Age'].fillna(data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
# Re-classify gender values into either 0 or 1
# 0 for male, 1 for female
# use .df.loc[row_index,col_indexer] pandas function
data.loc[data['Sex'] == 'male', 'Sex'] = 0
data.loc[data['Sex'] == 'female', 'Sex'] = 1
test_data.loc[test_data['Sex'] == 'male', 'Sex'] = 0
test_data.loc[test_data['Sex'] == 'female', 'Sex'] = 1

# Re-classify embarked values (where people got on the Titanic)
# 0 for S, 1 for C, 2 for  Q

# First fill in NULL values with S, the most common embarked port
data['Embarked'] = data['Embarked'].fillna('S')
test_data['Embarked'] = test_data['Embarked'].fillna('S')

data.loc[data['Embarked'] == 'S', 'Embarked'] = 0
data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2
test_data.loc[test_data['Embarked'] == 'S', 'Embarked'] = 0
test_data.loc[test_data['Embarked'] == 'C', 'Embarked'] = 1
test_data.loc[test_data['Embarked'] == 'Q', 'Embarked'] = 2

test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LogisticRegression(random_state=1)


# the data we're using to train the algorithm
train_predictors = (data[predictors])
train_target = data["Survived"]
alg.fit(train_predictors, train_target)
predictions = alg.predict(test_data[predictors])

# Create a new Data Frame with only the columns PassengerID and Survived
submission = pandas.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("kaggle.csv", index=False)

# Let's improve this score
