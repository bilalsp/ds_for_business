import pandas as pd
import numpy as np
import threading
import math

df = pd.read_csv('titanic_data.csv')

#region I part

#region change display width

# desired_width=20
# pd.set_option('display.width', desired_width)
# np.set_printoptions(linewidth=desired_width)
# pd.set_option('display.max_columns',10)

#endregion

#region surviving ratio

#pivot table to know avg_age by Class, Sex and Survived_Dummy
surv = pd.pivot_table(df, values = ['Age', 'Name'],
                      index=['Survived', 'Sex', 'Passenger Class'],
                      aggfunc={'Age': 'mean', 'Name': 'count'})
print("\n############ AVG AGE ############\n")
print(surv)

print("\n############ SURVIVNIG RATIOS ############\n")

#calculate Surv_ratio for each sex and class
for sex in ['Female', 'Male']:
    for passClass in ['First', 'Second', 'Third']:
        #survived and died avg_age for given sex and class
        survived = surv.loc[('Yes', sex, passClass), 'Name']
        died = surv.loc[('No', sex, passClass), 'Name']
        surv_ratio = survived / (survived + died)
        print('{} {} class: {:.2f}%'.format(sex, passClass, surv_ratio*100))

#endregion

#region average number at boats

print("\n############ AVG NUMBER OF PEOPLE ON THE BOATS ############\n")

boats = pd.pivot_table(df, values = 'Name', index=['Sex','Life Boat'], aggfunc='count')
female_avg = boats.loc[('Female'),'Name'].mean()
male_avg = boats.loc[('Male'),'Name'].mean()
print('Average male number on the boats: {:.0f}'.format(male_avg))
print('Average female number on the boats: {:.0f}'.format(female_avg))

#endregion

#endregion

#region II part

#region Family size

print("\n############ FAMILY SIZE ############\n")
df = df.drop(columns=['Life Boat'])
df['Family Size'] = df['No of Parents or Children on Board'] + \
                    df['No of Siblings or Spouses on Board'] + 1

print('Average family size: {:.2} people'.format(df['Family Size'].mean()))

print('The largest family was traveling by {} class'.format(
    df.iloc[np.argmax(np.array(df['Family Size'].values)),
            df.columns.get_loc('Passenger Class')]))

#endregion

#region Pass Fare

df['Number of passengers with this ticket'] = 0

for index in df.index:
    df.loc[index, 'Number of passengers with this ticket'] = len(
        df[df['Ticket Number'] == df.loc[index, 'Ticket Number']]
    )

df['Single passenger fare'] = df['Passenger Fare'] / df['Number of passengers with this ticket']
print("\n############ PASSENGERS' FARE ############\n")
print(df[['Name', 'Passenger Fare', 'Number of passengers with this ticket','Single passenger fare']])

#endregion
print(df.columns)
#region fill_na numerical

numerical = ['Age', 'No of Siblings or Spouses on Board',
             'No of Parents or Children on Board', 'Passenger Fare',
             'Family Size', 'Number of passengers with this ticket', 'Single passenger fare']

fill_pivot = pd.pivot_table(df,
                            values=numerical,
                            index=['Passenger Class', 'Sex', 'Port of Embarkation'])

def fill_function(col):
    for index in df.index:
        index1 = df.loc[index, 'Passenger Class']
        index2 = df.loc[index, 'Sex']
        index3 = df.loc[index, 'Port of Embarkation']
        if math.isnan(df.loc[index, col]):
            df.loc[index, col] = fill_pivot.loc[(index1, index2, index3), col]
            df.loc[index, col + suffix] = 1



suffix = ' presented'
for col in numerical:
    df[col + suffix] = 0

for col in numerical:
    fill_function(col)

#endregion

#region fill_na categorical

categorical = ['Passenger Class', 'Sex', 'Port of Embarkation',
               "Name", "Ticket Number", "Passenger Fare",
               "Cabin", "Survived"]

for col in categorical:
    dummies = pd.get_dummies(df[col])
    dummies = dummies.rename(
        {str(i): str(col) + '_' + str(i) for i in dummies.columns}, axis=1
    )
    df = pd.merge(df, dummies, left_index=True, right_index=True, suffixes=('', '_'+col))


df = df.drop(columns = categorical)
print("\n############ PREPROCESSED DATA ############\n")
print(df)

#endregion

#endregion

#region III part

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

data = df

col = 'Survived_Yes'
y_train = data[col]
x_train = data.drop(columns = [col]).select_dtypes(include=[np.number])

estimator = DecisionTreeClassifier(random_state=2020, max_depth=2)
estimator.fit(x_train, y_train)
y_pred = estimator.predict(x_train)
cm = confusion_matrix(y_train, y_pred)
accuracies = cross_val_score(estimator=estimator, X=x_train, y=y_train, cv=5, scoring='accuracy')

print("\n############ CONFUSSION MATRIX ############\n")
print(cm)
print("\n############ CROSS VALIDATION RESULTS ############\n")
print(accuracies)
print('Mean accuracy: {}'.format(accuracies.mean()))
#region

#endregion

#endregion