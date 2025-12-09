from sklearn.model_selection import train_test_split

x = df.drop('Churn', axis=1) # Contains all input features
y = df['Churn'].map({'No':0, 'Yes':1}) # Contains Churn variable mapped to 0 and 1

