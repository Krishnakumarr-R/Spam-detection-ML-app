import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #convert text to numerical
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Load data
raw_mail_data = pd.read_csv('mail_data.csv')

#Label the spam as 0 and ham as a 1
mail_data.loc[mail_data['catagory'] == "spam",'catagory',] = 0
mail_data.loc[mail_data['catagory'] == "ham",'catagory',] = 1

# sepearte the data as text and label
X = mail_data['Message']
Y = mail_data['Category']

#split the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# convert text to numerical (future vectors)
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#convert y_train and y_test values as integer
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


#Training the model using logistic Regression
model = LogisticRegression()
model.fit(X_train_features, Y_train)

#prediction on traing data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_traning_data = accuracy_score(Y_train, prediction_on_training_data)
print("Accuracy_on_traning_data",accuracy_on_traning_data)

#prediction on traing data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_train, prediction_on_test_data)
print("Accuracy_on_test_data",accuracy_on_test_data)


#Building priditive system
input_mail=["WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]
input_data_features = feature_extraction.transform(input_mail) #convert text to feature vector
prediction = model.predict(input_data_features)#making prediction
if(prediction[0]==1):
  print('Ham mail')
else:
  print('Spam mail')


#to save model
import joblib

# Save the model
joblib.dump(model, 'spam_classifier_model.joblib')

# Save the TfidfVectorizer
joblib.dump(feature_extraction, 'tfidf_vectorizer.joblib')

# Download files in Colab
from google.colab import files
files.download('spam_classifier_model.joblib')
files.download('tfidf_vectorizer.joblib')