import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

df_review = pd.read_csv("Sentiment_Analysis\IMDB Dataset.csv\IMDB Dataset.csv")
df_review

df_positive = df_review[df_review["sentiment"] == "positive"][:9000]
df_negative = df_review[df_review["sentiment"] == "negative"][:1000]

df_review_imb = pd.concat([df_positive, df_negative])

rus = RandomUnderSampler(random_state = 0)
df_review_bal, df_review_bal["sentiment"] = rus.fit_resample(df_review_imb[["review"]], df_review_imb["sentiment"])
df_review_bal

train, test = train_test_split(df_review_bal, test_size = 0.33, random_state = 42)

train_x, train_y = train["review"], train["sentiment"]
test_x, test_y = test["review"], test["sentiment"]

tfidf = TfidVectorizer(stop_words = "english")
train_x_vector = tfidf.fit_transform(train_x)
train_x_vector

pd.DataFrame.sparse.from_spmatrix(train_x_vector, index = train_x.index, columns = tfidf.get_feature_names())

test_x_vector = tfidf.transform(test_x)

svc = SVC(kernel = "linear")
svc.fit(train_x_vector, train_y)

print(svc.predict(tfidf.transform(["A good movie"])))
print(svc.predict(tfidf.transform(["An excellent movie"])))
print(svc.predict(tfidf.transform(["I did not like this movie at all"])))

dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)

gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)

# svc.score("Test Samples", "True Labels")

svc.score(test_x_vector, test_y)
dec_tree.score(test_x_vector, test_y)
gnb.score(test_x_vector.toarray(), test_y)
log_reg.score(test_x_vector, test_y)

f1_score(test_y, svc.predict(test_x_vector), labels = ["positive", "negative"], average = None)

print(classification_report(test_y, svc.predict(test_x_vector), labels = ["positive", "negative"]))

conf_mat = confusion_matrix(test_y, svc.predict(test_x_vector), labels = ["positive", "negative"])

# set the parameters
parameters = {"C": [1, 4, 8, 16, 32], "kernel": ["linear", "rbf"]}
svc = SVC()
svc_grid = GridSearchCV(svc, parameters, cv = 5)
svc_grid.fit(train_x_vector, train_y)
