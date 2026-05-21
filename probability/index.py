# print("start learning probability")


from sklearn.feature_extraction.text import CountVectorizer  
# convert text → numbers (word counts)

from sklearn.naive_bayes import MultinomialNB  
# Naive Bayes model (uses probability)

texts = ["win money", "hello friend", "free offer", "let's meet", "win prize"]
labels = [1, 0, 1, 0, 1]  # 1 = spam, 0 = not spam

# Step 1: convert text into matrix
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
# each word → feature, values = frequency

# Step 2: create model
model = MultinomialNB()
# uses P(word|class) internally

# Step 3: train model
model.fit(X, labels)
# calculates probability:
# P(spam), P(word|spam), P(word|not spam)

# Step 4: predict new message
test = ["win free money"]
test_vec = vectorizer.transform(test)

prediction = model.predict(test_vec)
print(prediction)  # 1 = spam