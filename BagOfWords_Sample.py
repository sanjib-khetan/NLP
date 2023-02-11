import numpy as np
import sklearn
import string
from nltk.corpus import stopwords

file = open("sample_data_train.txt", "r+")
f = file.readlines()
f1 = list(map(lambda a:a.strip("\n"), f))
f1 = [i for i in f1 if len(i) > 0]
X = [i[:-2] for i in f1]
Y = [float(i.split(" ")[-1]) for i in f1]

from collections import Counter
vocab = Counter()

def clean_doc(X):

    # split into tokens by white space
    for doc in X:
        tokens = doc.split()
        # remove punctuation from each token
        #tokens = [i.replace(".", "") for i in tokens]
        #tokens = [i.replace(",", "") for i in tokens]
        table = str.maketrans('', '', string.punctuation)
        
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        vocab.update(tokens)
        # filter out stop words
        #stop_words = set(stopwords.words('english'))
        #tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        #tokens = [word for word in tokens if len(word) > 1]

    return tokens

a = clean_doc(X)
print(vocab)


from keras.preprocessing.text import Tokenizer
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
'''docs = ["I am a good boy",
        "he is also a good boy",
        "they are playing football",
        "girls are playing kho kho",
        "good boy is playing football"]'''

tokenizer.fit_on_texts(X)
print(tokenizer)

# encode training data set
Xtrain = tokenizer.texts_to_matrix(X, mode='count')
print(Xtrain)



from keras.models import Sequential
from keras.layers import Dense


#define network
model = Sequential()
model.add(Dense(32, input_shape=(66,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit network
model.fit(Xtrain, y = np.array(Y), epochs=50, verbose=2)



#Create Test Data

file_test = open("sample_data_test.txt", "r+")
f_test = file_test.readlines()
f1_test = list(map(lambda a:a.strip("\n"), f_test))
f1_test = [i for i in f1_test if len(i) > 0]
X1 = [i[:-2] for i in f1_test]

X_test =[]
for doc in X1:
    tokens = doc.split()
    # remove punctuation from each token
    tokens = [i.replace(".", "") for i in tokens]
    
    tokens = [i.replace(",", "") for i in tokens]
    table = str.maketrans('', '', string.punctuation)
        
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    #print("==", tokens)
    l = [j for j in tokens if j in vocab]
    #print("=======", l)
    X_test.append(" ".join(l))
        
Y_test = [int(i.split(" ")[-1]) for i in f1_test]



# encode training data set
Xtest = tokenizer.texts_to_matrix(X_test, mode='freq')
#print(Xtest)


# evaluate
loss, acc = model.evaluate(Xtest, np.array(Y_test), verbose=0)
print('Test Accuracy: %f' % (acc*100))