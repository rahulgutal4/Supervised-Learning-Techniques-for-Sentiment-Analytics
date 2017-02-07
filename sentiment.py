import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
import numpy
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
nltk.download("stopwords")          # Download the stop words from nltk
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg

def flatten(train):
    vals = []
    for line in train:
        for word in line:
            vals.append(word)
    return vals        

def filterStopWords(train, stopwords):
    vec = []
    for line in train:
        l = set(line)
        vec.append(filter(lambda x: x not in stopwords,l))
    return vec
        
def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.

    pos_len = len(train_pos)
    neg_len = len(train_neg)
    
    train_pos = filterStopWords(train_pos, stopwords)
    train_neg = filterStopWords(train_neg, stopwords)

    pos_freq = nltk.FreqDist(flatten(train_pos, stopwords))
    neg_freq = nltk.FreqDist(flatten(train_neg, stopwords))
    
    features = []
    for t in pos_freq.items():
        key = t[0]
        freq = t[1]
        neg_F = 0
        if key in neg_freq:
            neg_F = neg_freq[key]
            
        if((freq >= 0.01*pos_len or neg_F >= 0.01*neg_len) and freq >= 2*neg_F):
            features.append(key)
    
    for key in neg_freq.keys():
        freq = neg_freq[key]
        pos_F = 0
        if key in pos_freq:
            pos_F = pos_freq[key]
        if((freq >= 0.01*neg_len or pos_F >= 0.01*pos_len) and freq >= 2*pos_F):
            features.append(key)
    
    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    train_pos_vec = getVectors(features, train_pos)
    train_neg_vec = getVectors(features, train_neg)
    test_pos_vec = getVectors(features, test_pos)
    test_neg_vec = getVectors(features, test_neg)
    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def getVectors(feature, text):
    vec = []
    length = len(feature)
    
    for line in text:
        v = []
        d = dict(zip(feature,[0]*length))
        
        for word in line:
            if word in d:    
                d[word] = 1
         
        for key in d.keys():
            v.append(d[key])
    
        vec.append(v)

    return vec 

def generateLabelledSentences(data, label):
    i=0
    labelledData = []
    for line in data:
        labelledData.append(LabeledSentence(words=line, tags=[label+str(i)]))
        i+=1
    return labelledData    

def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    
    labeled_train_pos = generateLabelledSentences(train_pos, 'TRAIN_POS_') 
    labeled_train_neg = generateLabelledSentences(train_neg, 'TRAIN_NEG_')
    labeled_test_pos = generateLabelledSentences(test_pos, 'TEST_POS_')
    labeled_test_neg = generateLabelledSentences(test_neg, 'TEST_NEG_')
    
    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)
    
    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    train_pos_vec = extractFeatureVectors(model, 'TRAIN_POS_', len(train_pos))
    train_neg_vec = extractFeatureVectors(model, 'TRAIN_NEG_', len(train_neg))
    test_pos_vec = extractFeatureVectors(model, 'TEST_POS_', len(test_pos))    
    test_neg_vec = extractFeatureVectors(model, 'TEST_NEG_', len(test_neg)) 
       
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def extractFeatureVectors(model, label, len):
    vec = []
    for i in range(0,len):
        vec.append(model.docvecs[label + str(i)])
    return vec
    
def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = train_pos_vec + train_neg_vec
    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model = BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(X,Y)
   
    lr_model = LogisticRegression()
    lr_model.fit(X,Y)
    
    return nb_model, lr_model

def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = train_pos_vec + train_neg_vec
    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model = GaussianNB()
    nb_model.fit(X,Y)
    
    lr_model = LogisticRegression()
    lr_model.fit(X,Y)

    return nb_model, lr_model

def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    res = model.predict(test_pos_vec)
    unique, counts = numpy.unique(res, return_counts=True)
    d = dict(zip(unique, counts))
    tp = float(d['pos'])
    fn = float(d['neg'])
    
    res = model.predict(test_neg_vec)
    unique, counts = numpy.unique(res, return_counts=True)
    d = dict(zip(unique, counts))
    tn = float(d['neg'])
    fp = float(d['pos'])
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
