import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import metrics
import pickle
import string
from nltk.corpus import stopwords
from sklearn.svm import SVC 
from sklearn.preprocessing import FunctionTransformer
from sacremoses import MosesDetokenizer


def main():     
    spam=readSpamMails()
    ham=readHamMails()
    Emails=joinHamAndSpamMails(ham,spam)
    train_set, test_set = createTrainAndTestSet(Emails)
      
    explDataAnalysis(Emails,train_set,test_set)
    compareHamSpamMailLength(Emails,train_set,test_set) 
    
    ## Apply preprocessing to remove punctuation and stop words from mails
    processed_train= train_set.emails.apply(preProcessing) 
    processed_train= processed_train.to_frame()
    processed_train['target'] = train_set['target']        
    print('train preprocessing is complete')
   
    
    processed_test= test_set.emails.apply(preProcessing)    
    processed_test= processed_test.to_frame()
    processed_test['target'] = test_set['target']
    print('test preprocessing is complete') 
    
    # Build models and store model name, F1-score and AUC for each model in 'accuracyDF' dataframe
    
    accuracyDF = pd.DataFrame(columns= ['ModelName','F1Score','AUC']) # A data frame to store the perfomance metric of different classifiers
    accuracyDF= accuracyDF.append(buildModel1(processed_train,processed_test), ignore_index = True)     
    accuracyDF= accuracyDF.append(buildModel2(processed_train,processed_test), ignore_index = True)
    accuracyDF= accuracyDF.append(buildModel3(processed_train,processed_test), ignore_index = True)        
    accuracyDF= accuracyDF.append(buildModel4(processed_train,processed_test), ignore_index = True)    
    accuracyDF= accuracyDF.append(buildModel5(processed_train,processed_test), ignore_index = True)    
    
    print(accuracyDF.to_string())
    # visualise model perfomance
    visualiseModelAccuracy(accuracyDF)

def readHamMails():    
    path_ham="./data/enron/ham/*.txt"  
    ham_files = glob.glob(path_ham)
    list_ham =[]
    for file_path in ham_files:
        with open(file_path) as f:
            list_ham.append(f.read())
    ham = pd.DataFrame(list_ham,columns = ["emails"])
    ham["target"] = 0
    return ham
  
def readSpamMails():
    path_spam = "./data/enron/spam/*.txt"
    spam_files = glob.glob(path_spam)
    list_spam =[]
    for file_path in spam_files:
        with open(file_path,encoding = "Latin-1") as f:
            list_spam.append(f.read())        
    spam = pd.DataFrame(list_spam,columns = ["emails"])
    spam["target"] = 1
    return spam

def joinHamAndSpamMails(ham,spam):
    Emails = ham.append(spam)
    Emails = Emails.sample(frac =1).reset_index(drop = True) 
    return Emails

def createTrainAndTestSet(Emails):
    #splitting using sklearn ,70% training data , 30% test data
    X_train,X_test,Y_train,Y_test = train_test_split(Emails['emails'],Emails['target'],test_size = 0.3,
                                                 random_state = 13)
    train_set = pd.concat([X_train,Y_train],axis = 1,join_axes =[X_train.index])
    test_set = pd.concat([X_test,Y_test],axis = 1,join_axes =[X_test.index])
    train_set.to_csv("train_data.csv",sep = ',',encoding = 'Latin-1',index = False)
    test_set.to_csv("test_data.csv",sep = ',',encoding = 'Latin-1',index = False)
    return train_set,test_set

def preProcessing(mail):    
    # punctuation removal  
    noPuncChars=[]
    cleanMails=[]
    for char in mail:
        if (char  not in string.punctuation) and (char.isdigit() == False): # Remove numbers
            noPuncChars.append(char)
    mailsWithnoPuncChars = ''.join(noPuncChars)
      
   ## remove stop words and word 'subject' as it is not relevent in classifying mails
    for word in mailsWithnoPuncChars.split():
        if (word.lower() not in stopwords.words('english')) and ( word != 'Subject'):
           cleanMails.append(word) 
   ## de tokenize sentences
    detokenizer = MosesDetokenizer()
    cleanMails=detokenizer.detokenize(cleanMails, return_str=True)
    return cleanMails  

def findTop20Words(mails, vectorizer):
    #vectorizer = CountVectorizer()
    vec = vectorizer.fit(mails.emails)
    bag_of_words= vectorizer.transform(mails.emails)  
    sum_words = bag_of_words.sum(axis=0)
    word_count_list=[]  
    for word, count in vec.vocabulary_.items():       
        word_count_list.append([word,sum_words[0, count]])
    words_freq =sorted(word_count_list, key = lambda x: x[1], reverse=True)        
    return (words_freq[:20])


def explDataAnalysis(Emails, train_set, test_set):   
    # produces a bar plot showing total no of ham and spam mails in training set
    ax1=train_set['target'].value_counts().plot(kind='bar')
    ax1.set_ylabel("Count of mails") 
    ax1.legend(["0= HAM 1= SPAM"])
    ax1.set_title("Count of spam and ham mails in training set")
    plt.show()
    
    # produces a bar plot showing total no of ham and spam mails in test set
    ax2=test_set['target'].value_counts().plot(kind='bar')
    ax2.set_ylabel("Count of mails") 
    ax2.legend(["0= HAM 1= SPAM"])
    ax2.set_title("Count of spam and ham mails in test set")
    plt.show()    
    
    # #Find the top-20 most frequently used words in spam and non-spam emails
    # and use a bar plot to show their relative frequencies.
    # filter spam mails
   
    s= train_set[train_set['target'] ==1]
    spamTopWords = findTop20Words(s,CountVectorizer())
    ## plot barchart
    labels, count = zip(*spamTopWords)
    plt.clf()
    plt.bar(labels, count, 1, align='center')    
    plt.savefig('topWords1.png')
    
    # top words are the, to etc. so lets remove stopwords
    spamTopWords=findTop20Words(s, CountVectorizer(stop_words='english'))
    labels, count = zip(*spamTopWords)
    plt.clf()
    plt.bar(labels, count, 1, align='center')    
    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.xlabel('Spam words')
    plt.ylabel('Frequency of words')
    #plt.savefig('topWordsSpam.png') 
    plt.show()
    hamTopWords = findTop20Words(train_set[train_set['target'] ==0],CountVectorizer(stop_words='english'))
    
    # word subject appears top. Remove this word from vocabulary
    labels, count = zip(*hamTopWords)
    plt.clf()
    plt.bar(labels, count, 1, align='center')    
    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.xlabel('Ham words')
    plt.ylabel('Frequency of words')
    plt.show()
    #plt.savefig('topWords_ham.png')

def compareHamSpamMailLength(Emails, train_set, test_set):
    #Compare the distribution of email lengths in spam and non
    #spam emails using appropriate plots (e.g. a boxplot). 
    
    spam_emails=train_set[train_set['target'] ==1]
    ham_emails=test_set[test_set['target'] ==0]
    
    # Add extra column for saving the length of each mail
    spam_emails['mail_length'] = spam_emails['emails'].map(lambda mail: len(mail))
    ham_emails['mail_length'] = ham_emails['emails'].map(lambda mail: len(mail))
    
    '''
    plt.clf()
    plt.hist(ham_emails.mail_length,bins=100, label = 'ham')
    plt.hist(spam_emails.mail_length,bins =100,label = 'spam')
    plt.legend(loc='upper right')
    plt.title("Distribution of length of mails")
    plt.ylabel("Mail Length")
    plt.show()
    '''
    
    ## Above commented area produced a histogram plot having long tail. So, select only mails having size less than 10000
    plt.clf() # to clear the figure
    plt.hist(ham_emails.mail_length[ham_emails.mail_length<10000],bins=100, label = 'ham')
    plt.hist(spam_emails.mail_length[spam_emails.mail_length<10000],bins =100,label = 'spam')
    plt.legend(loc='upper right')
    plt.title("Distribution of length of mails")
    plt.ylabel("Count")
    plt.xlabel("Mail Length")
    plt.show()
    
    #### Compare length of ham emails using boxplot
    
    plt.boxplot(ham_emails.mail_length)
    plt.ylabel("Length of Ham Mail")
    plt.xlabel("Ham") 
   
    
     #### Compare length of spam emails using boxplot
   
    plt.boxplot(spam_emails.mail_length)
    plt.ylabel("Length of Spam Mail")
    plt.xlabel("Spam") 
    
    # print mean length of emails
    print(np.mean(ham_emails.mail_length[ham_emails.mail_length<10000]))
    print(np.mean(spam_emails.mail_length[spam_emails.mail_length<10000]))
    
   
    # spam mails are longer than ham mails
    print(ham_emails.describe())    
    print(spam_emails.describe())

    
def findModelPerfomanceMetric(test_set,model,modelName):  
  predicted = model.predict(test_set.emails)
  f1= metrics.f1_score(test_set.target,predicted)  
  fpr, tpr, thresholds = metrics.roc_curve(test_set.target,predicted,pos_label =1)
  auc= metrics.auc(fpr, tpr)
  ## Plot ROC Curve for classifier
  if (modelName == 'WordsNLength' ):
      plt.title("Receiver Operating Characteristic Curve")  
      plt.plot(fpr,tpr)
      plt.ylabel("True Positive Rate")
      plt.xlabel("False Positive Rate")
      plt.show()
  #print(metrics.confusion_matrix(test_set.target,predicted))
  # save the model and accuracy details for later investigration
  accuracyDFTemp = pd.DataFrame([(modelName,f1,auc)],columns= ['ModelName','F1Score','AUC']) 
  return accuracyDFTemp
  
   

def saveModel(nameOfFile,model):
    file = open(nameOfFile,'wb') # w= writing, b= binary
    pickle.dump(model,file)
    file.close()



def buildModel1(train_set1,test_set1):    
    modelName = 'Logistic Regression'
    clf_pipe1 = Pipeline([
            ('vect', CountVectorizer()),            
            ('clf', LogisticRegression()) ])    
    clf_pipe1.fit(train_set1.emails, train_set1.target)
    accuracy =findModelPerfomanceMetric(test_set1,clf_pipe1,modelName)
    return accuracy


def buildModel2(train_set,test_set):
    modelName = 'Naive Bayes'
    clf_pipe = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()) ])
    clf_pipe.fit(train_set.emails, train_set.target) 
    disp = plot_precision_recall_curve(clf_pipe, test_set.emails, test_set.target)
    disp.ax_.set_title('2-class Precision-Recall curve:')
    accuracy = findModelPerfomanceMetric(test_set,clf_pipe,modelName)
    return accuracy

    
def buildModel3(train_set,test_set): 
    modelName = 'Support Vector Classifier'
    clf_pipe = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf',  SVC(kernel='linear'))])
    clf_pipe.fit(train_set.emails, train_set.target) 
    accuracy=findModelPerfomanceMetric(test_set,clf_pipe, modelName)
    return accuracy

def buildModel4(train_set,test_set):
    modelName = 'Decision Tree Classifier'
    clf_pipe = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf',  DecisionTreeClassifier())])
    clf_pipe.fit(train_set.emails, train_set.target) 
    accuracy= findModelPerfomanceMetric(test_set,clf_pipe,modelName)
    return accuracy

# define model by adding extra feature length
def getMailLength(train_set):
    return(np.array([len(mail) for mail in train_set]).reshape(-1,1))

def buildModel5(train_set,test_set):
    modelName = 'Random Forest Classifier'
    clf_pipe = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf',  RandomForestClassifier(n_estimators=10))])
    clf_pipe.fit(train_set.emails, train_set.target) 
    accuracy= findModelPerfomanceMetric(test_set,clf_pipe,modelName)
    return accuracy
    
def visualiseModelAccuracy(accuracyDF):
    ax1= accuracyDF['F1Score'].plot(kind='bar')
    ax1.set_ylabel("Model F1-Score")  
    ax1.set_xticklabels(accuracyDF['ModelName'])
    ax1.set_title("Model Comparison Based on F1Score")
    plt.show()
  
    ax2= accuracyDF['AUC'].plot(kind='bar')
    ax2.set_ylabel("Model AUC")  
    ax2.set_xticklabels(accuracyDF['ModelName'])
    ax2.set_title("Model Comparison Based on Area Under Curve")
    plt.show()
    
main()
