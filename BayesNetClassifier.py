import sys
import string
from collections import defaultdict
import re
import numpy as np

def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}

def cleaning(review, stopwords):

    # Remove the leading spaces and newline character
    review = review.strip()
    # Convert the characters lowercase
    review = review.lower()
    # Remove the 's from the words
    review = review.replace("'s", '')
    # Split the line into words
    # Remove the punctuation marks from the line
    review = re.sub('[^A-Za-z]+', ' ', review)
    # converting into list
    words = review.split(" ")
    words = [word.strip() for word in words]
    # removing Blank spaces
    words = list(filter(None, words))
    # removing stopwords
    words = [word for word in words if word not in stopwords]
    return words


def generate_word_count(label_dict, review_words):
    # label_dict to create wordcount in each review
    for word in review_words:
        if word not in label_dict.keys():
            label_dict[word] = 1
        else:
            label_dict[word] = label_dict[word] + 1
    return label_dict

def predict(words, deceptive, truthful, count_unique_words, deceptive_word_count, truthful_word_count, deceptive_reviews,
            truthful_reviews,alpha = 0.5):
    deceptive_prob, truthful_prob = 1,1
    deceptive_prob_log, truthful_prob_log = 0, 0
    p_deceptive = deceptive_reviews / (deceptive_reviews + truthful_reviews)
    p_truthful = truthful_reviews / (deceptive_reviews + truthful_reviews)

    # using normal prediction
    if all(word in words for word in deceptive.keys()) and all(word in words for word in truthful.keys()):
        for word in words:
            deceptive_prob *= deceptive[word]
            deceptive_prob_log += np.log(deceptive[word])
            truthful_prob *= truthful[word]
            truthful_prob_log += np.log(truthful[word])
        deceptive_prob_log += np.log(p_deceptive)
        truthful_prob_log += np.log(p_truthful)
        deceptive_prob *= p_deceptive
        truthful_prob *= p_truthful

    else:
        # using Laplace Smoothing
        for word in words:
            #for words not in training data but in test data
            if word not in deceptive.keys():
                #deceptive_prob *= (alpha) / (deceptive_reviews + (alpha * count_unique_words))
                deceptive_prob_log += np.log((alpha) / (deceptive_reviews + (alpha * count_unique_words)))

            if word not in truthful.keys():
                #truthful_prob *= (alpha) / (truthful_reviews + (alpha * count_unique_words))
                truthful_prob_log += np.log((alpha) / (truthful_reviews + (alpha * count_unique_words)))
                continue

           #for words in training data
            if word in deceptive_word_count.keys() and word in truthful_word_count.keys():
                #deceptive_prob *= ((deceptive_word_count[word] + alpha) / (deceptive_reviews + (alpha * count_unique_words)))
                deceptive_prob_log += np.log(((deceptive_word_count[word] + alpha) / (deceptive_reviews + (alpha * count_unique_words))))
                #truthful_prob *= ((truthful_word_count[word] + alpha) / (truthful_reviews + (alpha * count_unique_words)))
                truthful_prob_log += np.log(((truthful_word_count[word] + alpha) / (truthful_reviews + (alpha * count_unique_words))))

    # classifing a given review.
    if deceptive_prob_log>truthful_prob_log:
        return "deceptive"
    else:
        return "truthful"

def classifier(train_data, test_data):
    
    stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                 "yourselves",
                 "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
                 "their",
                 "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is",
                 "are", "was",
                 "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
                 "the",
                 "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
                 "against",
                 "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
                 "down", "in",
                 "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
                 "where", "why",
                 "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
                 "not", "only",
                 "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    #print(len(stopwords))
    # This is just dummy code -- put yours here!
    # for k,v in train_data.items():
    deceptive_word_count = {}
    truthful_word_count = {}
    deceptive_reviews = len([label for label in train_data["labels"] if label == "deceptive"])
    truthful_reviews = len(train_data["labels"]) - deceptive_reviews

   # print(deceptive_reviews,truthful_reviews)
    unique_words = []

    for object,label in zip(train_data["objects"], train_data["labels"]):
        # print(object, label)
        review_words = cleaning(object, stopwords)
        unique_words.extend(review_words)
        if label == "deceptive":
            deceptive_word_count.update(generate_word_count(deceptive_word_count, review_words))
        else:
            truthful_word_count.update(generate_word_count(truthful_word_count, review_words))



    #Remove words with count less than a threshold
    filter_deceptive_word_count = {}
    filter_truthful_word_count = {}

    unique_words_deceptive = 0
    for k in deceptive_word_count:
        #if deceptive_word_count[k] > 0:
            if len(k) > 3:
                filter_deceptive_word_count[k] = deceptive_word_count[k]
                unique_words_deceptive += 1
    unique_words_truthful = 0
    for k in truthful_word_count:
        #if truthful_word_count[k] > 0:
            if len(k) > 3:
                filter_truthful_word_count[k] = truthful_word_count[k]
                unique_words_truthful += 1
    count_unique_words = unique_words_truthful + unique_words_deceptive

    total_words_deceptive = sum(filter_deceptive_word_count.values())
    total_words_truthful = sum(filter_truthful_word_count.values())
    #print(total_words_deceptive, total_words_truthful)
    #Calculate probability of each word
    deceptive = {k:v/total_words_deceptive for k,v in filter_truthful_word_count.items()}
    truthful = {k:v/total_words_truthful for k,v in filter_truthful_word_count.items()}


    predictions = []
    for test_review in test_data["objects"]:
        # print(test_review)
        test_review_words = cleaning(test_review, stopwords)
        predictions.append(predict(test_review_words, deceptive, truthful, count_unique_words, filter_deceptive_word_count,
                                   filter_truthful_word_count, deceptive_reviews, truthful_reviews))


    return predictions#[test_data["classes"][0]] * len(test_data["objects"])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    if(sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

    results= classifier(train_data, test_data_sanitized)

    # calculate accuracy
    correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))