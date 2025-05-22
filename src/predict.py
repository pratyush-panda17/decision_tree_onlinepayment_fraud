from train_model import decision_tree
import pickle
import csv
import data_preprocessing
from data_preprocessing import SPLIT_CONDITIONS,CATEGORIES #For preprocessing tools

def createMetricsFile(model,data,targets): 
    tp = model.true_positives(data,targets)
    tn = model.true_negatives(data,targets)
    fp = model.false_positives(data,targets)
    fn = model.false_negatives(data,targets)


    precision = (tp/(tp+fp))*100
    recall = (tp/(tp+fn))*100
    f1_score = (2*precision*recall)/(precision + recall)

    metrics = open("./results/metrics.txt",'w')
    metrics.write("Classification Metrics: \n")

    metrics.write(f"Accuracy: {round(model.accuracy(data,targets),2)} \n")
  

    metrics.write(f"Precision: {round(precision,2)} \n")


    metrics.write(f"Recall: {round(recall,2)} \n")


    metrics.write(f"F1-Score: {round(f1_score,2)} \n")


    metrics.write(f"Confusion Matrix: \n")
    metrics.write(f"[[{round(tn,2)},{round(fp,2)}], \n [{round(fn,2)},{round(tp,2)}]]")


    metrics.close()

def createPredictionsCsv(model,data):
    predictions = [model.classify(data.iloc[i]) for i in range(len(data))]
    with open('./results/predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for prediction in predictions:
            writer.writerow([prediction])






DATA = data_preprocessing.getData()
training_data,training_targets,test_data,test_targets =data_preprocessing.test_training_split(DATA,80)
with open("./models/decision_tree_final.pkl", 'rb') as file:
    tree =pickle.load(file)

createMetricsFile(tree,test_data,test_targets)
print("created metrics file")
createPredictionsCsv(tree,test_data)
print("created csv file")