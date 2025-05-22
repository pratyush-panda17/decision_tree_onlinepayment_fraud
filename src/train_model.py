import pandas as pd
import numpy as np
import math


class decision_tree:
    def __init__(self,target,categories,impurity_function = "Gini Index",max_depth = 25,min_samples=15,min_impurity = 0.000000000000001):

        assert target.ndim==1, "Target array must be one dimensional"
        assert impurity_function == "Gini Index" or "Entropy" or "Classification", "Available impurity functions:'Gini Index','Entropy', 'Classification'"

        self.left = None
        self.right = None

        self.target = target #a one dimensional numpy array storing all the target values

        self.impurity_fun_name = impurity_function #impurity function chosen by the user. Default is the Gini Index
        self.impurity_function = decision_tree.giniIndex if impurity_function == "Gini Index" else decision_tree.classification_error if impurity_function == "Classification" else decision_tree.entropy

        self.label = np.bincount(target).argmax() #label attached to each class
        self.isLeafNode = True #every node at the time of initialization is a leaf node
        self.splittingCondition = None #splitting condition is defined in the grow function
        self.impurity = self.impurity_function(target,categories) 
        self.N = target.shape[0] #number of target values
        self.categories = categories #a list containing each category in the target array, to be provided by the user

        self.max_depth = max_depth #max depth, default value is 25
        self.min_samples = min_samples #minimum samples in each node to avoid overfitting
        self.minImpurity = min_impurity #minimum impurity for each node
    
    def giniIndex(data, categories): #Calculates the gini index given a numpy array and a list of  categories
        assert data.ndim==1, "Target array must be one dimensional"
        n = data.shape[0]
        if n==0:
            return 0
        gini_sum = 0
        for category in categories:
            frequency =np.sum(data==category)
            gini_sum+= (frequency/n)**2
        
        return 1 - gini_sum
    
    def entropy(data,categories): #Calculates the entropy given a numpy array and a list of  categories
        assert data.ndim==1, "Target array must be one dimensional"
        n = data.shape[0]
        if n==0:
            return 0
        entropy = 0
        for category in categories:
            prob = np.sum(data==category)/n
            sum+=(prob * math.log(prob,2))
        return -sum
    
    def classification_error(data,categories): #Calculates the classification error given a numpy array and a list of  categories
        assert data.ndim==1, "Target array must be one dimensional"
        n = data.shape[0]
        if n==0:
            return 0
        maximum = 0
        for category in categories:
            prob = np.sum(data==category)/n
            if prob>maximum:
                maximum = prob
        return 1-maximum
    
    def split(data,split_condition): #splits the pandas based on the column and value
        ld = data[data[split_condition[0]] == split_condition[1]]
        rd = data[data[split_condition[0]] != split_condition[1]]
        return ld,rd
    
    def checkCondition(row,split_condition): #checks if a particular data point meets a split condition. row refers to a panda series object that holds the information of one pandas row
        return row.loc[split_condition[0]]==split_condition[1]

    def find_best_split(self,data,split_conditions): #finds the best split based on a predicided list of splitting conditions
        information_gain = 0
        best_split_condition = None
        l_data,r_data = None,None
        for split_condition in split_conditions:
            ld,rd = decision_tree.split(data,split_condition)
            t1 = ld.iloc[:,-1].to_numpy() #getting target variable arrays to calculate impurity
            t2 = rd.iloc[:,-1].to_numpy()

            assert(len(t1)+len(t2)==self.N) #assures the split occured correctly

            if len(t1)==0 or len(t2)==0:
                continue #incase we use the same split on one tree path

            w1 = (len(t1)/self.N)
            w2 = (len(t2)/self.N)
            ig = (w1 * self.impurity_function(t1,self.categories)) + (w2 * self.impurity_function(t2,self.categories)) #weighted sum of impurities of child nodes
            ig = self.impurity - ig
            if ig>information_gain:
                information_gain = ig
                best_split_condition = split_condition
                l_data = ld
                r_data = rd
        return best_split_condition,l_data,r_data
    
    def grow(self,data,split_conditions,depth=0):
        if not (depth>=self.max_depth or self.N<=self.min_samples or self.impurity<=self.minImpurity): #pre-pruning conditions
            best_split,l_data,r_data = self.find_best_split(data,split_conditions)
            if best_split != None :
                self.isLeafNode = False #since this node is being split it is no longer a leaf node
                self.splittingCondition = best_split #assures all non leaf nodes have splitting conditions
                self.left = decision_tree(l_data.iloc[:,-1].to_numpy(),self.categories,self.impurity_fun_name) #
                self.right = decision_tree(r_data.iloc[:,-1].to_numpy(),self.categories,self.impurity_fun_name)
                self.left.grow(l_data,split_conditions,depth+1) #recursively growing the left and right trees
                self.right.grow(r_data,split_conditions,depth+1)
        
    
    def classify(self,row): #classifies based on the tree splitting conditions. Takes as input a pandas series object that represents one row of a dataframe
        if self.isLeafNode == True:
            return self.label #leaf nodes are the classifying nodes
        elif decision_tree.checkCondition(row,self.splittingCondition):
            return self.left.classify(row)
        return self.right.classify(row)
    
    def accuracy(self,data,targets): #accuracy given a test data set
        assert targets.ndim ==1 
        total = 0
        for i,target in enumerate(targets):
            if self.classify(data.iloc[i]) == target:
                total+=1
        return (total/targets.size) * 100

    def true_positives(self,data,targets):
        assert targets.ndim ==1
        total = 0
        for i,target in enumerate(targets):
            if self.classify(data.iloc[i]) == target and target==1:
                total+=1
        return total
    
    def false_positives(self,data,targets):
        assert targets.ndim ==1
        total = 0
        for i,target in enumerate(targets):
            if self.classify(data.iloc[i]) != target and target==0:
                total+=1
        return total
    
    def true_negatives(self,data,targets):
        assert targets.ndim ==1
        total = 0
        for i,target in enumerate(targets):
            if self.classify(data.iloc[i]) == target and target==0:
                total+=1
        return total
    
    def false_negatives(self,data,targets):
        assert targets.ndim ==1
        total = 0
        for i,target in enumerate(targets):
            if self.classify(data.iloc[i]) != target and target==1:
                total+=1
        return total
    
    def precision(self,data,targets):
        tp = self.true_positives(data,targets)
        fp = self.false_positives(data,targets)
        return (tp/(tp+fp))*100
    
    def recall(self,data,targets):
        tp = self.true_positives(data,targets)
        fn = self.false_negatives(data,targets)
        return (tp/(tp+fn))*100
    
    def f1_score(self,data,targets):
        precision = self.precision(data,targets)
        recall = self.recall(data,targets)
        return (2*precision*recall)/(precision + recall)
    
    def getConfusionMatrix(self,data,targets):
        return [[self.true_negatives(data,targets),self.false_positives(data,targets)],
                [self.false_negatives(data,targets),self.true_positives(data,targets)]]









