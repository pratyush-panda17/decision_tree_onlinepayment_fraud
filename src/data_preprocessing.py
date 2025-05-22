import pandas as pd

SPLIT_CONDITIONS = [('type','CASH_OUT'),('type','PAYMENT'),('type','TRANSFER'),
                    ('type','CASH_IN'),('type','DEBIT'),('oldbalanceDest',0),
                    ('newbalanceDest',0),('oldbalanceOrg',10000000.00),('oldbalanceOrg',0),
                    ('isFlaggedFraud',1),('amount',10000000),('amount',0),
                    ('newbalanceOrig',0)]

CATEGORIES = [0,1]

def getData():
    data = pd.read_csv('./data/fraud_train.csv')
    data =data.drop("nameOrig",axis = 1)
    data =data.drop('step',axis = 1)
    data = data.drop('nameDest',axis = 1)
    column_order = data.columns.tolist()
    column_order[-1],column_order[-2] = column_order[-2],column_order[-1]
    data = data.reindex(columns = column_order)
    return shuffle_data(data)

def shuffle_data(data):
    return data.sample(frac=1).reset_index(drop=True)

def test_training_split(data,split_percentage):
    index = int((split_percentage/100)*data.shape[0])
    h,t= data[:index],data[index:]  
    return (h,h.iloc[:,-1].to_numpy(),
            t,t.iloc[:,-1].to_numpy())





