import pandas as pd
import sklearn as skl
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import functools

class ModelFactory:
    def __init__(self):
        self.model_book = {}

    def register_model(self,model_name,model_obj):
        self.model_book[model_name] = model_obj

    def get_fitter(self,model_name):
        fitter = self.model_book.get(model_name)
        if not fitter:
            raise ValueError(model_name)
        return fitter

class DataSet:
    def __init__(self,x_train,y_train,x_test,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.fitter = None

    def fit(self,fitter,**kwargs):
        self.fitter = fitter
        self.fitter.set_data(self.x_train,self.y_train,self.x_test,self.y_test)
        self.fitter.fit(**kwargs)

    def get_score(self):
        return self.fitter.get_score()

    def get_confusion_matrix(self,**kwargs):
        return self.fitter.get_confusion_matrix(**kwargs)

def record(record_dic,record_key):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            record_dic[record_key] = result
        return wrapper
    return decorator

class ObjFitter:
    def __init__(self):
        self.fitter = None
        self.fitable = None
        self.score_book = {}
        self.method_name = None

    def fit(self,fitable, method,factory_obj,**kwargs):
        self.method_name = method
        self.fitter = factory_obj.get_fitter(method)
        self.fitable = fitable
        self.fitable.fit(self.fitter,**kwargs)

    def get_score(self):
        @record(self.score_book,self.method_name)
        def wrapper():
            return self.fitable.get_score()
        wrapper()
        return self.score_book[self.method_name]

    def get_confusion_matrix(self,**kwargs):
        return self.fitable.get_confusion_matrix(**kwargs)

    def get_score_book(self):
        return self.score_book

class Fitter_Core:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.y_pred = None

    def set_data(self,x_train,y_train,x_test,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = None
        self.y_pred = None
    def fit(self):
        pass
    def get_score(self):
        pass
    def get_confusion_matrix(self):
        pass

class DT_Fitter(Fitter_Core):
    def __init__(self):
        super().__init__()

    def fit(self,**kwargs):
        self.model = tree.DecisionTreeClassifier(**kwargs)
        self.model.fit(self.x_train,self.y_train)
        self.y_pred = self.model.predict(self.x_test)

    def get_score(self):
        assert self.model , "Model haven't fitted yet, please fit first"
        return self.model.score(self.x_test,self.y_test)

    def get_confusion_matrix(self,**kwargs):
        assert self.model, "Model haven't fitted yet, please fit first"
        return confusion_matrix(self.y_test,self.y_pred,**kwargs)

class RF_Fitter(Fitter_Core):
    def __init__(self):
        super().__init__()

    def fit(self,**kwargs):
        self.model = RandomForestClassifier(**kwargs)
        self.model.fit(self.x_train,self.y_train)
        self.y_pred = self.model.predict(self.x_test)

    def get_score(self):
        assert self.model , "Model haven't fitted yet, please fit first"
        return self.model.score(self.x_test,self.y_test)

    def get_confusion_matrix(self,**kwargs):
        assert self.model, "Model haven't fitted yet, please fit first"
        return confusion_matrix(self.y_test,self.y_pred,**kwargs)

class SVM_Fitter(Fitter_Core):
    def __init__(self):
        super().__init__()

    def fit(self,**kwargs):     # gamma='scale', decision_function_shape='ovo'
        self.model = svm.SVC(**kwargs)
        self.model.fit(self.x_train[:1000],self.y_train.ravel()[:1000])
        self.y_pred = self.model.predict(self.x_test)

    def get_score(self):
        assert self.model , "Model haven't fitted yet, please fit first"
        return self.model.score(self.x_test,self.y_test)

    def get_confusion_matrix(self,**kwargs):
        assert self.model, "Model haven't fitted yet, please fit first"
        return confusion_matrix(self.y_test,self.y_pred,**kwargs)

class DNN_Fitter(Fitter_Core):
    def __init__(self):
        super().__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.history = None
    def set_data(self,x_train,y_train,x_test,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.test_dataset  = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        self.model = None
        self.y_pred = None
        self.history = None
    def fit(self,**kwargs):
        BATCH_SIZE = 64
        SHUFFLE_BUFFER_SIZE = 100
        self.train_dataset = self.train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        self.test_dataset = self.test_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        self.model = tf.keras.Sequential([tf.keras.layers.Dense(128, input_shape=[2]),
                                          tf.keras.layers.Dense(128, activation='relu'),
                                          tf.keras.layers.Dense(100, activation='softmax')])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        self.history = self.model.fit(
            self.train_dataset,
            validation_data=self.test_dataset,
            epochs=2
        )
        self.y_pred = self.model.predict_classes(self.x_test)

    def get_score(self):
        assert self.model, "Model haven't fitted yet, please fit first"
        return self.history.history['sparse_categorical_accuracy'][-1]

    def get_confusion_matrix(self,**kwargs):
        assert self.model, "Model haven't fitted yet, please fit first"
        return confusion_matrix(self.y_test, self.y_pred, **kwargs)



if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import os

    os.chdir(r"D:\Grad3\690\FannieMae-master\FannieMae-master\2010Q1")
    AcquisitionColumnNames = (
        "LOAN_ID", "ORIG_CHN", "Seller.Name",
        "ORIG_RT", "ORIG_AMT", "ORIG_TRM", "ORIG_DTE",
        "FRST_DTE", "OLTV", "OCLTV", "NUM_BO",
        "DTI", "CSCORE_B", "FTHB_FLG", "PURPOSE",
        "PROP_TYP", "NUM_UNIT", "OCC_STAT", "STATE", "ZIP_3",
        "MI_PCT", "Product.Type", "CSCORE_C", "MI_TYPE",
        "RELOCATION_FLG"
    )

    PerformanceColumnNames = (
        "LOAN_ID", "Monthly.Rpt.Prd", "Servicer.Name",
        "LAST_RT", "LAST_UPB", "Loan.Age", "Months.To.Legal.Mat",
        "Adj.Month.To.Mat", "Maturity.Date", "MSA",
        "Delq.Status", "MOD_FLAG", "Zero.Bal.Code",
        "ZB_DTE", "LPI_DTE", "FCC_DTE", "DISP_DT",
        "FCC_COST", "PP_COST", "AR_COST", "IE_COST",
        "TAX_COST", "NS_PROCS", "CE_PROCS", "RMW_PROCS",
        "O_PROCS", "NON_INT_UPB", "PRIN_FORG_UPB_FHFA",
        "REPCH_FLAG", "PRIN_FORG_UPB_OTH", "TRANSFER_FLG"
    )

    acquisition_df = pd.read_csv(
        "Acquisition_2010Q1.txt",
        names=AcquisitionColumnNames,
        header=None,
        sep="|"
    )
    performance_df = pd.read_csv(
        "Performance_2010Q1.txt",
        names=PerformanceColumnNames,
        header=None,
        sep="|"
    )
    DS = set(performance_df['Delq.Status'])
    mapper = {}
    for ds in DS:
        try:
            mapper[ds] = int(ds)
        except:
            mapper[ds] = -1
    performance_df['Delq.Status'] = performance_df['Delq.Status'].map(mapper)
    loans = performance_df.groupby("LOAN_ID", sort=True)['Delq.Status'].max()
    ID_To_Delinq = {}
    for row in loans.iteritems():
        loan_id, delinq = row
        ID_To_Delinq[loan_id] = delinq
    def mapper(row):
        return ID_To_Delinq.get(row["LOAN_ID"], -1)
    acquisition_df['MAX_DELINQ'] = acquisition_df.apply(mapper, axis=1)
    DEL_NOTNAN = acquisition_df["MAX_DELINQ"].notna()
    df = acquisition_df[DEL_NOTNAN]

    DEL_NOTNEG = df['MAX_DELINQ'] >= 0

    df = df[DEL_NOTNEG]
    OLTV = df['OLTV'].notna()
    df = df[OLTV]
    CS = df['CSCORE_B'].notna()
    df = df[CS]

    credit_score  = np.array(df['CSCORE_B'])
    loan_to_value = np.array(df['OLTV'])
    max_delinq    = np.array(df['MAX_DELINQ'])

    X = np.array([credit_score, loan_to_value]).transpose()
    y = np.array([max_delinq]).transpose()
    print(X.shape)
    print(y.shape)

    Total = np.hstack([X, y])
    print(Total.shape)
    np.random.shuffle(Total)

    X = Total[:, :2]
    y = Total[:, 2:]

    print(X.shape)
    print(y.shape)

    prop = 0.8
    train_num = int(prop * len(Total))
    print(f"Train Number: {train_num}")

    X_train, X_test = X[:train_num], X[train_num:]
    y_train, y_test = y[:train_num], y[train_num:]

    print(f"X_Train: {X_train.shape}")
    print(f"X_Test: {X_test.shape}")
    print("=="*10)
    print(f"y_Train: {y_train.shape}")
    print(f"y_Test:  {y_test.shape}")

    class_names = np.unique(y)
##########################################################
factory = ModelFactory()
factory.register_model("dt",DT_Fitter())
data = DataSet(X_train,y_train, X_test, y_test)
final_fitter = ObjFitter()
final_fitter.fit(data,"dt",factory)
final_fitter.get_score()
final_fitter.get_confusion_matrix()

factory.register_model("rf",RF_Fitter())
final_fitter.fit(data,"rf",factory)
final_fitter.get_score()
final_fitter.get_confusion_matrix()

factory.register_model("svm",SVM_Fitter())
final_fitter.fit(data,"svm",factory,gamma='scale', decision_function_shape='ovo')
final_fitter.get_score()
final_fitter.get_confusion_matrix()

factory.register_model("dnn",DNN_Fitter())
final_fitter.fit(data,"dnn",factory)
final_fitter.get_score()
final_fitter.get_confusion_matrix()

final_fitter.get_score_book()

