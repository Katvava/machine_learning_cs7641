import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_score, accuracy_score, average_precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.base import clone
import matplotlib.pyplot as plt
import time


# load credit data
def load_credit_card_data(data_path, sample_rate=0.7):
    dataset = pd.read_csv(data_path)
    dataset = dataset.sample(n=int(len(dataset)*sample_rate), random_state=np.random.RandomState(0))

    dataset['logAmount'] = np.log(dataset['Amount']+1)
    # dataset['logAmount'].sort_values().plot.hist()
    dataset['normAmount'] = StandardScaler().fit_transform(dataset['Amount'].values.reshape (-1,1))
    dataset = dataset.drop (['Time', 'Amount','logAmount'], axis = 1);

    # get the data and labels 
    X = dataset.iloc[:, dataset.columns != 'Class']
    y = dataset.iloc[:, dataset.columns == 'Class']
    len(y[y.Class ==1])
    return X, y


#  load hotel booking data
def load_hotel_booking_data(data_path, sample_rate=0.5):
    data = pd.read_csv(data_path)
    data = data.sample(n=int(len(data)*sample_rate), random_state=np.random.RandomState(0))

    # print(data.isnull().sum())

    # drop company
    # data = data.drop(['company'], axis = 1)

    data['children'] = data['children'].fillna(0)
    data.drop(columns=['company','reservation_status_date'], inplace=True)
    data.country.fillna('PRT' , inplace=True)
    data.children.fillna(0.0 , inplace= True)
    data.agent.fillna(int(data['agent'].mean()), inplace=True)

    # data imputation
    """
    data['country'].fillna(data['country'].mode()[0], inplace = True)
    data['agent'].fillna(data['agent'].mode()[0], inplace = True)
    data['company'].fillna(data['company'].mode()[0], inplace = True)
    data['children'].fillna(data['children'].mode()[0], inplace=True)
    data['hotel'].fillna(data['hotel'].mode()[0], inplace=True)

    data['arrival_date_month'].fillna(data['arrival_date_month'].mode()[0],  inplace=True)
    data['arrival_date_week_number'].fillna(data['arrival_date_week_number'].mode()[0],  inplace=True)
    data['arrival_date_day_of_month'].fillna(data['arrival_date_day_of_month'].mode()[0],  inplace=True)
    data['market_segment'].fillna(data['market_segment'].mode()[0],  inplace=True)
    data['distribution_channel'].fillna(data['distribution_channel'].mode()[0],  inplace=True)
    data['is_repeated_guest'].fillna(data['is_repeated_guest'].mode()[0],  inplace=True)
    data['assigned_room_type'].fillna(data['assigned_room_type'].mode()[0],  inplace=True)
    data['deposit_type'].fillna(data['deposit_type'].mode()[0],  inplace=True)
    data['customer_type'].fillna(data['customer_type'].mode()[0],  inplace=True)
    """

    qualitative_data = pd.DataFrame()
    qualitative_data['is_canceled'] = data['is_canceled']
    qualitative_data['hotel'] = data['hotel'].map({'Resort Hotel':1, 'City Hotel':0})
    qualitative_data['arrival_date_month_new'] = data['arrival_date_month'].map({'July':6, 'August':7, 'September':8, 'October':9, 'November':10, 'December':11,
        'January':0, 'February':1, 'March':2, 'April':3, 'May':4, 'June':5})
    qualitative_data['arrival_date_week_number'] = data['arrival_date_week_number']
    qualitative_data['arrival_date_day_of_month'] = data['arrival_date_day_of_month']
    qualitative_data['market_segment'] = data['market_segment'].map({'Direct':0, 'Corporate':1, 'Online TA':2, 'Offline TA/TO':3,
        'Complementary':4, 'Groups':5, 'Undefined':6, 'Aviation':7})
    qualitative_data['distribution_channel'] = data['distribution_channel'].map({'Direct':0, 'Corporate':1, 'TA/TO':2, 'Undefined':3, 'GDS':4})
    qualitative_data['is_repeated_guest'] = data['is_repeated_guest']
    qualitative_data['assigned_room_type'] = data['assigned_room_type'].map({'C':0, 'A':1, 'D':2, 'E':3, 'G':4, 'F':5, 'I':10, 'B':8, 'H':6, 'L':7, 'K':11, 'P':9})
    qualitative_data['deposit_type'] = data['deposit_type'].map({'No Deposit':0, 'Refundable':1, 'Non Refund':2})
    qualitative_data['customer_type'] = data['customer_type'].map({'Transient':0, 'Contract':1, 'Transient-Party':2, 'Group':3})
    # print(qualitative_data.isnull().sum())
    qualitative_data.reset_index(inplace=True)

    quantitative_data = pd.DataFrame()
    quantitative_data['lead_time'] = data['lead_time']
    quantitative_data['stays_in_weekend_nights'] = data['stays_in_weekend_nights']
    quantitative_data['stays_in_week_nights'] = data['stays_in_week_nights']
    quantitative_data['adults'] = data['adults']
    quantitative_data['children'] = data['children']
    quantitative_data['babies'] = data['babies']
    quantitative_data['previous_cancellations'] = data['previous_cancellations']
    quantitative_data['previous_bookings_not_canceled'] = data['previous_bookings_not_canceled']
    quantitative_data['booking_changes'] = data['booking_changes']
    quantitative_data['days_in_waiting_list'] = data['days_in_waiting_list']
    quantitative_data['adr'] = data['adr']
    quantitative_data['required_car_parking_spaces'] = data['required_car_parking_spaces']
    quantitative_data['total_of_special_requests'] = data['total_of_special_requests']

    quantitative_data_minmaxscaler = pd.DataFrame(MinMaxScaler().fit_transform(quantitative_data))
    quantitative_data_minmaxscaler.columns = quantitative_data.columns
    # print(quantitative_data_minmaxscaler.isnull().sum())
    quantitative_data_minmaxscaler.reset_index(inplace=True)

    combined_data = pd.concat([qualitative_data, quantitative_data_minmaxscaler], axis=1)
    # print(combined_data.isnull().sum())
    # get the data and labels 
    X = combined_data.iloc[:, combined_data.columns != 'is_canceled']
    y = combined_data.iloc[:, combined_data.columns == 'is_canceled']
    return X,  y


def clean_dataset(dx, dy):
    assert isinstance(dx, pd.DataFrame), "df needs to be a pd.DataFrame"
    dx.dropna(inplace=True)
    indices_to_keep = ~dx.isin([np.nan, np.inf, -np.inf]).any(1)
    return dx[indices_to_keep],  dy[indices_to_keep]


def learning_Curve(train_X_all, train_y_all, test_x_all, test_y_all, classifier, metric, step=20000):

    train_metric, val_metric, indices, training_time, testing_time = [], [], [], [], []
    
    for m in range(step, len(train_X_all), step):
        train_X = train_X_all[:m]
        train_y = train_y_all[:m]
        start_training = time.time()
        classifier.fit(train_X, train_y)
        training_time.append(time.time() -  start_training)
        # print("training {} samples took {}".format(m, time.time() -  start_training))

        train_y_pred = classifier.predict(train_X)
        start_testing  = time.time()
        test_y_all_pred = classifier.predict(test_x_all)
        testing_time.append(time.time() - start_testing)

        print(m)
        # print(test_y_all, test_y_all_pred)
        print(classification_report(test_y_all, test_y_all_pred))

        test_avg = metric(test_y_all, test_y_all_pred)
        train_avg = metric(train_y, train_y_pred)

        val_metric.append(test_avg)
        train_metric.append(train_avg)
        indices.append(m)
    
    return train_metric, val_metric, indices, training_time, testing_time


def plot_learning_curve(train_scores, val_scores, indices, title, inverse_x=False):
    plt.plot(indices, train_scores , "r-+", linewidth=2, label="train")
    plt.plot(indices, val_scores, "b-+", linewidth=2, label="validation")
    plt.title(title)
    plt.legend()
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    if inverse_x:
        plt.gca().invert_xaxis()


def plot_learning_curve_time(train_time, val_time, indices, title, inverse_x=False):
    plt.plot(indices, train_time , "r-+", linewidth=2, label="train time")
    plt.plot(indices, val_time, "b-+", linewidth=2, label="testing time")
    plt.title(title)
    plt.legend()
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    if inverse_x:
        plt.gca().invert_xaxis()


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


def compare_fit_time(n,NNtime, SMVtime, kNNtime, DTtime, BTtime, title):
    plt.figure()
    plt.title("Model Training Times: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model Training Time (s)")
    plt.plot(n, NNtime, '-', color="b", label="Neural Network")
    plt.plot(n, SMVtime, '-', color="r", label="SVM")
    plt.plot(n, kNNtime, '-', color="g", label="kNN")
    plt.plot(n, DTtime, '-', color="m", label="Decision Tree")
    plt.plot(n, BTtime, '-', color="k", label="Boosted Tree")
    plt.legend(loc="best")
    plt.show()
    
def compare_pred_time(n,NNpred, SMVpred, kNNpred, DTpred, BTpred, title):
    plt.figure()
    plt.title("Model Prediction Times: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model Prediction Time (s)")
    plt.plot(n, NNpred, '-', color="b", label="Neural Network")
    plt.plot(n, SMVpred, '-', color="r", label="SVM")
    plt.plot(n, kNNpred, '-', color="g", label="kNN")
    plt.plot(n, DTpred, '-', color="m", label="Decision Tree")
    plt.plot(n, BTpred, '-', color="k", label="Boosted Tree")
    plt.legend(loc="best")
    plt.show()


def compare_learn_time(n,NNlearn, SMVlearn, kNNlearn, DTlearn, BTlearn, title):
    plt.figure()
    plt.title("Model Learning Rates: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.plot(n, NNlearn, '-', color="b", label="Neural Network")
    plt.plot(n, SMVlearn, '-', color="r", label="SVM")
    plt.plot(n, kNNlearn, '-', color="g", label="kNN")
    plt.plot(n, DTlearn, '-', color="m", label="Decision Tree")
    plt.plot(n, BTlearn, '-', color="k", label="Boosted Tree")
    plt.legend(loc="best")
    plt.show()