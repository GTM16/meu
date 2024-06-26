from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Carregar o CSV e preprocessar os dados
def load_and_preprocess():
    dataframe = pd.read_csv('BankChurners.csv', encoding='latin1', sep=',')
    dataframe.drop(["CLIENTNUM","Attrition_Flag","Total_Revolving_Bal","Total_Amt_Chng_Q4_Q1","Avg_Open_To_Buy","Total_Trans_Amt","Total_Trans_Ct","Total_Ct_Chng_Q4_Q1","Avg_Utilization_Ratio","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2","Months_on_book","Total_Relationship_Count","Months_Inactive_12_mon","Contacts_Count_12_mon"], axis=1, inplace=True)
    
    ordinal = OrdinalEncoder()
    for column in ['Card_Category', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Category']:
        dataframe[column] = ordinal.fit_transform(dataframe[column].values.reshape(-1,1))

    scaler = MinMaxScaler()
    dataframe["Customer_Age"] = scaler.fit_transform(dataframe[['Customer_Age']])

    return dataframe

dataframe = load_and_preprocess()

x = dataframe.drop(['Card_Category'], axis=1)
y = dataframe['Card_Category']
label_encoder = LabelEncoder()
y_num = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_num, test_size=0.20)
model_tree = DecisionTreeClassifier(max_depth=6)
model_tree.fit(x_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])
    input_data = preprocess_input(input_data)
    prediction = model_tree.predict(input_data)
    category = label_encoder.inverse_transform(prediction)
    return jsonify({'prediction': category[0]})

def preprocess_input(input_data):
    ordinal = OrdinalEncoder()
    for column in ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category']:
        input_data[column] = ordinal.fit_transform(input_data[column].values.reshape(-1,1))
    
    scaler = MinMaxScaler()
    input_data["Customer_Age"] = scaler.fit_transform(input_data[['Customer_Age']])
    
    return input_data

@app.route('/add', methods=['POST'])
def add_entry():
    data = request.json
    new_entry = pd.DataFrame([data])
    new_entry = preprocess_input(new_entry)
    global dataframe
    dataframe = dataframe.append(new_entry, ignore_index=True)
    dataframe.to_csv('/mnt/data/BankChurners.csv', index=False)
    return jsonify({'status': 'success'})

@app.route('/chart')
def chart():
    categories = dataframe['Card_Category'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=categories.index, y=categories.values, ax=ax)
    plt.xlabel('Card Category')
    plt.ylabel('Count')
    plt.title('Number of Cards by Category')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return f'<img src="data:image/png;base64,{plot_url}"/>'

if __name__ == '__main__':
    app.run(debug=True)

