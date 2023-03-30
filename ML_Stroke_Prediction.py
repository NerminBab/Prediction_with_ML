"""
According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.

1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient

A stroke is a medical emergency, and prompt treatment is crucial. Early action can reduce brain damage and other complications.

* Is there an explanation for the stroke?
* Are the elderly and smokers more likely to have a stroke?
* Or is smoking not a factor in having a stroke?
* Are those with high workload and stress more likely to have a stroke?

Let's see together, is it true?

"""

# First of all, import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import imblearn as imb

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, \
                                  RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC as Model
from imblearn.pipeline import Pipeline as imbalancedPipeline
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Read Data:
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.head()

# The id column is not relevant
df.drop(columns=['id'], inplace=True)


# Target variable analysis:
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': round(100 * (dataframe[col_name].value_counts()) / len(dataframe), 2)}))

    if plot:
        sns.countplot(x=col_name, data=dataframe)
        plt.show()

cat_summary(df, 'stroke', plot=True)
#       stroke    Ratio
#      0    4861  95.130
#      1     249  4.870
# Imbalanced classification

# Data Overview:
def check_data(dataframe,head=5):
    print(20*"-" + "Information".center(20) + 20*"-")
    print(dataframe.info())
    print(20*"-" + "Data Shape".center(20) + 20*"-")
    print(dataframe.shape)
    print("\n" + 20*"-" + "The First 5 Data".center(20) + 20*"-")
    print(dataframe.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(dataframe.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print(dataframe.isnull().sum())
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe().T)
check_data(df)


# visualization of categorical variables
fig,axes = plt.subplots(4,2,figsize = (16,16))
sns.set_style('darkgrid')
fig.suptitle("Count plot for various categorical features")

sns.countplot(ax=axes[0,0],data=df,x='gender')
sns.countplot(ax=axes[0,1],data=df,x='hypertension')
sns.countplot(ax=axes[1,0],data=df,x='heart_disease')
sns.countplot(ax=axes[1,1],data=df,x='ever_married')
sns.countplot(ax=axes[2,0],data=df,x='work_type')
sns.countplot(ax=axes[2,1],data=df,x='Residence_type')
sns.countplot(ax=axes[3,0],data=df,x='smoking_status')
sns.countplot(ax=axes[3,1],data=df,x='stroke')
plt.show()


# Categorical and Numerical Variables:
def grab_col_names(dataframe, cat_th=4, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
print(f'Categorical Variable: {cat_cols}')  # ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'hypertension', 'heart_disease', 'stroke']
print(f'Numerical Variable: {num_cols}') # ['age', 'avg_glucose_level', 'bmi']

only_2_unique = ['ever_married', 'Residence_type'] # object variables
no_op_cols = ['hypertension', 'heart_disease']
new_cat_cols = ['gender', 'work_type', 'smoking_status']
new_num_cols = ["age", "avg_glucose_level", "bmi"]


# Missing value handle and Standadization:
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())])

# OrdinalEncoder:
binary_transformer = Pipeline(steps=[
    ('binary', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

# OneHotEncoder:
non_binary_transformer = Pipeline(steps=[
    ('non_binary', OneHotEncoder(handle_unknown='ignore'))])

# Columns that will not be processed:
no_op_transformer = Pipeline(steps=[
    ('no_op', FunctionTransformer(lambda x: x))])


preprocessing_pipeline = ColumnTransformer(transformers=
[('numeric', numeric_transformer, new_num_cols),
('binary', binary_transformer, only_2_unique),
('non_binary', non_binary_transformer, new_cat_cols),
('no_op', no_op_transformer, no_op_cols)])


# SMOTE:
model_pipeline = imbalancedPipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('smote', SMOTE()),
    ('model', Model())])

features = df.drop('stroke', axis=1)
target = df['stroke']

# Train-Test split:
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# Cross-validation:
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=skf,
                            scoring="f1_macro")


for i, score in enumerate(cv_scores):
    print(f"Fold {i + 1} score: {score * 100:.2f}%")
# Fold 1 score: 55.53%
# Fold 2 score: 50.55%
# Fold 3 score: 52.39%
# Fold 4 score: 53.76%
# Fold 5 score: 52.47%

print(f"Ortalama F1-Skoru : {cv_scores.mean() * 100:.2f}%")
print(f"Standart Sapma: {cv_scores.std() * 100:.2f}%")
# Ortalama F1-Skoru : 52.94%
# Standart Sapma: 1.65%


model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize = (12, 8))
    sns.heatmap(confusion_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                linewidths=10,
                annot_kws={'size': 20}, cbar=False)

    plt.title('Confusion Matrix', size=18)
    plt.xticks([0.5, 1.5], ['Predicted Normal', 'Predicted Stroke'], size=14, rotation=25)
    plt.yticks([0.5, 1.5], ['Actual Normal', 'Actual Stroke'], size=14, rotation=25)
    plt.xlabel('Predicted Label', size=14)
    plt.ylabel('Actual Label', size=14)
    plt.show()
plot_confusion_matrix(confusion_matrix(y_test, y_pred))


print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support
#            0       0.96      0.82      0.88       960
#            1       0.14      0.44      0.21        62
#     accuracy                           0.80      1022
#    macro avg       0.55      0.63      0.55      1022
# weighted avg       0.91      0.80      0.84      1022
