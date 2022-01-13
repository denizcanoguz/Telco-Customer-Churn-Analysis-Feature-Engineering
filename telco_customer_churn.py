import numpy as np
import pandas as pd
import seaborn as sns
from _plotly_utils import png
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
# console-setup
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Veri Setini Getirelim.
def load_telco_customer_churn():
    data = pd.read_csv("datasets/Telco-Customer-Churn.csv")
    return data
df_ = load_telco_customer_churn()
df = df_.copy()
df.head()

# Telco Churn Feature Engineering
# ----------------------------------------------------------------------------------------------------------------------
# İş Problemi
# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
# geliştirilmesi istenmektedir.  Modeli geliştirmeden önce gerekli olan veri analizi
# ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.
# ----------------------------------------------------------------------------------------------------------------------
# Veri Seti Hikayesi
# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043
# müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali
# bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin
# hizmetlerinden ayrıldığını,kaldığını veya hizmete kaydolduğunu gösterir.
# Telco-Customer-Churn-Variables.png !!
# ----------------------------------------------------------------------------------------------------------------------
# Proje Görevleri
# ----------------------------------------------------------------------------------------------------------------------
# Görev 1 : Keşifçi Veri Analizi
# ----------------------------------------------------------------------------------------------------------------------
# Adım 1: Genel resmi inceleyiniz.
df.info()
def check_df(dataframe, head=5, tail=5, quan=False):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    if quan:
        print("##################### Quantiles #####################")
        print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)

df.columns = [col.upper() for col in df.columns]
# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=3, car_th=10):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

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


# Adım 3:  Numerik ve kategorik değişkenlerin analizini yapınız.
df['TOTALCHARGES'].replace([' '], '0.0', inplace=True)
df["TOTALCHARGES"] = df["TOTALCHARGES"].astype(float)
df["CHURN"].replace(["Yes"], "1", inplace=True)
df["CHURN"].replace(["No"], "0", inplace=True)
df["CHURN"] = df["CHURN"].astype(int)
# Numerik değişken analizi
num_cols
# ['TENURE', 'MONTHLYCHARGES']
def num_summary(dataframe, numerical_col, plot=True):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col)


# Kategorik değişken analizi
cat_cols
# ['GENDER','PARTNER','DEPENDENTS','PHONESERVICE','MULTIPLELINES','INTERNETSERVICE',
#  'ONLINESECURITY','ONLINEBACKUP','DEVICEPROTECTION','TECHSUPPORT','STREAMINGTV',
#  'STREAMINGMOVIES','CONTRACT','PAPERLESSBILLING','PAYMENTMETHOD','CHURN','SENIORCITIZEN']

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
for col in cat_cols:
    cat_summary(df, col)

df.isnull().sum().sum()
# Adım 4: Hedef değişken analizi yapınız.
# (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
def target_analyser(dataframe, target, num_cols, cat_cols):
    for col in dataframe.columns:
        if col in cat_cols:
            print(col, ":", len(dataframe[col].value_counts()))
            print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                                "RATIO": dataframe[col].value_counts() / len(dataframe),
                                "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
        if col in num_cols:
            print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(target)[col].mean()}), end="\n\n\n")
target_analyser(df, "CHURN", num_cols, cat_cols)
# Adım 5: Aykırı gözlem analizi yapınız.
def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
for col in num_cols:
    print(col, check_outlier(df, col))


# Adım 6: Eksik gözlem analizi yapınız.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
missing_values_table(df)


# Adım 7: Korelasyon analizi yapınız.

cor = df.corr(method='pearson')
cor
sns.heatmap(cor)
plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# Görev 2 : Feature Engineering
# ----------------------------------------------------------------------------------------------------------------------
# Adım 1:  Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

missing_values_table(df)
# Eksik Gözlem bulunmamakta

for col in num_cols:
    print(col, check_outlier(df, col))
# aykırı gözlem bulunmamakta

# Adım 2: Yeni değişkenler oluşturunuz.
# ----------------------------------------------------------------------------------------------------------------------

df.head()
df = df.drop(columns="CUSTOMERID")
df = df.drop(columns="SENIORCITIZEN") # anlam ifade etmiyor
df["Number_AdditionalServices"] = (df[["ONLINESECURITY","DEVICEPROTECTION",
                                       "STREAMINGMOVIES","TECHSUPPORT","STREAMINGTV",
                                       "ONLINEBACKUP"]] == "Yes").sum(axis=1)


# Adım 3:  Encoding işlemlerini gerçekleştiriniz.
# ----------------------------------------------------------------------------------------------------------------------
cat_cols, num_cols, cat_but_car = grab_col_names(df)
# RARE  analizimizi yapalım
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
rare_analyser(df, "CHURN", cat_cols)

# Label Encoding for identified columns.
features_le = ["GENDER","PARTNER","DEPENDENTS","PHONESERVICE","PAPERLESSBILLING"]
def label_encoding(features,df):
    for i in features:
        df[i] = df[i].map({"Yes":1, "No":0})
    return
label_encoding(["PARTNER","DEPENDENTS","PHONESERVICE","PAPERLESSBILLING"],df)
df["GENDER"] = df["GENDER"].map({"Female":1,"Male":0})


# One Hot Encoding
features_ohe = ["MULTIPLELINES","INTERNETSERVICE","ONLINESECURITY","ONLINEBACKUP",
                "DEVICEPROTECTION","TECHSUPPORT","STREAMINGTV","STREAMINGMOVIES",
                "CONTRACT","PAYMENTMETHOD","Number_AdditionalServices"]
df = pd.get_dummies(df,columns=features_ohe,drop_first=True)
df.head()



# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
# ----------------------------------------------------------------------------------------------------------------------

scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.head()

# Adım 5: Model oluşturunuz.
# ----------------------------------------------------------------------------------------------------------------------

y = df["CHURN"]
X = df.drop(["CHURN"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test) # 0.8135


# Yeni ürettiğimiz değişkenler ne alemde?
# ----------------------------------------------------------------------------------------------------------------------

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)