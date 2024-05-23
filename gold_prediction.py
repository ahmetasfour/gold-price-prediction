import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, KFold

gold_data = 'gld_price_data.csv'

df = pd.read_csv(gold_data)
df = df.drop(columns=['Date'], axis=1)

df.duplicated().sum()
print("TEKRARLANAN SATIRLAR SAYISI",df.duplicated().sum())
df. drop_duplicates (inplace=True)

print("TEKRERLENEN SATIRLAR ",df.duplicated().sum())

print("=============== RİSK BİLİRLEME =============== \n ")

min_gld = df['GLD'].min()
max_gld = df['GLD'].max()

fix= (max_gld - min_gld) / 4
print(f"RİSK KATMANLARI : {fix} \n")

print("****************************************************************")

very_low= min_gld + fix
low = very_low+fix
high = low +fix


df['risk'] = ''
df.loc[df['GLD'] <= very_low, 'risk'] = '-2'
df.loc[(df['GLD'] > very_low) & (df['GLD'] <= low), 'risk'] = '-1'
df.loc[(df['GLD'] > low) & (df['GLD'] <= high), 'risk'] = '1'
df.loc[df['GLD'] > high  , 'risk'] = '2'


print("===== RANDOM DATASETTEN 5 SATIR GÖSTERME ===== \n ")
print(df.sample(5))

print("=========== NULL DEĞERLER VARSA ============== \n")
print(df.isnull().sum() , "\n")

df['EUR/USD'].fillna(method='bfill', inplace=True)

df['SLV'].fillna(method='ffill', inplace=True)


average_spx = df['SPX'].mean()

df['SPX'].fillna(average_spx, inplace=True)


print("========NİTELİKLER ARSIDEKI ilişkisi======== \n")


correlation = df.corr()

plt.figure(figsize=(9, 9))
heatmap = sns.heatmap(correlation, annot=True, fmt='.1f', cmap='Blues', cbar=True, cbar_kws={'label': 'Correlation'})

heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=8)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=8)

plt.show()



print("============= Y TEST BILIRLEME  ============== \n")


X = df[['SPX', 'USO', 'SLV', 'EUR/USD', 'GLD']]

y = df['risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("y_test values:", set(y_test) ,"\n" )

print("##################################################")
print("                     knn_mode                     ")
print("##################################################\n")


num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=3)

cross_val_results = cross_val_score(knn_model, X_train, y_train, cv=kf, scoring='accuracy')


print(f'Cross-Validation Accuracy: {cross_val_results}')
print(f'Mean Accuracy: {cross_val_results.mean()}')

knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)
print("=============KNN Confusion Matrix:============= \n")

knn_conf_matrix = confusion_matrix(y_test, knn_model.predict(X_test))
print(f"{knn_conf_matrix}\n")

tp = knn_conf_matrix.diagonal()  # True Positive for each class
fp = knn_conf_matrix.sum(axis=0) - tp  # False Positive for each class
tn = knn_conf_matrix.sum() - (tp + fp)  # True Negative for each class
fn = knn_conf_matrix.sum(axis=1) - tp  # False Negative for each class

knn_acc = accuracy_score(y_test, knn_predictions)
knn_f1 = f1_score(y_test, knn_predictions, average='weighted')


print("============ TP , FP , TN , FN ================\n")
print(f"True Positive for each class: {tp}")
print(f"False Positive for each class: {fp}")
print(f"True Negative for each class: {tn}")
print(f"False Negative for each class: {fn} \n")
print("============ ACC , SEN ,PRE , F-SCORE =========== \n")
print(f"Accuracy: {knn_acc}")
print(f"F1-Score: {knn_f1}\n")

print("============= HER ŞEY İÇİN (RAPOR) ============== \n")
report = classification_report(y_test, knn_predictions)
print(report)


print("============= OVERFITING VARSA  ================ \n")

training_accuracies = []
testing_accuracies = []

for k in range(1, 6):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    
    training_accuracy = knn_model.score(X_train, y_train)
    testing_accuracy = knn_model.score(X_test, y_test)
    
    training_accuracies.append(training_accuracy)
    testing_accuracies.append(testing_accuracy)

plt.figure(figsize=(10, 6))

plt.plot(range(1, 5 + 1), training_accuracies, label='Training Accuracy', marker='o')

plt.plot(range(1,5 + 1), testing_accuracies, label='Testing Accuracy', marker='o')

plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Model Accuracy for Different Values of k')
plt.legend()
plt.show()


print("##################################################")
print("             logistic_regression_model            ")
print("##################################################\n")



random_seed = 42

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    X['cluster'] = kmeans.fit_predict(X)
    sse.append(kmeans.inertia_)


# رسم الكوع
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('GRUPLAR SAYISI (k)')
plt.ylabel('DATASET DF sse')
plt.title('Elbow Method')

plt.show()

kmeans = KMeans(n_clusters=4)  

X_train_clustered = X_train.join(X['cluster'])
X_test_clustered = X_test.join(X['cluster'])

logistic_regression_model = LogisticRegression(random_state=42)
logistic_regression_model.fit(X_train_clustered, y_train)

logistic_regression_predictions = logistic_regression_model.predict(X_test_clustered)


print("=====logistic regression Confusion Matrix:====== \n")
logistic_regression_conf_matrix = confusion_matrix(y_test, logistic_regression_predictions)

print(f"{logistic_regression_conf_matrix} \n ")



tp = logistic_regression_conf_matrix.diagonal()  # True Positive for each class
fp = logistic_regression_conf_matrix.sum(axis=0) - tp  # False Positive for each class
tn = logistic_regression_conf_matrix.sum() - (tp + fp)  # True Negative for each class
fn = logistic_regression_conf_matrix.sum(axis=1) - tp  # False Negative for each class

logistic_regression_acc = accuracy_score(y_test, logistic_regression_predictions)
logistic_regression_f1 = f1_score(y_test, logistic_regression_predictions, average='weighted')


# طباعة القيم
print("============ TP , FP , TN , FN ================\n")
print(f"True Positive for each class: {tp}")
print(f"False Positive for each class: {fp}")
print(f"True Negative for each class: {tn}")
print(f"False Negative for each class: {fn} \n")
print("============ ACC , SEN ,PRE , F-SCORE =========== \n")
print(f"Accuracy: {logistic_regression_acc}")

print(f"F1-Score: {logistic_regression_f1}\n")

print("============= HER ŞEY İÇİN (RAPOR) ============== \n")
report = classification_report(y_test, logistic_regression_predictions)

print(report)

print("============= Training ve Testing Accuracy  ============== \n")

training_accuracy = logistic_regression_model.score(X_train.join(X['cluster']), y_train)
print(f"Training Accuracy: {training_accuracy}" )

acc = accuracy_score(y_test, logistic_regression_predictions)
print(f"Testing Accuracy : {logistic_regression_acc} \n")

n=5
print("##################################################")
print("                    SVC MODEL                     ")
print("##################################################\n")
kmeans = KMeans(n_clusters=4)  
X['cluster'] = kmeans.fit_predict(X)

svc_model = SVC(kernel='rbf', C=n)
svc_model.fit(X_train.join(X['cluster']), y_train)

svc_predictions = svc_model.predict(X_test.join(X['cluster']))



print("=============SVC Confusion Matrix:=============== \n")

svc_conf_matrix = confusion_matrix(y_test, svc_predictions)
print(f"{svc_conf_matrix} \n ")


tp = svc_conf_matrix.diagonal()  # True Positive for each class
fp = svc_conf_matrix.sum(axis=0) - tp  # False Positive for each class
tn = svc_conf_matrix.sum() - (tp + fp)  # True Negative for each class
fn = svc_conf_matrix.sum(axis=1) - tp  # False Negative for each class

svc_acc = accuracy_score(y_test, svc_predictions)
svc_f1 = f1_score(y_test, svc_predictions, average='weighted')

print("============ TP , FP , TN , FN ================\n")
print(f"True Positive for each class: {tp}")
print(f"False Positive for each class: {fp}")
print(f"True Negative for each class: {tn}")
print(f"False Negative for each class: {fn} \n")
print("============ ACC , SEN ,PRE , F-SCORE =========== \n")
print(f"Accuracy: {svc_acc}")
print(f"F1-Score: {svc_f1}\n")

print("============= HER ŞEY İÇİN (RAPOR) ============== \n")
report = classification_report(y_test, svc_predictions)

print(report)

print("=============== Training ve Testing Accuracy   ================ \n")

training_accuracy = svc_model.score(X_train.join(X['cluster']), y_train)
print(f"Training Accuracy: {training_accuracy}" )

acc = accuracy_score(y_test, svc_predictions)
print(f"Testing Accuracy : {acc} \n")

print("#######################################################")
print("------  Accuracy ve F1-Score göre karışlaştırma ------- ")
print("#######################################################\n")

if knn_acc > logistic_regression_acc and knn_f1 > logistic_regression_f1:
    print("knn model logistic_regression modelden daha iyi ")

if knn_acc < logistic_regression_acc and knn_f1 < logistic_regression_f1:
    print("logistic_regression model knn modelden daha iyi ")
   
print("VE \n")    
    
if knn_acc > svc_acc and knn_f1 > svc_f1:
    print("knn model svc modelden daha iyi ")

if knn_acc < svc_acc and knn_f1 < svc_f1:
    print("svc model knn modelden daha iyi ")  

print("VE \n")     
    
if svc_acc > logistic_regression_acc and svc_f1 > logistic_regression_f1:
    print("svc model logistic_regression modelden daha iyi")

if svc_acc < logistic_regression_acc and svc_f1 < logistic_regression_f1:
    print("logistic_regression model svc modelden daha iyi\n")   


print("#######################################################")
print("---------------------  DENEME ------------------------- ")
print("#######################################################\n")    

new_data = pd.DataFrame({
    'SPX': [1352.07],
    'USO': [70.93],
    'SLV': [16.3],
    'EUR/USD': [1.47741],
    'GLD': [111.08],
})

predicted_risk_new_data = knn_model.predict(new_data)

new_data['Predicted_Risk'] = predicted_risk_new_data
print("KNN MODEL DENEMESI \n")
print(new_data[['SPX', 'USO', 'SLV', 'EUR/USD', 'GLD', 'Predicted_Risk']])


print("#######################################################")
print("---------------------  ekler ------------------------- ")
print("#######################################################\n")    



print("=============== en iyi k (knn) hespmlama  ================ \n")

accuracy_values = []

k_values = range(2, 26)

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    knn_predictions = knn_model.predict(X_test)

    accuracy = accuracy_score(y_test, knn_predictions)
    accuracy_values.append(accuracy)

    print(f'For k = {k}, Accuracy = {accuracy}')
    
max_accuracy = max(accuracy_values)
best_k = k_values[accuracy_values.index(max_accuracy)]

print(f'Max Accuracy: {max_accuracy} for k = {best_k}')

print("=============== en iyi k(kmean) hespmlama  ================ \n")
