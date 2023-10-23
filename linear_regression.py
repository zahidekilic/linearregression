######################################################
# Sales Prediction with Linear Regression
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################
df = pd.read_csv("datasets/advertising.csv")
df.head()
#sales: bağımlı değişken

df.shape

X = df[["TV"]] #bağımsız değişken
y = df["sales"] #bağımlı değişken

##########################
# Model
##########################
reg_model = LinearRegression().fit(X, y)

# katsayıları nasıl bulacağız?
# y_hat = b + w*x
# sabit (b - bias)
reg_model.intercept_
#TV'nin katsayısı nedir? (w1)
reg_model.coef_[0]


##########################
# Tahmin
##########################
# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?
reg_model.intercept_ + reg_model.coef_[0] * 150

# 500 birimlik TV harcaması olsa ne kadar satış olması beklenir?
reg_model.intercept_ + reg_model.coef_[0] * 500


df.describe().T
#buradan baktığımızda TV'nin maksimum değeri 296, ama artık ağırlıklar belirlendiği için istediğim bir değeri vererek modelden tahmin yapmasını isteyebilirim.
#yukarıda 500 sordum, listede olmadığı halde gösterdi, çünkü ağırlıklar elimde.


# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r") #ci: güven aralığı, color: regresyon çizgisinin rengi
# regplot: regresyon modelini çizmek için kullanılan bir metod.

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_, 2)} + TV*{round(reg_model.coef_[0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show(block=True)

##########################
# Doğrusal Regresyonda Tahmin Başarısı
##########################
y_pred = reg_model.predict(X)


#MSE
mean_squared_error(y, y_pred) #ilk parametre gerçek değerler, ikinci parametre tahmin edilen değerler
#değer 10.5 çıktı, ama minimum mu bilemiyorum, o yüzden y'nin ortalamasına ve standart sapmasına bakacağım, ona göre elimdeki 10.5 değeri minimum mu diye bakacağım:
y.mean() #14.02
y.std() #5.21
#buradan baktığım zaman değerlerin 9 ila 19 arasında değişiyor olmasını bekliyorum.. 10.5 biraz büyük geldi.

#RMSE
np.sqrt(mean_squared_error(y, y_pred)) #MSE'nin kareköküdür 3.24

#MAE
mean_absolute_error(y, y_pred) #2.54

#burada metrikler birbiriyle kıyaslanmaz, bir tane metrik seçilir ve değişiklikler yaparak aynı metriğin önceki-sonraki değerlerine bakılır.

#R-SQUARE / R-KARE
# Doğrusal regresyon modellerinde modelin başarısını gösteren çok önemli bir metriktir
# Verisetindeki bağımsız değişkenlerin, bağımlı değişkeni açıklama yüzdesidir.
# Mesela burası için TV değişkeninin, satışların yüzde kaçını açıkladığını göstermektedir.
# burada önemli bir nokta vardır: Değişken sayısı arttıkça r-kare şişmeye meyillidir.Burada düzeltilmiş r-kare  değerinin de göz önünde bulundurulması gerekir.
# makine öğrenmesinde istatistik, ekonometrik modellerle bir alakamız yoktur.
reg_model.score(X, y) #0.61 --> TV satışları, tüm satışların %61'ini açıklayabilmektedir.


######################################################
# Multiple Linear Regression - Çoklu Doğrusal Regresyon
######################################################
df = pd.read_csv("datasets/advertising.csv")

X = df.drop("sales", axis=1) #bağımsız değişkenler
y = df[["sales"]] #bağımlı değişken

##########################
# Model
##########################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1) #hoca ile aynı sonuca ulaşmak için random_state kullanıyoruz
# yukarıdaki kod, bize verimizi 4 parçaya bölmektedir.

X_train.shape #160
y_train.shape #160
X_test.shape #40
y_test.shape #40

#train setiyle model kuruyoruz, test setiyle test ediyoruz
#reg_model = LinearRegression()
#reg_model.fit(X_train, y_train)

#ya da şu şekilde yazabiliriz
reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_ # array([2.90794702])

# coefficients (w - weights)
reg_model.coef_ # array([[0.0468431 , 0.17854434, 0.00258619]])


##########################
# Tahmin
##########################
df.head()


# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# 2.90
# 0.0468431 , 0.17854434, 0.00258619

#MÜLAKAT SORUSU:
#1- MODEL DENKLEMİNİ YAZINIZ:
# sales = b + w1*TV + w2*radio + w3*newspaper
# sales = 2.90 + 0.0468431*30 + 0.17854434*10 + 0.00258619*40

#2- BEKLENEN SATIŞIN NE OLDUĞUNU TAHMİN EDİNİZ:
2.90 + 0.0468431*30 + 0.17854434*10 + 0.00258619*40

#bunu fonksiyonel olarak nasıl yaparız:
yeni_veri = [[30],[10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri) #6.202131 -->benimkiyle arasındaki fark küsüratlardan kaynaklanmakta.


##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE
y_pred = reg_model.predict(X_train)
mean_absolute_error(y_train, y_pred) #1.3288

#TRAIN RKARE : bağımsız değişkenlerin bağımlı değişkeni etkileme oranı
reg_model.score(X_train, y_train)  #0.8959 --> öncekine göre çok yükseldi, buradan yeni değişken eklediğimizde başarılı bir model olduğunu düşünebiliriz.

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_absolute_error(y_test, y_pred)) #1.01
#normalde test hatası train hatasından daha yüksek çıkar, burada daha düşük çıktı, bu güzel bir durumdur

# Test RKARE
reg_model.score(X_test, y_test) #0.8927

# 10 Katlı Cross Validation Yöntemi
# RMSE ile yapıyoruz
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring = "neg_mean_squared_error"))) #1.69
#burada verisetimiz küçük olduğu için çapraz doğrulamanın sonucu diğer başarı değerlendirme metriklerine göre daha güvenilirdir.


######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################
#Sıfırdan gradient descentin nasıl çalıştığını anlamaya çalışmak için kod yazıyoruz

# Cost function MSE: 2 değişkenli bir problemde kullanılır
def cost_function(Y, b, w, X): #Y:bağımlı değişken, b:sabit, w:ağırlık, X:bağımsız değişken
    m = len(Y) #formülizasyondaki toplam formülündeki N değeri.. (gözlem sayısı)
    sse = 0 #sum of squared error

    for i in range(0, m):
        y_hat = b + w * X[i] #tahmin edilen y değeri..
        y = Y[i]
        sse += (y_hat - y) ** 2 #her iterasyonda elde edilen değeri topladığımız için fordan çıktığında toplam hatayı bulmuş olacağız.

    mse = sse / m
    return mse


# update_weights
def update_weights(Y, b, w, X, learning_rate): #Y:bağımlı değişken, b:sabit, w:ağırlık, X:bağımsız değişken, learning_rate:öğrenme oranı
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m): #tüm gözlem birimlerini gez
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y) #sabitin kısmi türevi
        w_deriv_sum += (y_hat - y) * X[i] #ağırlığın kısmi türevi
    new_b = b - (learning_rate * 1 / m * b_deriv_sum) #sabitin nihai ortalama değeri
    new_w = w - (learning_rate * 1 / m * w_deriv_sum) #ağırlığın nihai ortalama değeri
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters): #num_iters:iterasyon sayısı
    # başlangıçta elimizde ne var diye yazdırıyoruz
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)

        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse)) #her 100 iterasyonda bir raporla


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X))) #son raporlama
    return cost_history, b, w


df = pd.read_csv("datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

#parametreleri kafamızdan güncelliyoruz
# hyperparameters: verisetinden bulunamayan, kullanıcı tarafından ayarlanması gereken parametrelerdir
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)
df["radio"].mean()