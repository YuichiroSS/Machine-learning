# データの読み込み
import pandas as pd

df = pd.read_csv('karaage_data.csv')
print(df.head()) # 最初の5行を表示


# x：来場者[万人]
x = df[['x']] 
# y：出店数
y = df[['y']]

# グラフによる可視化
import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(x, y, 'o')
plt.show()

# scikit-learnも使った単回帰分析
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(x, y)

# 可視化
plt.plot(x, y, 'o')
plt.plot(x, model_lr.predict(x), linestyle="solid")
plt.show()

print('モデル関数の回帰変数 w1：%.3f' %model_lr.coef_)
print('モデル関数の切片 w2：%.3f' %model_lr.intercept_)
print('y = %.3fx + %.3f' % (model_lr.coef_, model_lr.intercept_))
print('決定係数 R^2：', model_lr.score(x, y))