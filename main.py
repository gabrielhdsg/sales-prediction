import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#import database
table = pd.read_csv("advertising.csv")

#get table correlation
print(table.corr())

#create a graph with these informations
sns.heatmap(table.corr(), cmap="Greens", annot=True)
plt.show()
  
#creating prediction model
y = table["Vendas"]
x = table[["TV", "Radio", "Jornal"]]
x_train, x_test, y_train, y_test = train_test_split(x, y)

#creating models
model_lr = LinearRegression()
model_rt = RandomForestRegressor()

#training models
model_lr.fit(x_train, y_train)
model_rt.fit(x_train, y_train)

#comparing models
prediction_lr = model_lr.predict(x_test)
prediction_rt = model_rt.predict(x_test)

print(r2_score(y_test,prediction_lr))
print(r2_score(y_test,prediction_rt))
 
#creating model graph
aux_table = pd.DataFrame()
aux_table["y_test"] = y_test
aux_table["Random Tree"] = prediction_lr
aux_table["Linear Regression"] = prediction_rt
print(aux_table)

sns.lineplot(data=aux_table)
plt.show()

#using prediction model
new_table = pd.read_csv["novos.csv"]
prediction = model_rt.predict(new_table)
print(prediction)