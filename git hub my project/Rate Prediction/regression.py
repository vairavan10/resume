import pandas as pd
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('house1.csv')


X = dataset[['No of house', 'Size of house']]
y = dataset['Price']


model = LinearRegression()


model.fit(X, y)


new_data = pd.DataFrame({'No of house': [5, 10], 'Size of house': [2000, 3000]})
predictions = model.predict(new_data)


for i, prediction in enumerate(predictions):
    print(f"Prediction for house {i+1}: {prediction}")

import matplotlib.pyplot as plt
dataset.plot(x='Price',y='No of bedroom',style='o')
plt.xlabel('price')
plt.ylabel('size')
plt.show()