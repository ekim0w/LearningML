import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

# Загружаем датасет
df = pd.read_csv("filename.csv", sep=";", decimal= ',')

print(df.head())
print(df.columns)

X = df[["Protein", "Fats", "Carbo"]].values
y = df["Callories"].values

# Масштабирование и train/test
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Модель
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1, activation='linear')
])

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])

# Обучение
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Проверка
loss, mae = model.evaluate(X_test, y_test)
print(f"Средняя ошибка: {mae:.2f} ккал")

