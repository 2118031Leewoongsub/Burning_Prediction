import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 데이터 불러오기
filename = './archive/forestfires.csv'
data = pd.read_csv(filename)

# 데이터 전처리
X = data[['temp', 'RH', 'rain', 'wind']]  # 사용할 특성들
y = data['FFMC']  # 예측할 타겟 변수

# 모델 선택 및 학습
model = LinearRegression()
model.fit(X, y)

# 예측
predictions = model.predict(X)

# 성능 평가
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 실제 값과 예측 값의 비교 그래프
plt.scatter(y, predictions, c=['blue' if actual > predicted else 'red' for actual, predicted in zip(y, predictions)])
plt.xlabel('Actual FFMC')
plt.ylabel('Predicted FFMC')
plt.title('Actual vs Predicted FFMC')
plt.show()
