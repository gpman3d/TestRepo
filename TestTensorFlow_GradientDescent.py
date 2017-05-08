import tensorflow as tf             #텐서플로우 tf로 쓰겠음.
import numpy as np                  #벡터 산술연산 / 브로드캐스팅을 제공.
import matplotlib.pyplot as plt     #그래프를 그리는데 사용.

''' ------------------------ Make Random Data -------------------------------------'''
num_points = 500                    #반복할 수.
vectors_set = []                    #랜덤 백터값을 생성.
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)        # 0.0~0.55 사이의 랜덤 값을 numpy를 이용해 빠르게 산출.
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)  # 0.3이상 + X 값에 따라 미세한 영향이 있는 랜덤 값을 산출.
#    y1 = np.random.normal(0.0, 0.55)        # 테스트로 넣어봤다.
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]        # 파이썬 특기. 리스트 내포 문법. v[0]는 X값, V[1]에는 Y에는 Y값을 담고 있다.
y_data = [v[1] for v in vectors_set]

''' ------------------------ Computational Graph Designing -------------------------------------'''
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 텐서플로우 변수의 값을 초기화. 정규분포 난수를 생성. .Variable 메소드를 호출하면 텐서플로우 내부의 그래프 데이터 구조에 만들어질 하나의 변수를 정의하는 것으로 봐야 한다.
b = tf.Variable(tf.zeros([1]))                      # 텐서플로우 변수 값을 초기화.
y = W * x_data + b                                  # 백터 값에 X값을 곱해 변화시키고 b값을 적용( b값은 그래프를 위로 올리는 역할을 할 듯 )
loss = tf.reduce_mean(tf.square(y - y_data))        # 손실 함수 정의 ( y값 에서 y_data 값을 빼고 제곱. 경사하강법을 사용하기 위함으로 추정. )

# Optimization Operation Definition
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 기울기 최소값을 찾기 위해 텐서플로우에 그레디언트 디센트를 적용.
train = optimizer.minimize(loss)                    # 정해진 그래디언트 디센트 값을 최적화하여 train에 세팅.
init = tf.initialize_all_variables()                # 변수를 초기화해야 텐서플로우는 동작한다.
''' ------------------------ TensorFlow Graph Designing is Done --------------------------------'''
# We instantiate a vehicle
sess = tf.Session()                         # 시즌을 생성.
sess.run(init)                              # 시즌을 초기화.

for step in range(8):                       # x번의 훈련을 처리.
    sess.run(train)                         # 시즌 시작.
    print( step, sess.run(W), sess.run(b), sess.run(loss))  # 훈련과정의 값을 찍어보자.

# Show the linear regression result
# 그래프를 그려보자.
plt.figure(1)                       # 그림 인스턴스.
plt.title('Linear Regression');     # 그래프의 이름.
plt.xlabel('x');                    # X축 이름 지정.
plt.ylabel('y')                     # Y축 이름 지정.
plt.plot(x_data, y_data, 'ro')      # x_data,와 y_data의 그래프를 그린다. "ro"는 점형태로 그래프를 표현하는 것.
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))    # x_data와 탠서플로우 결과값?을 이용해 그래프를 생성.
#plt.legend(loc= 'upperleft')       # 범례를 지정하기 위한 것으로 보이는데 등장하고 있지 않다.....
plt.show()                          # 그래프를 보여주시오.
