import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
Xbar = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
w_init = np.random.randn(Xbar.shape[1])
lamda = 0.0001
def sigmoid(x):
    return 1/(1+np.exp(-x))
def loss(w):
    a=sigmoid(Xbar.dot(w))
    return -np.mean(y*np.log(a)+(1-y)*np.log(1-a)+0.5*lamda*np.sum(w**2))
def numerical_grad(w):
    eps= 1e-4
    g=np.zeros_like(w)
    for i in range(w.shape[0]):
        w1= w.copy() #nếu như chỉ gán 1 lần thì những vòng lặp tiếp theo sẽ khiến kq 
                    #của các ptu ở vòng lặp trước bị sai
        w2=w.copy()
        w1[i]+=eps
        w2[i]-=eps
        g[i]=(loss(w1) -loss(w2))/(2*eps)
    return g 
def logistic_reg(w, X,y, lamda, eta=0.1, num_epoches=2000):
    ep=0
    N=X.shape[0]
    d=X.shape[1]
    while ep<num_epoches:
        ep+=1
        mix_idx=np.random.permutation(N)
        for i in mix_idx:
            xi=X[i]
            yi=y[i]
            ai=sigmoid(xi.dot(w))
            w=w-eta*((ai-yi)*xi+lamda*w)
        if (np.linalg.norm(numerical_grad(w))/d<1e-3): break
    return w
def logistic_reg2(w, X,y, lamda, eta=0.1, num_epoches=2000):
    ep=0
    N=X.shape[0]
    d=X.shape[1]
    w_old=w
    while ep<num_epoches:
        ep+=1
        mix_idx=np.random.permutation(N)
        for i in mix_idx:
            xi=X[i]
            yi=y[i]
            ai=sigmoid(xi.dot(w))
            w=w-eta*((ai-yi)*xi+lamda*w)
        if (np.linalg.norm(w-w_old)/d<1e-6): break
        w_old=w
    return w
w=logistic_reg(w_init, Xbar, y, lamda,eta=0.05, num_epoches=500)
print('Solution of Logistic Regression: ', w)
print('Solution of Logistic Regression (Book): ', logistic_reg2(w_init, Xbar, y, lamda,eta=0.05, num_epoches=500))
print('Tỉ lệ đậu ở dữ liệu ban đầu là: ', sigmoid(X*w[0] + w[1]).T)
x0=X[np.where(y==0)]
y0=y[np.where(y==0)]
x1=X[np.where(y==1)]
y1=y[np.where(y==1)]
plt.plot(x0,y0, 'ro', label='Rớt')
plt.plot(x1,y1,'bs', label='Đậu')
plt.axis((-2,8,-1,2))
plt.xlabel('Số giờ học')
plt.ylabel('Phần trăm đậu')
xx=np.linspace(0,6,1000)
w0=w[1]
w1=w[0]
plt.grid(True)
plt.title('Đồ thị biểu diễn tỉ lệ đậu rớt')
plt.legend(loc='upper left')
yy=sigmoid(w0+w1*xx)
plt.plot(xx,yy,'g-')
plt.plot(-w0/w1, 0.5, 'y^', markersize=8) #Ngưỡng cứng
plt.show()

        