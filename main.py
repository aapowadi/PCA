# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def Question1(n,d,theta):
    # Generating data
    X = []
    for i in range(n):
        # mu_x
        mu_x = np.arange(5, 15, 0.1)
        mu_x = np.mat(mu_x).T
        # tmp
        tmp = np.random.randn(d, d)
        # evals
        t1 = 100 * np.random.randn(int(np.rint(d / 8)), 1)
        t2 = np.random.randn(d - int(np.rint(d / 8)), 1)
        t3 = np.squeeze(np.concatenate((t1 * t1, t2 * t2), axis=0))
        evals = np.diag(t3)
        # Covariance
        sigma_x = np.dot(np.dot(tmp, np.sqrt(evals)), tmp.T)
        np.shape(sigma_x)
        x = mu_x + np.dot(sigma_x, np.random.randn(d, 1))
        X.append(x)
    X = np.squeeze(X)
    # noise
    e = np.random.normal(0, 1e-6, n)
    e = np.mat(e)
    # np.shape(e)
    y = (np.dot(X, theta) + e).T

    # Train test Split
    n_train = 50
    n_test = n - n_train
    X_train, y_train = X[:n_train, :], y[:n_train, :]
    X_test, y_test = X[n_train:, :], y[n_train:, :]
    X_train.shape, y_train.shape, X_test.shape, y_test.shape
    return X,X_test,y,y_test

def Question2(n, d, X, X_test, y, y_test):
    t,test_mses = PCA(n, d, X, X_test, y, y_test)
    plt.scatter(t, test_mses)
    plt.xlabel('d')
    plt.ylabel('norm_mse')
    plt.show()
    plt.savefig('Question2')

def Question3():
    df1 = pd.read_csv("blogData_test-2012.02.01.00_00.csv")
    df2 = pd.read_csv("blogData_test-2012.02.02.00_00.csv")
    df3 = pd.read_csv("blogData_test-2012.02.03.00_00.csv")
    d1 = np.array(df1)
    d2 = np.array(df2)
    d3 = np.array(df3)
    tmp = np.concatenate((d1,d2),axis=0)
    X = tmp[:,:-1]
    y = np.mat(tmp[:,-1]).T
    X_test = d3[:, :-1]
    y_test = np.mat(d3[:, -1]).T
    # df.append(d1[1:,1:])
    # df.append(d2[1:,1:])
    n = len(X[:,1])
    d = len(X[1,:])
    t,test_mses = PCA(n,d,X,X_test,y,y_test)
    plt.scatter(t, test_mses)
    plt.xlabel('d')
    plt.ylabel('norm_mse')
    plt.show()
    plt.savefig('Question3')


def PCA(n,d,X,X_test,y,y_test):
    test_mses=[]
    for r in range(1,d):
        OnesMu = np.ones((1,d))
        mu_cap = cal_Mu_cap(n,X)
        Z = X - np.dot(OnesMu, mu_cap.T)
        Ufull, Sfull, Vtfull = np.linalg.svd(Z)
        V = Vtfull.T[:, : r]
        B = np.dot(X, V)
        OnesB = np.ones((n,1))
        B_tilda = np.concatenate((OnesB,B), axis=1)
        B_pseudo = np.dot(np.linalg.inv(np.dot(B_tilda.T,B_tilda)),B_tilda.T)
        thetav_tilda_cap = np.dot(B_pseudo,y)  # Pseudo Inverse of B_tilda * y
        c_cap = thetav_tilda_cap[0][0]
        thetav_cap = thetav_tilda_cap[1:r + 1, 0]
        y_cap_test = np.dot(X_test,np.dot(V,thetav_cap)) + c_cap
        norm_test_mse = np.linalg.norm(y_test - y_cap_test) / np.linalg.norm(y_test)
        test_mses.append(norm_test_mse)
    t = np.arange(1,d,1)

    return t,test_mses



def cal_Mu_cap(n,x):
    temp = 0
    for i in range(1, n):
        temp = temp + x[i]
    return temp/n



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n = 80; d =100
    theta = np.arange(100,50,-0.5).T
    X, X_test, y, y_test = Question1(n,d,theta)
    Question2(n, d, X, X_test, y, y_test)
    Question3()


