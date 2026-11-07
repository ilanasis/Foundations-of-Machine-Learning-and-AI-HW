# Simple code for DATA 37711 Homework 2
# Put this file in the same folder as the provided data files.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import norm

np.random.seed(1)
torch.manual_seed(1)
torch.set_num_threads(2)

# A small, standard function to train a network.
def train_net(model, X, y, epochs=8, batch=1024, lr=0.01):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        order = torch.randperm(len(X))
        for inds in order.split(batch):
            pred = model(X[inds])
            loss = ((pred - y[inds])**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

# Problem 1: Gaussian mean and standard deviation network.
# Use the supplied converted CSV file; no extra Parquet package is needed.
def problem1():
    taxi = pd.read_csv('taxi_trips.csv.gz')
    taxi = taxi.dropna()
    print('Taxi columns:', list(taxi.columns))
    # The target is trip_duration, and the other 7 columns are inputs.
    X = taxi.drop(columns=['trip_duration']).to_numpy(dtype='float32')
    y = taxi['trip_duration'].to_numpy(dtype='float32')
    X0, Xtest, y0, ytest = train_test_split(X, y, test_size=.2, random_state=1)
    Xtrain, Xval, ytrain, yval = train_test_split(X0, y0, test_size=.25, random_state=1)

    class GaussianMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(7, 32), nn.ReLU(), nn.Linear(32, 2))
        def forward(self, X):
            out = self.net(X)
            mu = out[:, 0]
            sigma = torch.nn.functional.softplus(out[:, 1]) + 1e-4
            return mu, sigma

    y_mean = ytrain.mean()
    y_std = ytrain.std()
    model = GaussianMLP()
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    A = torch.tensor(Xtrain)
    B = torch.tensor((ytrain - y_mean) / y_std)
    for epoch in range(15):
        order = torch.randperm(len(A))
        for ids in order.split(1024):
            mu, sig = model(A[ids])
            loss = (torch.log(sig) + .5 * ((B[ids]-mu)/sig)**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        mu, sig = model(torch.tensor(Xtest))
        mu, sig = mu.numpy() * y_std + y_mean, sig.numpy() * y_std
    prob = norm.cdf((45-mu)/sig)
    actual = (ytest < 45).astype(float)
    brier = np.mean((prob-actual)**2)
    accuracy = np.mean((prob >= .5) == actual)
    # Compare Gaussian validation NLL against a constant normal baseline.
    with torch.no_grad():
        val_mu, val_sig = model(torch.tensor(Xval))
        val_mu = val_mu * y_std + y_mean
        val_sig = val_sig * y_std
        val_nll = (torch.log(val_sig) + .5*((torch.tensor(yval)-val_mu)/val_sig)**2).mean().item()
    base_mean = np.mean(ytrain)
    base_std = np.std(ytrain)
    base_nll = np.mean(np.log(base_std) + .5*((yval-base_mean)/base_std)**2)
    print('1 baseline and model validation NLL:', base_nll, val_nll)
    base_prob = np.mean(ytrain < 45)
    base_brier = np.mean((base_prob-actual)**2)
    base_accuracy = np.mean((base_prob >= .5) == actual)
    print('1 baseline Brier, accuracy:', base_brier, base_accuracy)
    print('1 Gaussian MLP Brier, accuracy:', brier, accuracy)

# Problem 2: target encoding and embedding models.
def problem2():
    retail = pd.read_csv('retail.csv')
    print('2 unique StockCode, CustomerID:', retail.StockCode.nunique(), retail.CustomerID.nunique())
    train, other = train_test_split(retail, test_size=.4, random_state=1)
    val, test = train_test_split(other, test_size=.5, random_state=1)
    train, val, test = train.copy(), val.copy(), test.copy()
    # log(1 + Quantity) reduces the effect of very large purchases.
    for df in [train, val, test]:
        df['target'] = np.log1p(df.Quantity)

    # Target encoding using only the training set (no validation/test leakage).
    global_mean = train.target.mean()
    for name in ['StockCode', 'CustomerID']:
        means = train.groupby(name).target.mean()
        for df in [train, val, test]:
            df[name + '_mean'] = df[name].map(means).fillna(global_mean)

    cols = ['StockCode_mean', 'CustomerID_mean']
    reg = Ridge(alpha=10).fit(train[cols], train.target)
    pred_ridge = np.expm1(reg.predict(test[cols])).clip(min=0)
    ridge_logmse = mean_squared_error(test.target, reg.predict(test[cols]))
    print('2 target encoding log MSE and quantity MAE:', ridge_logmse,
          mean_absolute_error(test.Quantity, pred_ridge))

    # Integer codes for embedding indices, reserving 0 for unseen categories.
    for name in ['StockCode', 'CustomerID']:
        lookup = {v: i + 1 for i, v in enumerate(train[name].unique())}
        for df in [train, val, test]:
            df[name + '_id'] = df[name].map(lookup).fillna(0).astype(int)
    sizes = [train.StockCode_id.max()+1, train.CustomerID_id.max()+1]

    class RetailNet(nn.Module):
        def __init__(self, combined=False):
            super().__init__()
            self.combined = combined
            self.e1 = nn.Embedding(sizes[0], 8)
            self.e2 = nn.Embedding(sizes[1], 8)
            self.net = nn.Sequential(nn.Linear(18 if combined else 16, 32),
                                     nn.ReLU(), nn.Linear(32, 1))
        def forward(self, x):
            a = self.e1(x[:, 0].long())
            b = self.e2(x[:, 1].long())
            z = torch.cat([a, b], 1)
            if self.combined:
                z = torch.cat([z, x[:, 2:4]], 1)
            return self.net(z)

    def inputs(df):
        return df[['StockCode_id','CustomerID_id']+cols].to_numpy(dtype='float32')
    xtrain, xval, xtest = inputs(train), inputs(val), inputs(test)
    for combined in [False, True]:
        model = RetailNet(combined)
        train_net(model, xtrain, train.target.values, epochs=5, batch=2048, lr=.005)
        with torch.no_grad():
            log_pred = model(torch.tensor(xtest)).numpy().ravel()
        pred = np.expm1(log_pred).clip(min=0)
        print('2', 'combined' if combined else 'embedding', 'log MSE, quantity MAE:',
              mean_squared_error(test.target, log_pred), mean_absolute_error(test.Quantity, pred))

# Problem 3: toy data, RKS and MLP.
def problem3():
    train = pd.read_csv('train.csv'); val = pd.read_csv('val.csv'); test = pd.read_csv('test.csv')
    Xtr = train[['x1','x2']].values; ytr = train.y.values
    Xv = val[['x1','x2']].values; yv = val.y.values
    Xt = test[['x1','x2']].values
    plt.figure()
    plt.scatter(Xtr[:,0], Xtr[:,1], c=ytr, s=12)
    plt.colorbar(label='y');plt.xlabel('x1');plt.ylabel('x2');plt.tight_layout()
    plt.savefig('training_data.png');plt.close()

    class RKS:
        def __init__(self, D):
            self.w = np.random.randn(2, D//2)
            self.b = np.random.uniform(0, 2*np.pi, D//2)
        def features(self, X):
            z = X @ self.w + self.b
            return np.column_stack([np.sin(z), np.cos(z)])
        def fit(self, X, y):
            self.reg = Ridge(alpha=.01).fit(self.features(X), y)
            return self
        def predict(self, X):
            return self.reg.predict(self.features(X))

    Ds = [20, 50, 100, 200]
    rks_tr=[];rks_val=[];rks_models=[]
    for D in Ds:
        model = RKS(D).fit(Xtr,ytr)
        rks_models.append(model)
        rks_tr.append(mean_squared_error(ytr,model.predict(Xtr)))
        rks_val.append(mean_squared_error(yv,model.predict(Xv)))
    ibest = int(np.argmin(rks_val)); rks_best = rks_models[ibest]
    print('3 RKS D/tr MSE/val MSE:',list(zip(Ds,rks_tr,rks_val)))
    print('3 best RKS:',Ds[ibest],rks_val[ibest])

    class MLP(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2,hidden),nn.Tanh(),nn.Linear(hidden,1))
        def forward(self,X):return self.net(X)

    widths = [16,64,128]
    mlp_tr=[];mlp_val=[];mlp_models=[]
    for width in widths:
        model = MLP(width)
        train_net(model,Xtr,ytr,epochs=180,batch=128,lr=.005)
        with torch.no_grad():
            ptr = model(torch.tensor(Xtr,dtype=torch.float32)).numpy().ravel()
            pv = model(torch.tensor(Xv,dtype=torch.float32)).numpy().ravel()
        mlp_tr.append(mean_squared_error(ytr,ptr))
        mlp_val.append(mean_squared_error(yv,pv))
        mlp_models.append(model)
    jbest = int(np.argmin(mlp_val)); mlp_best = mlp_models[jbest]
    print('3 MLP width/tr MSE/val MSE:',list(zip(widths,mlp_tr,mlp_val)))
    print('3 best MLP:',widths[jbest],mlp_val[jbest])

    for name,x,tr,va in [('rks',Ds,rks_tr,rks_val),('mlp',widths,mlp_tr,mlp_val)]:
        for label,losses in [('train',tr),('validation',va)]:
            plt.figure();plt.plot(x,losses,'o-');plt.xlabel('D' if name=='rks' else 'hidden units')
            plt.ylabel('MSE');plt.title(name.upper()+' '+label);plt.tight_layout()
            plt.savefig(name+'_'+label+'.png');plt.close()

    # One prediction per row, no headers.
    np.savetxt('rks_test_predictions.csv',rks_best.predict(Xt),delimiter=',')
    with torch.no_grad():
        prediction = mlp_best(torch.tensor(Xt,dtype=torch.float32)).numpy().ravel()
    np.savetxt('mlp_test_predictions.csv',prediction,delimiter=',')

    xs=np.linspace(-3,3,90)
    a,b=np.meshgrid(xs,xs)
    grid=np.column_stack([a.ravel(),b.ravel()])
    with torch.no_grad():
        mlp_surface=mlp_best(torch.tensor(grid,dtype=torch.float32)).numpy().reshape(a.shape)
    for name,z in [('rks',rks_best.predict(grid).reshape(a.shape)),('mlp',mlp_surface)]:
        plt.figure();plt.contourf(a,b,z,levels=25);plt.colorbar(label='prediction')
        plt.xlabel('x1');plt.ylabel('x2');plt.title(name.upper()+' surface')
        plt.tight_layout();plt.savefig(name+'_surface.png');plt.close()

if __name__ == '__main__':
    problem1()
    problem2()
    problem3()
