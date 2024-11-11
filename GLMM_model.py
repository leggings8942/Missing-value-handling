import numpy as np
import pandas as pd
from scipy import stats

def poisson_distribution(k, λ):    
    return stats.poisson.pmf(round(k), λ)

def binomial_distribution(k, n, p):
    return stats.binom.pmf(round(k), round(n), p)

def normal_distribution(x, loc=0, scale=1):
    return stats.norm.pdf(x, loc=loc, scale=scale)

def uniform_distribution(x, loc=0, scale=1):
    return stats.uniform.pdf(x, loc=loc, scale=scale)

def moving_average(w1:int, x_list, y_list):
    if len(x_list) != len(y_list):
        raise ValueError()
    
    mov_ave = np.convolve(y_list, [1 / w1] * w1, mode='same')
    return x_list[w1:-w1], mov_ave[w1:-w1]

def diff_convolve(w1:int, x_list, y_list):
    if len(x_list) != len(y_list):
        raise ValueError()
    
    conv_list = np.convolve(y_list, [0.5, 0, -0.5], mode='full')
    tuple_xy  = moving_average(w1, x_list[1:-1], conv_list[2:-2])
    return tuple_xy

def arg_extremum(w1:int, α1:int, x_list, y_list):
    if len(x_list) != len(y_list):
        raise ValueError()
    
    x_list, y_list = diff_convolve(w1, x_list, y_list)
    x_list, y_list = diff_convolve(w1, x_list, y_list)
    
    convol_y  = np.diff(y_list)
    sign_list = np.sign(convol_y[:-1] * convol_y[1:])
    base_std  = np.std(y_list)
    
    sign_idx  = np.where(((sign_list == -1) & (convol_y[1:] > 0)) & (y_list[1:-1] < -α1 * base_std))[0]
    extre_min = 1 + sign_idx
    
    sign_idx  = np.where(((sign_list == -1) & (convol_y[1:] < 0)) & (y_list[1:-1] >  α1 * base_std))[0]
    extre_max = 1 + sign_idx
    
    return extre_min + 2 * (w1 + 1), extre_max + 2 * (w1 + 1)

class Update_Rafael:
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, beta3=0.9999, rate=1e-3):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.rate  = rate
        self.beta1t = self.beta1
        self.beta2t = self.beta2
        self.beta3t = self.beta3
        self.m = np.array([])
        self.v = np.array([])
        self.w = np.array([])
        self.isFirst = True

    def update(self, grads):
        if self.isFirst == True:
            self.m = np.zeros(grads.shape)
            self.v = np.zeros(grads.shape)
            self.w = np.zeros(grads.shape)
            self.isFirst = False

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        m_hat = self.m / (1 - self.beta1t)

        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        v_hat = self.v / (1 - self.beta2t)

        self.w = self.beta3 * self.w + (1 - self.beta3) * ((grads - m_hat) ** 2)
        w_hat = self.w / (1 - self.beta3t)
        
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
        self.beta3t *= self.beta3

        return self.alpha * np.sign(grads) * np.abs(m_hat) / np.sqrt(v_hat + 1e-8) / np.sqrt(w_hat + self.rate)





class SigmoidLinearRegression:
    def __init__(self, isStandardization=True, tol=1e-7, fix_intercept=None, max_iterate=100000, learning_rate=0.001, random_state=None):
        self.alpha   = np.array([], dtype=np.float64)
        self.alpha0  = np.float64(0.0)
        self.isStandardization = isStandardization
        self.standardization = np.empty([2, 1])
        self.tol = tol
        self.fix_intercept = fix_intercept
        self.max_iterate = max_iterate
        self.correct_alpha   = Update_Rafael(alpha=learning_rate)
        self.correct_alpha0  = Update_Rafael(alpha=learning_rate)

        self.random_state = random_state
        if random_state != None:
            self.random = np.random
            self.random.seed(seed=self.random_state)
        else:
            self.random = np.random
        
        if self.fix_intercept != None:
            self.isStandardization = False

    def fit(self, x_train, y_train, visible_flg=False):
        if type(x_train) is pd.core.frame.DataFrame:
            x_train = x_train.to_numpy()

        if type(y_train) is pd.core.series.Series:
            y_train = y_train.to_numpy()
        
        if type(x_train) is list:
            x_train = np.array(x_train)
        
        if type(y_train) is list:
            y_train = np.array(y_train)
        
        if (x_train.ndim != 2) or (y_train.ndim != 1):
            print(f"x_train dims = {x_train.ndim}")
            print(f"y_train dims = {y_train.ndim}")
            print("エラー：：次元数が一致しません。")
            return False
        
        num, s = x_train.shape
        x_train = np.log(x_train + 1)

        if self.isStandardization:
            self.standardization = np.empty([2, s])
            self.standardization[0] = np.mean(x_train, axis=0)
            self.standardization[1] = np.std( x_train, axis=0)

            if np.all(self.standardization[1] == 0):
                self.alpha  = np.log(y_train[0] + 1e-16) / x_train[0, :] / s
                self.alpha0 = 0
                self.standardization[0] = 0
                self.standardization[1] = 1
                return True

            x_train = (x_train - self.standardization[0]) / self.standardization[1]
        elif np.all(np.std( x_train, axis=0) == 0):
            self.alpha  = np.log(y_train[0] + 1e-16) / x_train[0, :] / s
            self.alpha0 = 0
            return True
        y_train = y_train.reshape([num, 1])

        if self.fix_intercept == None:
            #正規方程式
            A = np.hstack([x_train, np.ones([num, 1])])
            b = y_train.reshape([num])

            x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
            self.alpha, self.alpha0 = x[0:s], x[s]

        else:
            self.alpha, self.alpha0 = self.random.random([1, s]), self.fix_intercept
            self.alpha = self.alpha.reshape([1, s])

            update = 99
            now_ite = 0
            while (update > self.tol) and (now_ite < self.max_iterate):
                diff_alpha   = np.zeros(self.alpha.shape)
                diff_alpha0  = np.float64(0)

                if self.fix_intercept != None:
                    self.alpha0 = self.fix_intercept

                lambda_vec = np.sum(self.alpha * x_train, axis=1) + self.alpha0
                lambda_vec = lambda_vec.reshape([num, 1])

                diff_alpha_calc = y_train - lambda_vec
                
                diff_alpha   = np.sum(diff_alpha_calc * x_train, axis=0).reshape([1, s]) / num
                diff_alpha0  = np.sum(diff_alpha_calc)                                   / num

                tmp_alpha    = self.correct_alpha.update(  diff_alpha)
                self.alpha   += tmp_alpha
                tmp_alpha0   = self.correct_alpha0.update( diff_alpha0)
                self.alpha0  += tmp_alpha0

                update_diff = np.sqrt(np.sum(diff_alpha ** 2) + diff_alpha0 ** 2)
                update  = np.sqrt(np.sum(tmp_alpha ** 2) + tmp_alpha0 ** 2)
                now_ite = now_ite + 1

                if self.fix_intercept != None:
                    update  = np.sqrt(np.sum(tmp_alpha ** 2))

                if (now_ite % 10 == 0) and visible_flg:
                    lambda_ = np.sum(self.alpha * x_train, axis=1) + self.alpha0
                    lambda_ = lambda_.reshape([num, 1])
                    mse     = np.sum((y_train - lambda_) ** 2) / num

                    print(f"ite:{now_ite}  alpha0:{self.alpha0}  alpha:{self.alpha}  update_diff:{update_diff}  update:{update}  MSE:{mse}", flush=True)

        return True
    
    def log_likelihood(self, x_train, y_train):
        if type(x_train) is pd.core.frame.DataFrame:
            x_train = x_train.to_numpy()

        if type(y_train) is pd.core.series.Series:
            y_train = y_train.to_numpy()
        
        if type(x_train) is list:
            x_train = np.array(x_train)
        
        if type(y_train) is list:
            y_train = np.array(y_train)
        
        if (x_train.ndim != 2) or (y_train.ndim != 1):
            print(f"x_train dims = {x_train.ndim}")
            print(f"y_train dims = {y_train.ndim}")
            print("エラー：：次元数が一致しません。")
            return False
        
        num, _  = x_train.shape
        y_train = y_train.reshape([num, 1])
        
        predict = self.predict(x_train)
        predict = predict.reshape([num, 1])
        
        prob           = np.frompyfunc(lambda x, y, z: normal_distribution(x, loc=y, scale=z), 3, 1)(y_train, predict, np.std(y_train - predict))
        prob           = prob.astype(float).reshape([num, 1])
        log_likelihood = np.sum(np.log(prob + 1e-16)) / num
        return log_likelihood
    
    def calc_coef_of_determin(self, x_train, y_train):
        if type(x_train) is pd.core.frame.DataFrame:
            x_train = x_train.to_numpy()

        if type(y_train) is pd.core.series.Series:
            y_train = y_train.to_numpy()
        
        if type(x_train) is list:
            x_train = np.array(x_train)
        
        if type(y_train) is list:
            y_train = np.array(y_train)
        
        if (x_train.ndim != 2) or (y_train.ndim != 1):
            print(f"x_train dims = {x_train.ndim}")
            print(f"y_train dims = {y_train.ndim}")
            print("エラー：：次元数が一致しません。")
            return False
        
        num, _  = x_train.shape
        y_train = y_train.reshape([num, 1])
        y_mean  = np.mean(y_train)

        predict = self.predict(x_train)
        predict = predict.reshape([num, 1])
        coef_of_determin = 1 - np.sum((y_train - predict) ** 2) / np.sum((y_train - y_mean) ** 2)
        return coef_of_determin
    
    def model_reliability(self, x_train, y_train):
        if type(x_train) is pd.core.frame.DataFrame:
            x_train = x_train.to_numpy()

        if type(y_train) is pd.core.series.Series:
            y_train = y_train.to_numpy()
        
        if type(x_train) is list:
            x_train = np.array(x_train)
        
        if type(y_train) is list:
            y_train = np.array(y_train)
        
        if (x_train.ndim != 2) or (y_train.ndim != 1):
            print(f"x_train dims = {x_train.ndim}")
            print(f"y_train dims = {y_train.ndim}")
            print("エラー：：次元数が一致しません。")
            return False
        
        num, _  = x_train.shape
        
        log_likelihood = self.log_likelihood(x_train, y_train)
        if log_likelihood == False:
            return False
        
        reliability = 2/3 * np.sqrt(num)

        return log_likelihood + reliability

    def predict(self, x_test, sample=100, step=1):
        if type(x_test) is pd.core.frame.DataFrame:
            x_test = x_test.to_numpy()
        
        if type(x_test) is list:
            x_test = np.array(x_test)
        
        if x_test.ndim != 2:
            print(f"x_train dims = {x_test.ndim}")
            print("エラー：：次元数が一致しません。")
            return False
        
        x_test = np.log(x_test + 1)

        if self.fix_intercept != None:
            self.alpha0 = self.fix_intercept
        
        if self.isStandardization:
            x_test = (x_test - self.standardization[0]) / self.standardization[1]

        lambda_vec = np.sum(self.alpha * x_test, axis=1) + self.alpha0
        
        return lambda_vec
