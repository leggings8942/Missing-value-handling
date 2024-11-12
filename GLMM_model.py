import numpy as np
import pandas as pd
from scipy import stats, special

def poisson_distribution(k, λ):    
    return stats.poisson.pmf(round(k), λ)

def binomial_distribution(k, n, p):
    return stats.binom.pmf(round(k), round(n), p)

def normal_distribution(x, loc=0, scale=1):
    return stats.norm.pdf(x, loc=loc, scale=scale)

def uniform_distribution(x, loc=0, scale=1):
    return stats.uniform.pdf(x, loc=loc, scale=scale)

def moving_average(w1:int, x_list):
    mov_ave = np.convolve(x_list, [1 / w1] * w1, mode='same')
    return mov_ave[w1:-w1]

def diff_convolve(w1:int, x_list):
    conv_list = np.convolve(x_list, [0.5, 0, -0.5], mode='full')
    tuple_xy  = moving_average(w1, conv_list[2:-2])
    return tuple_xy

def arg_extremum(w1:int, α1:int, x_list):
    x_list = diff_convolve(w1, x_list)
    x_list = diff_convolve(w1, x_list)
    
    convol_y  = np.diff(x_list)
    sign_list = np.sign(convol_y[:-1] * convol_y[1:])
    base_std  = np.std(x_list)
    
    sign_idx  = np.where(((sign_list == -1) & (convol_y[1:] > 0)) & (x_list[1:-1] < -α1 * base_std))[0]
    extre_min = 1 + sign_idx
    
    sign_idx  = np.where(((sign_list == -1) & (convol_y[1:] < 0)) & (x_list[1:-1] >  α1 * base_std))[0]
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





def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normal_cumulative(x, loc=0, scale=1):
    return stats.norm.cdf(x, loc=loc, scale=scale)

def normal_4th_distribution(x):
    return 2 * np.exp(-(x ** 4))

def normal_4th_cumulative(x):
    return 0.5 + 1/2 * np.sign(x) * special.gammainc(1/4, x ** 4)

class LogitLinearRegression:
    def __init__(self,
                 isStandardization:bool,          # 入力値の正規化処理の有無
                 window_size:int = 11,            # 移動平均処理時のウィンドウサイズ
                 norm_z:float = 2.17,             # 標準正規分布のZ値
                 tol:float = 1e-7,                # 許容誤差
                 fix_intercept:float|None = None, # 切片の設定値(固定)
                 max_iterate:int = 100000,        # 最大ループ回数
                 learning_rate:float = 0.001,     # 学習係数
                 random_state:int|None = None):   # 乱数のシード値
        self.alpha0  = np.float64(0.0)
        self.alpha1  = np.array([], dtype=np.float64)
        self.alpha2  = np.float64(0.0)
        self.beta1   = np.float64(0.0)
        self.beta2   = np.float64(0.0)
        self.isStandardization = isStandardization
        self.x_standardization = np.empty([2, 1])
        self.y_standardization = np.empty([2, 1])
        self.window_size       = window_size
        self.norm_z            = norm_z
        self.tol               = tol
        self.fix_intercept     = fix_intercept
        self.max_iterate       = max_iterate
        self.correct_alpha0    = Update_Rafael(alpha=learning_rate)
        self.correct_alpha1    = Update_Rafael(alpha=learning_rate)
        self.correct_alpha2    = Update_Rafael(alpha=learning_rate)
        self.correct_beta1     = Update_Rafael(alpha=learning_rate)
        self.correct_beta2     = Update_Rafael(alpha=learning_rate)

        self.random_state = random_state
        if random_state != None:
            self.random = np.random
            self.random.seed(seed=self.random_state)
        else:
            self.random = np.random

    def fit(self, x_train, y_train, visible_flg:bool=False):
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
        
        # 正規化指定の有無
        if self.isStandardization:
            # x軸の正規化
            _, s = x_train.shape
            self.x_standardization    = np.empty([2, s])
            self.x_standardization[0] = np.mean(x_train, axis=0)
            self.x_standardization[1] = np.std( x_train, axis=0)

            # 標準偏差が0の場合
            zero_judge = (self.x_standardization[1] == 0)
            self.x_standardization[0][zero_judge] = 0
            self.x_standardization[1][zero_judge] = 1

            x_train = (x_train - self.x_standardization[0]) / self.x_standardization[1]
            
            # y軸の正規化
            self.y_standardization    = np.empty(2)
            self.y_standardization[0] = np.mean(y_train)
            self.y_standardization[1] = np.std( y_train)

            # 標準偏差が0の場合
            if self.y_standardization[1] == 0:
                self.y_standardization[0] = 0
                self.y_standardization[1] = 1

            y_train = (y_train - self.y_standardization[0]) / self.y_standardization[1]
        
        # 標準正規分布の確立分布　数表
        # URL:https://kyozaikenkyu-statistics.blog.jp/%E6%A8%99%E6%BA%96%E6%AD%A3%E8%A6%8F%E5%88%86%E5%B8%83%E6%95%B0%E8%A1%A8.pdf
        # 主に以下の値が利用されると想定する
        # 優位水準8%   (片側4.0%) ・・・1.75
        # 優位水準5%   (片側2.5%) ・・・1.96
        # 優位水準3%   (片側1.5%) ・・・2.17
        # 優位水準1%   (片側0.5%) ・・・2.58
        # 優位水準0.5% (片側0.25%)・・・2.81
        
        extre_min, extre_max = arg_extremum(self.window_size, self.norm_z, y_train)
        if (len(extre_min) == 0) or (len(extre_max) == 0): # 候補点が発見できない場合
            extre_min = round(len(y_train) / 2)
            extre_max = round(len(y_train) / 2)
        else:
            extre_min, extre_max = np.min(extre_min), np.max(extre_max)
        
        if extre_min > extre_max:                          # 始値 > 終値 となっている場合
            extre_min = round(len(y_train) / 2)
            extre_max = round(len(y_train) / 2)
        
        x_train_min, y_train_min = x_train[:extre_min, :], y_train[:extre_min]
        x_train_max, y_train_max = x_train[extre_max:, :], y_train[extre_max:]
        
        num_min, s = x_train_min.shape
        num_max, _ = x_train_max.shape

        y_train_min = y_train_min.reshape([num_min, 1])
        y_train_max = y_train_max.reshape([num_max, 1])

        # 学習係数の初期化
        self.alpha0 = self.random.random()
        self.alpha1 = self.random.random([1, s])
        self.alpha2 = self.random.random()
        self.beta2  = self.random.random()
        
        if self.fix_intercept == None:       # 切片の指定がない場合
            self.beta1 = self.random.random()
        else:                                # 切片の指定がある場合
            self.beta1 = self.fix_intercept

        update = 99
        now_ite = 0
        while (update_diff > self.tol) and (update > self.tol) and (now_ite < self.max_iterate):
            output_min  = normal_4th_cumulative(np.sum(self.alpha1 * x_train_min, axis=1) + self.alpha2).reshape([num_min, 1])
            ΔLoss_min   = y_train_min - self.alpha0 * output_min - self.beta1
            Δoutput_min = normal_4th_distribution(np.sum(self.alpha1 * x_train_min, axis=1) + self.alpha2).reshape([num_min, 1])
            
            diff_alpha0 = np.sum(ΔLoss_min * output_min)
            diff_alpha1 = np.sum(ΔLoss_min * self.alpha0 * Δoutput_min * x_train_min, axis=0).reshape([1, s])
            diff_alpha2 = np.sum(ΔLoss_min * self.alpha0 * Δoutput_min)
            diff_beta1  = np.sum(ΔLoss_min)
            
            output_max  = normal_4th_cumulative(np.sum(self.alpha1 * x_train_max, axis=1) + self.alpha2).reshape([num_max, 1])
            ΔLoss_max   = y_train_max - self.alpha0 * output_max - self.beta1 + np.exp(self.beta2)
            Δoutput_max = normal_4th_distribution(np.sum(self.alpha1 * x_train_max, axis=1) + self.alpha2).reshape([num_max, 1])
            
            diff_alpha0 += np.sum(ΔLoss_max * output_max)
            diff_alpha1 += np.sum(ΔLoss_max * self.alpha0 * Δoutput_max * x_train_max, axis=0).reshape([1, s])
            diff_alpha2 += np.sum(ΔLoss_max * self.alpha0 * Δoutput_max)
            diff_beta1  += np.sum(ΔLoss_max)
            diff_beta2   = np.sum(ΔLoss_max * np.exp(self.beta2))
            
            diff_alpha0 = diff_alpha0 / (num_min + num_max)
            diff_alpha1 = diff_alpha1 / (num_min + num_max)
            diff_alpha2 = diff_alpha2 / (num_min + num_max)
            diff_beta1  = diff_beta1  / (num_min + num_max)
            diff_beta2  = diff_beta2  / num_max

            tmp_alpha0  = self.correct_alpha0.update(diff_alpha0)
            self.alpha0 += tmp_alpha0
            tmp_alpha1  = self.correct_alpha1.update(diff_alpha1)
            self.alpha1 += tmp_alpha1
            tmp_alpha2  = self.correct_alpha2.update(diff_alpha2)
            self.alpha2 += tmp_alpha2
            tmp_beta2   = self.correct_beta2.update(-diff_beta2)
            self.beta2  += tmp_beta2
            
            if self.fix_intercept == None:   # 切片の指定がない場合
                tmp_beta1  = self.correct_beta1.update(diff_beta1)
                self.beta1 += tmp_beta1
                
                update_diff = np.sqrt(diff_alpha0 ** 2 + np.sum(diff_alpha1 ** 2) + diff_alpha2 ** 2 + diff_beta1 ** 2 + diff_beta2 ** 2)
                update      = np.sqrt(tmp_alpha0  ** 2 + np.sum(tmp_alpha1  ** 2) + tmp_alpha2  ** 2 + tmp_beta1  ** 2 + tmp_beta2  ** 2)
                now_ite     = now_ite + 1
                
            else:                            # 切片の指定がある場合
                update_diff = np.sqrt(diff_alpha0 ** 2 + np.sum(diff_alpha1 ** 2) + diff_alpha2 ** 2 + diff_beta2 ** 2)
                update      = np.sqrt(tmp_alpha0  ** 2 + np.sum(tmp_alpha1  ** 2) + tmp_alpha2  ** 2 + tmp_beta2  ** 2)
                now_ite     = now_ite + 1

            if (now_ite % 10 == 0) and visible_flg:# 学習状況の可視化
                mse = (np.sum(ΔLoss_min ** 2) + np.sum(ΔLoss_max ** 2)) / (num_min + num_max)
                print(f"ite:{now_ite}  alpha0:{self.alpha0}  alpha1:{self.alpha1}  alpha2:{self.alpha2}  beta1:{self.beta1}  exp(beta2):{np.exp(self.beta2)}  update_diff:{update_diff}  update:{update}  MSE:{mse}", flush=True)

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

    def predict(self, x_test):
        if type(x_test) is pd.core.frame.DataFrame:
            x_test = x_test.to_numpy()
        
        if type(x_test) is list:
            x_test = np.array(x_test)
        
        if x_test.ndim != 2:
            print(f"x_train dims = {x_test.ndim}")
            print("エラー：：次元数が一致しません。")
            return False

        if self.fix_intercept != None:
            self.beta1 = self.fix_intercept
        
        if self.isStandardization:
            x_test = (x_test - self.x_standardization[0]) / self.x_standardization[1]

        output = self.alpha0 * normal_4th_cumulative(np.sum(self.alpha1 * x_test, axis=1) + self.alpha2) + self.beta1
        if self.isStandardization:
            output = output * self.y_standardization[1] + self.y_standardization[0]
        
        return output
