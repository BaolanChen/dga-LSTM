"""Train and test LSTM classifier"""
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional,Flatten
import sklearn
from sklearn.cross_validation import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
import itertools
import domain_a

def build_LSTM_model(max_features, maxlen):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')
    return model
def build_BiLSTM_model(max_features, maxlen):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')
    return model

def run(
        dgacsv_path,normalcsv_path,    
        model_save_path='model.h5',
        matrix_save_path='matrix.txt',
        model_flag = 'lstm',
        max_epoch=25, nfolds=10,batch_size=128,
        ):
    #从csv文件构建dga和normal数据集
    dga = pd.read_csv(dgacsv_path)
    normal = pd.read_csv(normalcsv_path)
    
    #获取两种数据集的标签
    dga_y = list(dga['labels'])
    normal_y = list(normal['labels'])
    X = list(dga['dga']) + list(normal['normal'])
    y = dga_y + normal_y

    #maxlen = domain_a.main()
    maxlen = 38     #model的input_length，所有域名里数量占有99.9%的域名的长度界限

    a =sorted(list(set(''.join((X))))) # 得到所有域名里出现过的字符，并按照ascii顺序排序，用于固定编码顺序
    valid_chars = {x:idx+1 for idx, x in enumerate(a)} # 从1开始编码，得到编码字典

    with open('tran.txt','w') as file:
        for i in valid_chars:
            file.write(i+' '+str(valid_chars[i])+'\n') 
    file.close()

    max_features = len(valid_chars) + 1 #字典长度，model中input_dim

    X = [[valid_chars[y] for y in x] for x in X] # 将X里的域名按每个字符进行编码，如['google',...]->[[2,5,5,2,45,6],...]
    X = sequence.pad_sequences(X, maxlen=maxlen) # 按照maxlen补0并转为矩阵，如max_len=8, [[2,5,5,2,45,6],...]->[[0 0 2 5 5 2 45 6] [..]...]

    final_data = []
    for fold in range(nfolds): # 设置nfolds，每for进行一个nfolds就从数据集重新选择训练集、测试集，用新选的数据集再训练。
        print ("fold %u/%u" % (fold+1, nfolds)) # 输出当前fold
        X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2)#按4:1选出训练集和测试集

        print ('Building model')
        if model_flag=='lstm':
            model = build_LSTM_model(max_features, maxlen)
            print('LSTM model has been built.')
        elif model_flag == 'bilstm':
            model = build_BiLSTM_model(max_features, maxlen)
            print('BiLSTM model has been built.')
        else:
            print("model_flag is error!")
            return
        
        model.summary()
        print ("Start train!")
        X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05)#从训练集里再按照19:1选出小训练集和小测试集
        best_iter = -1 #记录最好一轮
        best_auc = 0.0 #记录最好的准确值
        out_data = {}  #记录每训练一次的结果

        for ep in range(max_epoch):#max_epoch手动设置用选取的数据集进行训练的轮数，下面nb_epochs设置为1就是每次fit一轮，而用max_epoch训练多次。这样做等价于fit(...,epochs=max_epoch)。但是这样可以把每次fit的结果保存下来，如果直接fit(...,epochs=10)那么只有训练10轮后的结果，每一轮的结果看不到。
            #训练
            model.fit(X_train, y_train, batch_size=batch_size, epochs=1)#nb_epoch等于epochs，在keras2.0后弃用bn_epoch。该参数设置训练的轮数。

            #用holdout部分数据进行预测返回正类的概率估计t_probs，用正确值y_holdout计算准确值。如x_holdout=[23,32,44,11]->t_probs=[0.1,0.3,0.4,0.9]（预测出1的概率），y_holdout=[0,0,1,1]->t_auc=3/4=0.75
            t_probs = model.predict_proba(X_holdout)
            t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)

            print ('Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc))

            if t_auc > best_auc:
                best_auc = t_auc
                best_iter = ep

                probs = model.predict_proba(X_test)#预测，返回概率值，即被预测成y_test的概率。
                smc = sklearn.metrics.confusion_matrix(y_test, probs > .5)#返回[[c00 c01][c10 c11]],c00:normal被判成normal的数量，c01:normal被判成dga的数量，c10:dga被判成normal的数量，c11：dga被判成dga的数量
                out_data = {'y':y_test,  'probs':probs, 'epochs': ep,
                            'confusion_matrix': smc}
                
                
                with open(matrix_save_path,'w') as file:
                    file.write(str(smc))
                file.close()
            else:
                # 比较每次训练后的准确率，如果准确率不再提升，结束训练
                if (ep-best_iter) > 2:
                    break

        final_data.append(out_data)
    model.save(model_save_path)

    return final_data

def calc_macro_roc(fpr, tpr):
    """Calcs macro ROC on log scale"""
    # Create log scale domain
    all_fpr = sorted(itertools.chain(*fpr))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(tpr)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    return all_fpr, mean_tpr / len(tpr), auc(all_fpr, mean_tpr) / len(tpr)

def ROCimg(results,ROCimgSave_path):
    fpr = []
    tpr = []
    for result in results:
        t_fpr, t_tpr, _ = roc_curve(result['y'], result['probs'])
        fpr.append(t_fpr)
        tpr.append(t_tpr)
    binary_fpr, binary_tpr, binary_auc = calc_macro_roc(fpr, tpr)

    # Save figure
    
    with plt.style.context('bmh'):
        plt.plot(
                binary_fpr, binary_tpr,
                label='biLSTM (AUC = %.4f)' % (binary_auc, ), 
                rasterized=True
                )   
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC - Binary Classification')
        plt.legend(loc="upper right")
        plt.tick_params(axis='both')
        plt.tight_layout()
        plt.savefig(ROCimgSave_path)
        plt.close()

def main(
        dga_path,normal_path,
        ROCimgSave_path,
        model_save_path,
        matrix_save_path,
        ifgetROCimg=True,
        model_flag='lstm',
        max_epoch=1, nfolds=1,batch_size=128, 
        ):
    
    results = run(
                        dgacsv_path=dga_path,normalcsv_path=normal_path,
                        model_save_path=model_save_path,
                        matrix_save_path=matrix_save_path,
                        model_flag=model_flag,
                        max_epoch=max_epoch,nfolds=nfolds,batch_size=batch_size,           
                        )
    if ifgetROCimg == True:
        ROCimg(results,ROCimgSave_path)

if __name__ == "__main__":   
    main(
        'data/dga.csv','data/normal.csv',
        'unigram-lstm-ROCimg.png',
        'unigram-lstm.h5',
        'unigram/unigram-lstm.txt',
        ifgetROCimg=True,
        model_flag='lstm',
        max_epoch=25, nfolds=10,batch_size=128, )
