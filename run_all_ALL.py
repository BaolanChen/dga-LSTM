import unigram_run as uni
import bigram_run as bi

uni.main(
        'new_data/dga.csv','new_data/normal.csv',
        'unigram/ALL-unigram-lstm-ROCimg.png',
        'unigram/ALL-unigram-lstm.h5',
        'unigram/ALL-unigram-lstm.txt',
        ifgetROCimg=True,
        model_flag='lstm',
        max_epoch=2, nfolds=1,batch_size=128, 
        )

uni.main(
        'new_data/dga.csv','newtest_data/normal.csv',
        'unigram/ALL-unigram-bilstm-ROCimg.png',
        'unigram/ALL-unigram-bilstm.h5',
        'unigram/ALL-unigram-bilstm.txt',
        ifgetROCimg=True,
        model_flag='bilstm',
        max_epoch=2,nfolds=1,batch_size=128, 
        )

bi.main(
        'new_data/dga-bigram.csv','new_data/normal-bigram.csv',
        'bigram/ALL-bigram-lstm-ROCimg.png',
        'bigram/ALL-bigram-lstm.h5',
        'bigram/ALL-bigram-lstm.txt',
        ifgetROCimg=True,
        model_flag='lstm',
        max_epoch=2, nfolds=1,batch_size=128, 
        )

bi.main(
        'new_data/dga-bigram.csv','new_data/normal-bigram.csv',
        'bigram/ALL-bigram-bilstm-ROCimg.png',
        'bigram/ALL-bigram-bilstm.h5',
        'bigram/ALL-bigram-bilstm.txt',
        ifgetROCimg=True,
        model_flag='bilstm',
        max_epoch=2, nfolds=1,batch_size=128, 
        )