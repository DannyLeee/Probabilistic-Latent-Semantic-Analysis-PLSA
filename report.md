<p style="text-align:right;">
姓名:李韋宗<br>
學號:B10615024<br>
日期:2020/11/26<br>
</p>

<h1 style="text-align:center;"> Homework 4: PLSA

## 建置環境與套件
* Python 3.6.9, numpy, collections.Counter, tqdm.tqdm, datetime.datetime, timezone, timedelta, functools.reduce, argparse.ArgumentParser, ArgumentDefaultsHelpFormatter, pickle, scipy.sparse, csr_matrix, numba.jit

## 資料前處理
* 按照 doc_list.txt/query_list.txt 的順序讀入文章及 query
* 將所有文章及 query 存成 list of string 方便後續操作
    * 並且集中 file I/O 的時間
    * 同時用 `Counter` 計算字典型態的 TF
* 透過 TF 計算字典型態的 DF
    * 過濾出 term ferquency 較高的前 10000 個字當單字

## 模型參數調整
* TF 使用出現文字出現在文章中的次數
* latent topic number $K$: 16
* $\alpha = 0.7$
* $\beta = 0.2$
* EM iteration step: 30

## 模型運作原理
* 為節省記憶體使用量，將單字表壓縮在 term ferquency 較高的前 10000 個字當字典
* 透過 numba.jit 加速 EM 迭代計算

## 心得
* 最開始先縮小字典大小為 query 所出現的單字，修改 EM 迴圈中的索引，把依次索引改為矩陣運算，稍微加速。然而，還沒改變 baseline 前，一直做不出好的結果，最低的 baseline 更新與時做細節公布後，發現 unigram 與 background LM 的地方有算錯，修改後竟直接超過第 2 個 baseline，接著就開始修改 PLSA 的公式，發現最開始初始化 $P(w_i|T_k)$ 時，機率限制相加為 1 的維度錯誤，修改後卻仍無法超過自己不用 PLSA 的分數。我認為字典大小影響很大，因為 PLSA 就是可以理解同義字的相關性，然而一開始卻只用 query 出現的字，完全喪失了 PLSA 的原意，於是加大字典到 1 萬個字。而速度卻出現了瓶頸，於是參照發布在 kaggle 的做法，使用 numba 大幅加速，成功突破自己的 baseline。最後加大 K 的數量到 16 又有更好的提升，逼近最高的 baseline，原本打算再增加 K 到 32，無奈記憶體大小不夠，加上 numba 無法使用純 list 或 sparse 且速度又沒有 numba 快，於是放棄嘗試增加 K。另外，也試過 unigram LM 的 tf 調整或其他公式的平方也都無法突破最高分，最後作罷。