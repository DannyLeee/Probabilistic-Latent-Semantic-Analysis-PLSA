#%%
import numpy as np
from collections import Counter
from tqdm import tqdm
from datetime import datetime,timezone,timedelta
from functools import reduce
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import random
import pickle

def timestamp(msg=""):
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    print(str(dt2)[:-13] + '\t' + msg)

def file_iter(_type):
    if _type == "q":
        for name in q_list:
            with open(query_path+name+'.txt') as f:
                yield f.readline()
    elif _type == "d":
        for name in d_list:
            with open(doc_path+name+'.txt') as f:
                yield f.readline()

#%%
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-from_scratch", type=int, default=0, help="default load computed tf, BG and voc")
parser.add_argument("-train_from", type=int, default=0)
parser.add_argument("-A", type=float, metavar="alpha", required=True)
parser.add_argument("-B", type=float, metavar="beta", required=True)
parser.add_argument("-K", type=int, required=True, help="K latent topic")
parser.add_argument("-step", type=int, required=True, help="EM iterattion time")
args = parser.parse_args()

A = args.A
B = args.B
K = args.K

#%%
timestamp()
doc_path = "../HW4 PLSA/data/docs/"
query_path = "../HW4 PLSA/data/queries/"

d_list = []
with open('../HW4 PLSA/data/doc_list.txt', 'r') as d_list_file:
    for line in d_list_file:
        line = line.replace("\n", "")
        d_list += [line]

q_list = []
with open('../HW4 PLSA/data/query_list.txt', 'r') as q_list_file:
    for line in q_list_file:
        line = line.replace("\n", "")
        q_list += [line]

#%%
# tf
list_q_tf = []
list_d_tf = []
query_list = []
timestamp("counter TF")
for txt in tqdm(file_iter("q")):
    list_q_tf += [Counter(txt.split())]
    query_list += [txt]

#%%
doc_len = []
for txt in tqdm(file_iter("d")):
    if args.from_scratch:
        list_d_tf += [Counter(txt.split())]
    doc_len += [len(txt)]
total_size = sum(doc_len)

if not args.from_scratch or args.train_from != 0:
    with open("model/tf.pkl", "rb") as fp:
        list_d_tf = pickle.load(fp)

#%%
if not args.from_scratch or args.train_from != 0:
    with open("model/voc.pkl", "rb") as fp:
        voc = pickle.load(fp)
else:
    # voc
    voc = reduce(set.union, map(set, map(dict.keys, list_q_tf))) ##### small test
    voc = list(voc)

#%%
# tf array
timestamp("counter to array")

if not args.from_scratch or args.train_from != 0:
    tf_array = np.load("model/tf_array.npy")
else:
    tf_array = np.zeros([len(voc), len(d_list)]) # V*D tf array
    for i, w_i in tqdm(enumerate(voc)):
        for j in range(len(d_list)):
            tf_array[i][j] = list_d_tf[j][w_i]

#%%
# df
timestamp("df")

# if True:
if not args.from_scratch or args.train_from != 0:
    with open("model/BG.pkl", "rb") as fp:
        BG_counter = pickle.load(fp)
else:
    BG_counter = Counter()
    for c in tqdm(list_d_tf):
        BG_counter += c

#%%
# EM
T_given_wd = np.zeros([K, len(voc), len(d_list)]) # K*i*j matrix
if args.train_from == 0:
    # initial
    timestamp("random initial")
    while True:
        w_given_T = []
        # i*K random distribution matrix
        for _ in tqdm(range(len(voc))):
            temp = np.array([random.random() for __ in range(K)])
            w_given_T += [temp / temp.sum()]
        w_given_T = np.array(w_given_T)
        T_given_d = np.array(K * [[1/K for _ in range(len(d_list))]]) # K*j uniform distribution matrix
        # T_given_d = 1/K # uniform distribution matrix

        for i in range(len(voc)):
            if abs(w_given_T[i].sum() - 1) > 0.1:
                print("fk")
        for j in range(len(d_list)):
            if abs(T_given_d[:, j].sum() - 1) > 0.1:
                print("fk")

        if np.count_nonzero(np.isnan(T_given_wd)) + np.count_nonzero(np.isnan(w_given_T)) + np.count_nonzero(np.isnan(T_given_d)) == 0:
            np.save(f"./model/P(T|w,d)_init", T_given_wd)
            break
else:
    w_given_T = np.load("./model/P(w|T)_" + str(args.train_from) + ".npy")
    T_given_d = np.load("./model/P(T|d)_" + str(args.train_from) + ".npy")

#%%
loss = 0.0
for step in tqdm(range(args.train_from, args.step)):
    # E-step
    timestamp(f"\n{step+1}\t--E-step--")
    timestamp("P(T|w,d)")
    for k in range(K):
        denominator = np.matmul(w_given_T, T_given_d)
        for i in range(len(voc)):
            for j in range(len(d_list)):
                if denominator[i][j] == 0:
                    T_given_wd[k][i][j] = 0
                else:
                    T_given_wd[k][i][j] = w_given_T[i][k]*T_given_d[k][j] / denominator[i][j]

    #%%
    # M-step
    timestamp("--M-step--")
    timestamp("P(w|T)")
    for k in range(K):
        denominator = (tf_array*T_given_wd[k]).sum()
        for i in range(len(voc)):
            w_given_T[i][k] = (tf_array[i]*T_given_wd[k][i]).sum()
        w_given_T[:, k] /= denominator

    # for i in range(len(voc)):
    #     w_given_T[i] /= w_given_T[i].sum() # standardization

    #%%
    timestamp("P(T|d)")
    for k in range(K):
        for j in range(len(d_list)):
            for i, w_i in enumerate(voc):
                T_given_d[k][j] += list_d_tf[j][w_i] * T_given_wd[k][i][j]
        T_given_d[k] = T_given_d[k] / doc_len[j]

    # for j in range(len(d_list)):
    #     T_given_d[:, j] /= T_given_d[:, j].sum()
    timestamp("---")

    #%%
    # Loss
    last_loss = loss
    loss = 0.0
    w_given_d = np.log((1/K)*np.matmul(w_given_T, T_given_d))
    for i in range(len(voc)):
        for j in range(len(d_list)):
            loss += tf_array[i][j]*w_given_d[i][j]
    timestamp(f"step_{step+1}\tLoss: {loss:.2f}")

    np.save(f"./model/P(T|w,d)_{step+1}", T_given_wd) 
    np.save(f"./model/P(w|T)_{step+1}", w_given_T)
    np.save(f"./model/P(T|d)_{step+1}", T_given_d)

#%%
# score for each qd-pair
"""P(q|d_j) = production_for_i_in_query_len[A*P(w_i|d_j) + B*(SUM_for_k_in_K(P(w_i|T_k)*P(T_k|d_j))) + (1-A-B)*P_BG(w_i)]"""
timestamp("sim_array")
sim_array = np.ones([len(q_list), len(d_list)])
for q, query in tqdm(enumerate(query_list)):
    for j in range(len(d_list)):
        for w_i in query.split():
            i = voc.index(w_i)
            term1 = A * (list_d_tf[j][w_i] / doc_len[j])
            term2 = 0
            for k in range(K):
                term2 += w_given_T[i][k] * T_given_d[k][j]
            term2 *= B
            term3 = (1-A-B) * (BG_counter[w_i] / total_size)
            sim_array[q][j] *= (term1 + term2 + term3)

#%%
# output
timestamp("output")
with open('result.csv', 'w') as output_file:
    output_file.write("Query,RetrievedDocuments\n")
    for i, q_id in tqdm(enumerate(q_list)):
        output_file.write(q_id+',')
        sorted = np.argsort(sim_array[i])
        sorted = np.flip(sorted)[:1000]
        for _, j in enumerate(sorted):
            output_file.write(d_list[j]+' ')
        output_file.write('\n')
timestamp()