#!/usr/bin/env python

"""orient.py

Analysis on KNN-Classification:-
    For KNN Classification, we used the Eucledian Distance as a function to calculate the nearest neighbours. Problem we faced during the
    classification was that the training set took too much time classify. we tried out with different values of K and tried out different
    optimisation methods. One of the methods we came across was to convert the RGB File into a monochrome values. KNN Classification which
    we implemented gave us an accuracy of 65% But the problem was it took a long time to finish. So, the KNN Classification is an average
    level algorithm which in a way uses to brute force to identify K-Neighbours to the test set and select the majority class.
    For improving the KNN in future, we have come up with a way to normalize training data values in each iteration so that the classification
    is done in a more efficient way.
    Accuracy :  62.8844114528
    Matrix :
    ================================================================================
    Value :		0		90		180		270
    ================================================================================
    0	:	    198     12 		24 		5
    90	:	    62 	    137 	6 		19
    180	:	    84 	    23 		126 	3
    270	:	    67 		30 		15 		132
Analysis on Nueral Network:-
    For Nueral Network Algorithm, we used the standard method to assign random weights to each nodes in the network and based on the output
    value and target value we adjusted the values of the weights using the derivatives of the error rate. This step allowed us to adjust the
    weights of the network to a good distribution. This step, however didn't give us an expected value. So, we tried to run the test set for
    the weights which our network calculated. This yielded us a accuracy of 30% which was really poor. We should have understood the algorithm
    to a greater extent and so we get the weights in such a way that it works on the test data. We were not able to implement the algorithm
    effectively. Our Code has all the implementation of algorithms correctly. We were just not able to implement if fully.
Analysis on Best Algorithm:-
    For the BEST algorithm, we have tried multiple approaches. There are as follows.

    1. Conversion from RGB to Greyscale. To increase efficiency and enhance the accuracy, we tired converting the training data
    into greyscale mode using the standard conversion technique. We understood that for identifying the orientation of an image,
    a greyscale image could deliver just as good of information that a RGB image could provide.
    We generated the following code to generate the file file. (File also uploaded).

    hand = open("train-data.txt","r")
    trainf= hand.readlines()
    outph=open('model_file','w+')

    for i in trainf:
        outputlist = []
        tmp  = i.split()
        new_tmp = tmp[0]+" "+tmp[1]+" "
        for j in range(2,len(tmp),3):
            tmp2 = ((.299*float(tmp[j]))+(float(tmp[j+1])*.587)+(float(tmp[j+2])*.114))
            new_tmp += str(tmp2)+ " "
        new_tmp += "\n"
        outputlist = [str(i) for i in outputlist]
        str = tmp[0]+" "+tmp[1]+" "+outputlist+"\n"
        outph.write(new_tmp)

    RESULTS  :
    1. We were however not able to completely implement the new model on time, hence we resorted to second approach:

    2. Optimal value of "k" :
    After multiple iterations we tried playing with different distance functions and parameter ranges of K, and come up with the best algorithm.
    It gives an accuracy of 66% with k = 60 and Euclidian function for the distance. Time taken  = 35 minutes

Description :."""

__author__ = "Vikas Reddy,Prashanth Balasubramani"

import numpy as np
import sys
import random
import math
import time

class Classifier:
    def __init__(self):
        self.test_lst = []
        self.train_lst = []
        self.min_val = 100000000000000
        self.orientation = []

    def train(self, trainfile, test_file, k, knn, type=''):
        op_file = open("knn_output.txt","w+")
        count = 0
        tot_cnt = 0
        acc_mat = {}
        acc_mat["0"] = {}
        acc_mat["90"] = {}
        acc_mat["180"] = {}
        acc_mat["270"] = {}
        for i in ["0","90","180","270"]:
            for j in [0,90,180,270]:
                acc_mat[i][j] = 0
        fl = open(test_file, "r")
        f1 = fl.readlines()
        for line in f1:
            tot_cnt += 1
            total = 0
            kn = k
            k_num = knn
            tmp = line.split()
            for i in range(2,len(tmp)):
                self.test_lst.append(tmp[i])
            self.test_lst = [int(i) for i in self.test_lst]
            fl = open(trainfile,"r")
            f1 = fl.readlines()
            test = np.array(self.test_lst)
            for ln in f1:
                kn -= 1
                if kn==0:
                    break
                tmp1 = ln.split()
                for i in range(2,len(tmp1)):
                    self.train_lst.append(tmp1[i])
                self.train_lst = [int(i) for i in self.train_lst]
                train = np.array(self.train_lst)
                temp_dist = test - train
                list_new = temp_dist
                for i in list_new:
                    total += i * i
                #print total
                total = math.sqrt(total)
                self.train_lst = []
                if total <= self.min_val:
                    self.min_val = total
                    k_num -= 1
                    if k_num == 0:
                        self.orientation.pop()
                    self.orientation.append(tmp1[1])
            cnt = []
            cnt.append(self.orientation.count("0"))
            cnt.append(self.orientation.count("90"))
            cnt.append(self.orientation.count("180"))
            cnt.append(self.orientation.count("270"))
            #print "Count :",cnt
            key = cnt.index(max(cnt))
            if key == 0:
                label = 0
            elif key == 1:
                label = 90
            elif key == 2:
                label = 180
            elif key == 3:
                label = 270
            if int(tmp[1]) == label:
                count += 1
            op = tmp[0]+" " +tmp[1] +" " + str(label) +"\n"
            op_file.write(op)
            acc_mat[tmp[1]][label] += 1
            self.test_lst = []
            self.min_val = 100000000000000
            self.orientation = []
        #print "Correct :",count
        crct = (count/float(tot_cnt))*100
        print "Accuracy : ",crct
        print "Matrix : "
        print "===="*20
        print "Value :" + "\t\t" + "0"+"\t\t"+"90"+"\t\t"+"180"+"\t\t"+"270"
        print "===="*20
        for i in ["0","90","180","270"]:
            print i+"\t"+":"+"\t",str(acc_mat[i][0]),"\t\t"+str(acc_mat[i][90]),"\t\t"+str(acc_mat[i][180]),"\t\t"+str(acc_mat[i][270])


def nnet(num):
    #Training File
    hand = open("train-data.txt","r")
    trainf= hand.readlines()

    #Number of Nodes in the hidden Layer
    n = num
    crct = 0
    #Weights for INPUT_LAYER(i), Hidden_LAYER(j) and Output_Layer(k)
    weight_ij = [] #Input to Hidden layer weights
    weight_jk = [] #Hidden to Output layer weights
    sig_k = [] #Sigma Values for Output Layer
    sig_j = [] #Sigma Values for Hidden Layer
    change_wghts_ij = []
    change_wghts_jk = []

    #Function to calculate Sigma Value for each node in output layer
    def osig(output_K,target_k,sig_k):
        sig_k = []
        for i in range(4):
            err = target_k[i] - output_K[i]    # Add to break Value if error rate is less than 0.3
            sig_k.append(err * (1 - (output_K[i]**2)))
        return sig_k

    def nsig(sig_k,output_J,weight_jk):
        sig_j = []
        for i in range(len(output_J)):
            err = 0.0
            for j in range(len(sig_k)):
                err += sig_k[j] * weight_jk[i][j]
            sig_j.append(err * (1 - (output_J[i]**2)))
        #print "Sig_J",sig_j
        return sig_j

    def update_weight(sig_j,sig_k,oij,ojk,change_wghts_ij,change_wghts_jk,weight_ij,weight_jk,ip):
        cnst1 = 0.5
        cnst2 = 0.1
        for i in range(len(sig_j)):
            for j in range(len(sig_k)):
                change = sig_k[j] * oij[i]
                weight_jk[i][j] += cnst1*change + cnst2*change_wghts_jk[i][j]
                change_wghts_jk[i][j] = change

        for i in range(len(weight_ij)):
            for j in range(len(sig_j)):
                change = sig_j[j] * ip[i]
                weight_ij[i][j] += cnst1*change + cnst2*change_wghts_ij[i][j]
                change_wghts_ij[i][j] = change


        return weight_ij,weight_jk

    def calc_err(orient):
        targetlist = []
        for i in [0,90,180,270]:
            if i==orient:
                targetlist.append(1)
            else:
                targetlist.append(0)
        return targetlist

    acc_mat = {}
    acc_mat["0"] = {}
    acc_mat["90"] = {}
    acc_mat["180"] = {}
    acc_mat["270"] = {}
    for i in ["0","90","180","270"]:
        for j in [0,90,180,270]:
            acc_mat[i][j] = 0

    for i in range(0,192):
        tmp = []
        for j in range(n):
            tmp.append(random.uniform(-0.0000002, 0.0000002))
        weight_ij.append(tmp)
        change_wghts_ij.append([0.0]*n)

    for i in range(0,n):
        tmp = []
        for j in range(4):
            tmp.append(random.uniform(-20.0, 20.0))
        weight_jk.append(tmp)
        change_wghts_jk.append([0.0]*4)

    #print "Weights IJ",weight_ij
    #print "Weights JK",weight_jk

    for j in trainf:
        x = j.split()
        orient = int(x[1])   #Target Orientation of the Image
        filename = x[0]
        x = x[2:]

        for i in range(10):
            oij = []
            ojk = []
            for i in range (0,n):           #Calculating O/P of N nodes in the Hidden Layer
                sumz = 0
                for j in range (0,192):
                    x[j] = int(x[j])
                    sumz = sumz + (x[j]*weight_ij[j][i])
                #sumz = sumz * -1
                #print "SUM:",sumz
                gofx = math.tanh(sumz)
                oij.append(gofx)



            for k in range (0,4):           #Calculating O/P of the nodes in the Output Layer
                sumz = 0
                for j in range (0,n):
                    sumz = sumz + (oij[j]*weight_jk[j][k])
                #sumz = sumz * -1
                gofx = math.tanh(sumz)
                ojk.append(gofx)   #Ouput of O/P Later


            #key = int(orient)/90
            #if ojk[key] > 0.7:
            #    break

            tk = calc_err(orient)
            sig_k = osig(ojk,tk,sig_k)
            #print "Sigma_K",sig_k
            sig_j = nsig(sig_k,oij,weight_jk)

            weight_ij,weight_jk = update_weight(sig_j,sig_k,oij,ojk,change_wghts_ij,change_wghts_jk,weight_ij,weight_jk,x)
        #print "Weights of IJ",weight_ij
        #print "Weights of JK",weight_jk


    hand = open("test-data.txt","r")
    testf= hand.readlines()
    crct = 0
    cnt = 0
    for j in testf:
        cnt +=1
        x = j.split()
        orient = int(x[1])   #Target Orientation of the Image
        filename = x[0]
        x = x[2:]
        oij = []
        ojk = []
        for i in range (0,n):           #Calculating O/P of N nodes in the Hidden Layer
            sumz = 0
            for j in range (0,192):
                x[j] = int(x[j])
                sumz = sumz + (x[j]*weight_ij[j][i])
            gofx = math.tanh(sumz)
            oij.append(gofx)

        for k in range (0,4):           #Calculating O/P of the nodes in the Output Layer
            sumz = 0
            for j in range (0,n):
                sumz = sumz + (oij[j]*weight_jk[j][k])
            gofx = math.tanh(sumz)
            ojk.append(gofx)

        #print "OJK Value is ",new_ojk
        key = ojk.index(max(ojk))
        #print "Key is ",key
        if key == 3:
            label = "270"
        elif key == 2:
            label = "180"
        elif key == 1:
            label = "90"
        elif key == 0:
            label = "0"
        if orient == int(label):
            crct += 1
        #print filename,orient,label
        acc_mat[tmp[1]][label] += 1
    print "Accuracy :",(crct/float(cnt))*100
    print "Matrix : "
    print "===="*20
    print "Value :" + "\t\t" + "0"+"\t\t"+"90"+"\t\t"+"180"+"\t\t"+"270"
    print "===="*20
    for i in ["0","90","180","270"]:
        print i+"\t"+":"+"\t",str(acc_mat[i][0]),"\t\t"+str(acc_mat[i][90]),"\t\t"+str(acc_mat[i][180]),"\t\t"+str(acc_mat[i][270])


def main(train_file,test_file,method,option):
    if method == 'knn':
        x1 = Classifier()
        x1.train(train_file,test_file,5000,int(option))
    elif method == 'nnet':
        nnet(int(option))
    elif method == 'best':
        x1 = Classifier()
        #train_file = option
        opt = 60
        x1.train(train_file,test_file,5000,opt,method)

start_time = time.time()
train_file = sys.argv[1]
test_file = sys.argv[2]
method = sys.argv[3]
option = sys.argv[4]
main(train_file,test_file,method,option)
#print "--- %s seconds ---" % (time.time() - start_time)
