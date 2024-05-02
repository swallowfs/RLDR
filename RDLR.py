import random

import numpy as np
import pandas as pd
import os

def load_case():
    df=pd.read_csv('5_1.csv',header=None)
    return df.values

def load_case_time():
    #所有case穷举过
    df=pd.read_csv('5_2_2_4_0.csv',header=None)
    return df.values

def load_matrix():
    matdata=pd.read_csv("download.csv",encoding='utf-8',header=None)
    return matdata.values

def load_datavolume():
    datavolume=pd.read_csv('swimres.csv',encoding='utf-8')
    datavolume=datavolume.values
    datavolume=datavolume[0]
    return datavolume

def loadfnum(dnum):
    dirpath = os.getcwd() + '/' + str(dnum) + '_1/'
    filelist = os.listdir(dirpath)
    print(filelist)
    fnum = len(filelist)
    return fnum


class Particle:
    def __init__(self, d, r, k, min_values, max_values):
        n=d+r
        self.L = np.random.uniform(min_values, max_values, size=(n,n))
        self.best_value = float('inf')
        self.optimaltime = float('inf')
        self.velocity = 0
        self.solstr = ''
        self.d = d
        self.k = k
        self.r = r


#输出三元组[from(主),to()，期望时间]
def objective_function_list(L, currentset, otherset):
    index=0
    curlen=len(currentset)
    otherlen=len(otherset)
    linknum=curlen*otherlen
    resdata = np.zeros((linknum,3))
    for i in currentset:
        for j in otherset:
            resdata[index,0]=i
            resdata[index,1]=j
            resdata[index,2]=L[i, j]
            index+=1
    return resdata

#选择一个链路,算出当前
def objective_function(L, currentset, otherset, resdata=None):
    resdata=objective_function_list(L, currentset, otherset, resdata=None)
    regtime=np.max(resdata[:,0])
    return regtime


#选择一个链路
#输出两元组
def select_link(inset):
    Lvalue = inset[:, 2]
    Lmax=np.max(Lvalue)
    Vvalue=Lmax-Lvalue
    Vsum=sum(Vvalue)
    #根据value来选择link
    selection_point=random.uniform(0,Vsum)
    current_sum=0
    for i,item in enumerate(Vvalue):
        current_sum += item
        if current_sum >= selection_point:
            return inset[i,0], inset[i,1]

def seq_tree(currentset,case1):
    index=0
    currentstr=np.array2string(currentset)
    currentstr = currentstr.replace("[", "")
    currentstr = currentstr.replace("]", "")
    currentstr = currentstr.replace(" ", "")
    for item in case1:
        if item==currentstr:
            return index
        index+=1

def initilize_list(arr):
    column_of_zeros = np.zeros((arr.shape[0], 1))
    new_arr = np.hstack((arr, column_of_zeros))
    return new_arr
def Update_set(currentset,otherset,regtree,sounode,nextnode):
    sounode=int(sounode)
    nextnode=int(nextnode)
    currentset.remove(sounode)
    otherset.append(sounode)
    regtree[sounode]=nextnode
    return currentset,otherset,regtree
def prim_algorithm(Lmax,r,d,case1):
    regtree=np.zeros(r)
    for i in range(r):
        newcom=d
        currentset=list(range(0,d))
        otherset=[newcom]
        oneregtree=np.zeros(d)
        for j in range(d):
            inset=objective_function_list(Lmax, currentset, otherset)
            sounode,nextnode=select_link(inset)
            currentset,otherset,oneregtree=Update_set(currentset,otherset,oneregtree,sounode,nextnode)
        #完成了我来对比一下是那个数据
        treeid=seq_tree(oneregtree,case1)
        regtree[i]=treeid
    return regtree

def convert_to_str(topdata):
    #将regtree转化为string
    topresult = np.array2string(topdata, threshold=np.inf)
    topresult = topresult.replace("[", "")
    topresult = topresult.replace("]", "")
    topresult = topresult.replace(" ", "")
    return topresult
def find_newcase(caseadd):
    #调整到最近一个没有搜索到的发地址
    #寻找顺序下面第一个非零元素
    #重现在特征的下一位开始搜索
    current_case=caseadd[:,3]
    index=np.where(current_case==0)[0]
    if len(index)>0:
        resindex = index[0]
        return resindex
    else:
        return -1
def is_deduplicate(regtree,caseadd,r):
    for i in range(r):
        tmpres=regtree[i]==caseadd[:,i]
        if i==0:
            res = tmpres
        else:
            res = res & tmpres
    tempcaseadd=caseadd[res,:]
    tempcaseadd=tempcaseadd[0]
    index1=r+1
    flag=tempcaseadd[index1]
    value = tempcaseadd[r]
    if flag==0:
        caseadd[res, index1]=1#标记为已访问
    else:
        index=find_newcase(caseadd)
        if index!=-1:
            caseadd[index,index1]=1
            tempcaseadd=caseadd[index,:]
            #tempcaseadd=tempcaseadd[0]
            regtree=tempcaseadd[0:r]
            value=tempcaseadd[r]
        else:
            return -1,-1,-1
    return regtree,caseadd,value
def case_min_value(caseadd,r):
    tempcase=caseadd[:,r]
    minvalue=np.min(tempcase)
    return minvalue
def particle_swarm_optimization(objective_function, case1,caseadd,
                                d, r, k,
                                inertia_weight,cognitive_weight,social_weight,
                                min_values,max_values,
                                num_particles, num_iterations):

    swarm = [Particle(d,k,r, min_values, max_values) for _ in range(num_particles)]
    n=d+r
    global_best_L = np.random.uniform(min_values, max_values, n)
    global_best_value = float('inf')
    dim=1
    minlist=np.zeros(num_iterations)
    minvalue=case_min_value(caseadd,r)
    for index in range(num_iterations):

        #防止冲突
        #regtreestr=convert_to_str(regtree)
        #if     调查 is_deduplicate()==T
        for particle in swarm:
            regtree = prim_algorithm(particle.L, r, d, case1)
            regtree,caseadd,value = is_deduplicate(regtree,caseadd,r)
            #value = objective_function(particle.L, regtree)
            if value < particle.best_value:
                particle.best_position = particle.L
                particle.best_value = value
            if value < global_best_value:
                global_best_L = particle.L
                global_best_value = value
        for particle in swarm:
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            particle.velocity = (inertia_weight * particle.velocity +
                                 cognitive_weight * r1 * (particle.best_position - particle.L) +
                                 social_weight * r2 * (global_best_L - particle.L))
            particle.L += particle.velocity
            # Ensure particles stay within bounds
            particle.L = np.clip(particle.L, min_values, max_values)
        minlist[index] = global_best_value
    return minlist, global_best_value,minvalue
def record_result(minlist,inertia_weight,cognitive_weight,social_weight,num_particles,rndnum):
    #记录已输出参数结果
    inertia_weight_str=str(inertia_weight)
    cognitive_weight_str=str(cognitive_weight)
    social_weight_str=str(social_weight)
    num_particles_str=str(num_particles)
    rndnum_str=str(rndnum)
    filename='./'+inertia_weight_str+'_'+cognitive_weight_str+'_'+social_weight_str+'_'+num_particles_str+'_'+rndnum_str+'.csv'
    np.savetxt(filename,minlist)
if __name__ == "__main__":
    case1 = load_case()
    caseadd = load_case_time()
    caseadd = initilize_list(caseadd)
    print(case1)

    min_values= 0
    max_values= 1000
        #(float('inf'))
    num_particles = 20
    num_iterations = 100

    inertia_weight = 0.7
    cognitive_weight = 1.5
    social_weight = 1.5

    d=5
    k=2
    r=2
    rndnum=10
    for i in range(rndnum):
        num_particles=20
        for inertia_weight in [0.1,0.3,0.5,0.7,0.9]:
            minlist, searchvalue, minvalue = particle_swarm_optimization(objective_function, case1, caseadd,
                                                                         d, r, k,
                                                                         inertia_weight, cognitive_weight, social_weight,
                                                                         min_values, max_values,
                                                                         num_particles, num_iterations)
            record_result(minlist, inertia_weight, cognitive_weight, social_weight, num_particles,i)
            print(inertia_weight)
        inertia_weight=0.7
        for cognitive_weight in [0.5,1.0,1.5,2.0,2.5]:
            minlist, searchvalue, minvalue = particle_swarm_optimization(objective_function, case1, caseadd,
                                                                         d, r, k,
                                                                         inertia_weight, cognitive_weight, social_weight,
                                                                         min_values, max_values,
                                                                         num_particles, num_iterations)
            record_result(minlist, inertia_weight, cognitive_weight, social_weight, num_particles,i)
            print(cognitive_weight)
        cognitive_weight = 1.5
        for social_weight in  [0.5,1.0,1.5,2.0,2.5]:
            minlist, searchvalue, minvalue = particle_swarm_optimization(objective_function, case1, caseadd,
                                                                         d, r, k,
                                                                         inertia_weight, cognitive_weight, social_weight,
                                                                         min_values, max_values,
                                                                         num_particles, num_iterations)
            record_result(minlist, inertia_weight,cognitive_weight, social_weight,num_particles,i)
            print(social_weight)
        social_weight=1.5
        for num_particles in [10,15,20,25,30]:
            minlist, searchvalue, minvalue = particle_swarm_optimization(objective_function, case1,caseadd,
                                                        d,r, k,
                                                        inertia_weight,cognitive_weight,social_weight,
                                                        min_values, max_values,
                                                        num_particles, num_iterations)
            record_result(minlist,inertia_weight,cognitive_weight,social_weight,num_particles,i)
            print(num_particles)

