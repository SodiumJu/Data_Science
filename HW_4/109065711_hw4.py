#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import math
# you must use python 3.6, 3.7, 3.8, 3.9 for sourcedefender
import sourcedefender
from HomeworkFramework import Function


# In[370]:


class CMA_ES_optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func, sigma):
        super().__init__(target_func) # must have this init to work normally
        self.random_t=100
        self.terminate=False
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        #print(self.lower)
        #print(self.upper)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func
        self.mean=np.random.uniform(np.full(self.dim, self.lower), np.full(self.dim, self.upper), self.dim)
        #self.mean=np.zeros(self.dim)
        self.std=1
        self.center=(self.upper+self.lower)/2
        self.sigma=sigma
        self.sample_size=4+2*(math.floor(3*math.log(self.dim,10)))
        
        self.eval_times = 0
        self.value = float("inf")
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)
        
        self.C=np.identity(self.dim)
        self.mu=math.floor(self.sample_size/2)
        self.weight=self.get_weights()
        self.parent_samples=[]
        #self.get_random_sol() #for get first parent samples
        #self.initialize_mean()        
        self.child_samples=[]
        
        self.p_c=np.zeros(self.dim)
        self.p_sigma=np.zeros(self.dim)
        
        self.mu_w=1/sum(self.weight**2)
        self.mueff=sum(self.weight)**2/sum(self.weight**2)
        self.c_c=4/self.dim
        #self.c_c=(4+self.mueff/self.dim)/(self.dim+4 + 2*self.mueff/self.dim)
        self.c_sigma=4/self.dim
        #self.c_sigma=(self.mueff+2)/(self.dim+self.mueff+5)
        self.damps = 1 + 2*max(0, math.sqrt((self.mueff-1)/(self.dim+1))-1) + self.c_sigma
          
        self.c_mu=self.mu_w/(self.dim**2)
        #self.c_mu=2 * (self.mueff-2+1/self.mueff) / ((self.dim+2)**2+2*self.mueff/2)
        self.c_1=2/(self.dim**2)
        #self.c_1 = 2 / ((self.dim+1.3)**2+self.mueff)
        
        if((self.c_1+self.c_mu)>1):
            print("ERROR")
        self.y_w=np.zeros(self.dim)
        self.C_mu=np.zeros((self.dim,self.dim))
        self.chiN=math.sqrt(self.dim)*(1-1/(4*self.dim)+1/(21*(self.dim**2)))
        self.d_sigma=1+math.sqrt(self.mu_w/self.dim)
        self.BD=np.identity(self.dim)
        
    def RS_optimizer(self):       
        for i in range(self.random_t):
            solution = np.random.uniform(np.full(self.dim, self.lower), np.full(self.dim, self.upper), self.dim)
            value = self.f.evaluate(func_num, solution)
            self.eval_times += 1
            if value == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                break 
            if value < self.optimal_value:
                self.optimal_solution[:] = solution
                self.optimal_value = float(value)
                self.mean=solution
            
    
    def update_opt_sol(self,sol):       
                   
        if sol[1] < self.optimal_value:
            self.optimal_solution[:] = sol[0]
            self.optimal_value = float(sol[1])
    
    def get_weights(self):
        
        seq = np.arange(self.mu)+1
        weights=-np.log(seq)+math.log((self.mu+0.5))
        weights_1=weights/sum(weights)
        weights_1=weights_1.reshape((weights_1.size, 1))
        
        return weights_1

    def update_p_c(self):
        #self.p_c=(1-self.c_c)*self.p_c+math.sqrt(1-(1-self.c_c)**2)*math.sqrt(self.mu_w)*self.y_w
        self.p_c=(1-self.c_c)*self.p_c+math.sqrt(1-(1-self.c_c)**2)*math.sqrt(self.mu_w)*self.y_w
    def update_p_sigma(self):      
        #self.p_sigma=(1-self.c_c)*self.p_sigma+math.sqrt(1-(1-self.c_sigma)**2)*math.sqrt(self.mu_w)*np.matmul(self.BD,self.y_w.reshape((self.y_w.size, 1)))
        self.p_sigma=(1-self.c_sigma)*self.p_sigma+math.sqrt(1-(1-self.c_sigma)**2)*math.sqrt(self.mu_w)*np.matmul(self.BD,self.y_w.reshape((self.y_w.size, 1))).reshape((self.y_w.size))
    def update_C(self):
        self.C=(1-self.c_1-self.c_mu)*self.C+self.c_1*self.p_c*self.p_c.reshape((self.p_c.size, 1))+self.c_mu*self.C_mu
    def update_sigma(self):
        #self.sigma=self.sigma * math.exp((self.c_sigma/self.damps)*(np.linalg.norm(self.c_sigma)/self.chiN - 1))
        self.sigma=self.sigma*math.exp((self.c_sigma/self.d_sigma)*(np.linalg.norm(self.p_sigma)/self.chiN-1))
    
    def get_samples(self):
                
        self.parent_samples=self.child_samples
        self.child_samples=[]
        self.y_w=np.zeros(self.dim)
        
        
        for i in range(self.sample_size):
            #y_i=np.random.multivariate_normal(np.zeros(self.dim),self.C) #note
            y_i=np.matmul(self.BD,np.random.normal(0,self.std,size=(self.dim)))
            new_sol=y_i*self.sigma+self.mean
            new_sol_c=np.clip(new_sol,self.lower,self.upper)
            out=True                                
            #print(new_sol)
            value = self.f.evaluate(func_num, new_sol_c)
            if(value == "ReachFunctionLimit"):
                self.terminate=True
                break
            self.eval_times=self.eval_times+1
            self.value=value
            self.child_samples.append([new_sol,value])
            #self.child_samples[new_sol]=value
        if(self.terminate==False):  
            Sorted_Sol=self.child_samples+self.parent_samples
            #Sorted_Sol=sorted(Solutions.items(), key=lambda x:x[1])
            Sorted_Sol.sort(key=lambda x:x[1])
            mu_Sol = Sorted_Sol[0 : self.mu]
            mu_Sol=np.array(mu_Sol)

            self.update_opt_sol(mu_Sol[0])
            mu_Sol_x=np.array(np.transpose(mu_Sol)[0])
            #print(mu_Sol_x)
            self.C_mu=np.zeros((self.dim,self.dim))
            new_mean=np.zeros(self.dim)
            self.y_w=np.zeros(self.dim)
            for i in range(self.mu):
                y_i=(mu_Sol_x[i]-self.mean)/self.sigma
                new_mean=new_mean+mu_Sol_x[i]*self.weight[i][0]
                self.y_w=self.y_w+y_i*self.weight[i][0]
                #print(self.weight[0][1])
                self.C_mu=self.C_mu+y_i*y_i.reshape((y_i.size, 1))*self.weight[i][0]
            #self.y_w=(new_mean-self.mean)/self.sigma
            self.mean=new_mean


            self.update_p_c()
            self.update_C()
            self.update_BD()
            self.update_p_sigma()
            self.update_sigma()
        
    def update_BD(self):
        try:
            egv, B = np.linalg.eig(self.C)
            
        except np.linalg.LinAlgError as e:
            self.C=np.nan_to_num(self.C)
            egv, B = np.linalg.eig(self.C)
        self.BD = np.matmul(B,np.diag(np.sqrt(egv)))
        
    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES): # main part for your implementation
        self.random_t=math.floor(FES*0.1)
        if self.eval_times < FES:
            self.RS_optimizer()
        while self.eval_times < FES:
            if(self.terminate==True):
                print("ReachFunctionLimit\n")
                break 
            else:
                print('=====================FE=====================')
                print(self.eval_times)
                self.get_samples()
                print("optimal: {}\n".format(self.get_optimal()[1]))


# In[373]:


if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500

        # you should implement your optimizer
        op = CMA_ES_optimizer(func_num,0.5)
        op.run(fes)
        
        best_input, best_value = op.get_optimal()
        print(best_input, best_value)
        
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format("109065711", func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1 


# In[ ]:




