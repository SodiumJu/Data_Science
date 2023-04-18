#!/usr/bin/env python
# coding: utf-8

# In[506]:


import requests


# In[507]:


import re


# In[508]:


from bs4 import BeautifulSoup


# In[509]:


Nonce="Nonce: "
Number_of_Transactions="Number of Transactions: "
Final_Balance="Final Balance: "
Total_Sent="Total Sent: "
Total_Received="Total Received: "
Total_Fees="Total Fees: "
Date="Date: "
To="To: "
Amount="Amount: "
ETH=" ETH"
Line="--------------------------------------------------------------------------"
Arrow=" -> "

ID="109065711"


# In[510]:


def find_info(divs):
    v_Nonce=re.findall(r'Nonce</span></div></div>.* opacity="1">([0-9]*[\,]?[0-9]+)</span>.*Number of Transactions', str(divs[0]))[0]
    v_Transactions=re.findall(r'Number of Transactions</span></div></div>.*opacity="1">([0-9]*[\,]?[0-9]+)</span>.*Final Balance', str(divs[0]))[0]
    v_Balance=re.findall(r'Final Balance</span></div></div>.*opacity="1">([0-9]+.[0-9]+) ETH</span>.*Total Sent', str(divs[0]))[0]
    v_T_Sent=re.findall(r'Total Sent</span></div></div>.*opacity="1">([0-9]+.[0-9]+) ETH</span>.*Total Received', str(divs[0]))[0]
    v_T_Received=re.findall(r'Total Received</span></div></div>.*opacity="1">([0-9]+.[0-9]+) ETH</span>.*Total Fees', str(divs[0]))[0]
    v_T_Fees=re.findall(r'Total Fees</span></div></div>.*opacity="1">([0-9]+.[0-9]+) ETH</span>.*', str(divs[0]))[0]
    return v_Nonce,v_Transactions,v_Balance,v_T_Sent,v_T_Received,v_T_Fees


# In[511]:


def print_to(not_end,f,v_Nonce,v_Transactions,v_Balance,v_T_Sent,v_T_Received,v_T_Fees,v_T_Date,v_To,v_amount):
    f.write(Nonce+v_Nonce+"\n")
    f.write(Number_of_Transactions+v_Transactions+"\n")
    f.write(Final_Balance+v_Balance+ETH+"\n")
    f.write(Total_Sent+v_T_Sent+ETH+"\n")
    f.write(Total_Received+v_T_Received+ETH+"\n")
    f.write(Total_Fees+v_T_Fees+ETH+"\n")
    if(not_end):
        f.write(Date+v_T_Date+"\n")
        f.write(To+v_To+"\n")
        f.write(Amount+v_amount+ETH+"\n")
    f.write(Line+"\n")


# In[512]:


def tracing_ornot(transactions):
    not_end=False
    send_transactions=[]
    for item in transactions:
        if(re.findall(r'Amount</span></div></div>.* opacity="1">(-[0-9]+.[0-9]+) ETH',str(item))!=[]):
            send_transactions.append(item)
            not_end=True
    return not_end,send_transactions


# In[513]:


def track_hash(target):
    targeturl="https://www.blockchain.com/eth/address/"+target+"?view=standard"
    request=requests.get(targeturl)
    soup= BeautifulSoup(request.text,"html.parser")
    divs=soup.find_all("div", class_ = "enzKJw")
    transactions=soup.find_all("div", class_ = "koYsLf",direction="vertical")
    return divs, transactions


# In[514]:


def find_oldest_tran(send_transactions):
    old_date=99202006061458
            
    #for item in send_transactions:
    for index, item in enumerate(send_transactions):
        temp_date=re.findall(r'Date</span></div></div>.*opacity="1">([0-9]+-[0-9]+-[0-9]+ [0-9]+:[0-9]+)',str(item))
        date=int(re.sub(r'-| |:', "",temp_date[0]))
        if(date<old_date):
            old_date=date
            old_index=index
    last_hash=re.findall(r'<g id="arrow_x5F_full_x5F_right">.*opacity="1">(0x[a-z_0-9]+)</a>',str(send_transactions[old_index]))[0]
    last_date=re.findall(r'Date</span></div></div>.*opacity="1">([0-9]+-[0-9]+-[0-9]+ [0-9]+:[0-9]+)',str(send_transactions[old_index]))[0]
    last_amount=re.findall(r'(-[0-9]+\.[0-9]+) ETH</span></div></div></div></div></div>',str(send_transactions[old_index]))[0]
    return last_date,last_hash, last_amount


# In[515]:


def main_loop(line,f):
    not_end=True
    target=line
    count=0
    v_T_Date, v_To, v_amount="","",""
    hash_list=[]
    
    while(not_end and count<4):
        
        count=count+1
        divs,transactions=track_hash(target)
        not_end,send_transactions=tracing_ornot(transactions)
        
        v_Nonce,v_Transactions,v_Balance,v_T_Sent,v_T_Received,v_T_Fees=find_info(divs)
        
        if(not_end):
            v_T_Date, v_To, v_amount=find_oldest_tran(send_transactions)
            target=v_To
            if(count<4):
                hash_list.append(v_To)
        
        print_to(not_end,f,v_Nonce,v_Transactions,v_Balance,v_T_Sent,v_T_Received,v_T_Fees,v_T_Date,v_To,v_amount)
    f.write(line)
    for item in enumerate(hash_list):
        f.write(Arrow+item[1])
    f.write("\n")
    f.write(Line+"\n")


# In[516]:


file = open("ex_input_hw1.txt", "r")
f = open(ID+"_hw1_output.txt", "w")
lines = file.readlines()
for line in lines:
    #print(line.strip())
    main_loop(line.strip(),f)
    
f.close()


# In[ ]:




