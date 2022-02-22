# Case Study
## nature dataset - dengyr
check_n_epoch=10.  
Lr0.001.  
Cost=v1.  
C1r=0.5.  
C2cutoff=0.3.  
Down = neighbors0_200_200.  
Ratio = 0.125.  
N=30.  
O=Adam.  

## altas - mouse gastrulation
to be add

# Simulation Case Study

## multipath
### simulation
ratio_list=[0.2,0.4,0.6,0.8]
path1_sample=1000 #upper
path2_sample=path1_sample/ratio_list

genn_amt=1000.  
beta1_list=np.random.uniform(low=9, high=11, size=(genn_amt,)) #upper side.  
beta2_list=np.random.uniform(low=30, high=50, size=(genn_amt,)) #downside.  
gamma1_list=np.random.uniform(low=30, high=50, size=(genn_amt,)).  
gamma2_list=np.random.uniform(low=5, high=25, size=(genn_amt,)).  
path1_pct=95.  
path2_pct=95.  
alpha1_list=np.random.uniform(low=20, high=35, size=(genn_amt,)).  
alpha2_list=np.random.uniform(low=45, high=55, size=(genn_amt,)).  

### celldancer
check_n_epoch = 5
ratio0.2
Lr0.001
traceR0
corrcoefR0
C2cf0.3
Downneighbors0_200_200
Ratio0.125
N30
OAdam

## wingpath
### simulation

ratio_list=[0.2,0.4,0.6,0.8]
path1_sample=1000 #upper
path2_sample=path1_sample/ratio_list

genn_amt=1000
alpha1_list=np.random.uniform(low=10, high=15, size=(genn_amt,))
alpha2_list=np.random.uniform(low=30, high=40, size=(genn_amt,))
beta1_list=np.random.uniform(low=0.9, high=1.1, size=(genn_amt,)) 
beta2_list=np.random.uniform(low=0.9, high=1.1, size=(genn_amt,)) 
gamma1_list=np.random.uniform(low=0.125, high=0.125, size=(genn_amt,)) NOTICE - CURRENT VERSION: low=0.125, high=0.125, high should be 0.2727272727272727
gamma2_list=np.random.uniform(low=0.2727, high=0.2727, size=(genn_amt,)) NOTICE - CURRENT VERSION: low=0.2727, high=0.2727, high should be 0.4
path1_pct_list=np.random.uniform(low=90, high=95, size=(genn_amt,))
path2_pct_list=np.random.uniform(low=50, high=70, size=(genn_amt,))

### celldancer
epoch200
check_nNone
Lr0.001
C2cf0.3
Downneighbors0_200_200
Ratio0.125
N30
OAdam
traceR0.0
corrcoefR0.0

## backpath
### simulation
to be added
### celldancer
to be added
