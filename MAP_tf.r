# Just copy and paste everything to an R console.
# You may come across a numerical problem of NaN. 
# A feasible solution is to use a smaller learning rate. 
# We will fix the problem soon.
library(tensorflow)
load("synthetic2.RData")
set.seed(1)
X=Xd
y=yd
s_t=s_td
stepsize_r=0.02
stepsize_bet=0.02
ns=600
burnin=350
LL=20
K=10
y[s_t>3.5]=0 # y=0 if right censored
Is=ifelse(y==0,0,1)
Is_m=matrix(rep(Is,LL),nrow=LL,byrow=T)
r_random=T
SGD=F # stochastic gradient descent
if(SGD) batch_size=500  else batch_size=length(y)
sb=1
TT=2
mult=matrix(0,nrow=length(y),ncol=TT)
for(j in 1:TT)  mult[y==j,j]=1
mult[apply(mult,1,sum)==0,]=1

### MAP estimation
if(T)
{
    TT=max(y[y<99])
    n=length(y)
    p=ncol(X)
    # set prior for beta, N(b, Sig=diag(alpha))
    a0=1; b0=0.01
    # set prior for c0_t, Gam(e0_t, 1/f0_t). For now, just set e0_t=0.01; f0_t=0.01 for all t
    e0=0.01; f0=0.01
    c0=1; gam0=1
    r_val=0.3
    r=matrix(r_val, nrow=TT, ncol=K) 
    rr=array( r_val, c(TT, K, ns-burnin) )  
    Ik=matrix(1,TT,K) 
    #bet=array(0,c(p, TT, K) )     # coefficient beta_tk, no lab_seq
    bet=array(rnorm(p*TT*K,sd=1),c(p,TT,K))
    bet=array(0,c(p,TT,K))
    betbet=array(0, c(p,TT,K,ns-burnin))
    er_cum=numeric(ns-burnin)
    er_uncum=er_cum
}
eta=array(1,c(LL,batch_size,TT,K))
for(j in 1:TT) for(k in 1:K) eta[,,j,k]=matrix(rgamma(LL*batch_size,shape=r[j,k]),LL)
eta_tf=tf$placeholder(tf$float64, shape(LL,NULL,TT,K)) 
#r_tf=tf$placeholder(tf$float64, shape(LL,1,TT,K))
if(r_random) til_r_tf=tf$Variable(log(r), dtype=tf$float64) else til_r_tf=tf$constant(log(r), dtype=tf$float64)
bet_tf=tf$Variable(array(bet ,c(p,TT,K)), dtype=tf$float64)
if(K>1)Ik_tf=tf$placeholder(tf$float64, shape(TT,K))  else Ik_tf=tf$constant(Ik,tf$float64)
Is_tf=tf$placeholder(tf$float64, shape(NULL))
Is_m_tf=tf$placeholder(tf$float64, shape(LL,NULL))
st_tf=tf$placeholder(tf$float64, shape(NULL)) #added
x_tf=tf$placeholder(tf$float64, shape(NULL, p))
mult_tf=tf$placeholder(tf$float64, shape(NULL,  TT))
#Xbet_tf=tf$einsum('ij,jkl->ikl',x_tf,bet_tf) # dim=n*TT*K
#Xbet_tf_max0=Xbet_tf-tf$reduce_max(Xbet_tf)

Xbet_tf=tf$einsum('ij,jkl->kli',x_tf,bet_tf)# dim=TT*K*n
Xbet_tf_max1=Xbet_tf-tf$reduce_max(Xbet_tf,c(0L,1L))# dim=TT*K*n
Xbet_tf_max0=tf$transpose(Xbet_tf_max1,c(2L,0L,1L))#dim=n*TT*K
temp=tf$exp(Xbet_tf_max0)*eta_tf*Ik_tf # dim=LL*n*TT*K
numer=tf$reduce_sum(temp, 3L) # LL*n*TT
denom0=tf$reduce_sum(numer, 2L) # LL*n
denom=tf$expand_dims(denom0,-1L) # LL*n*1
pijl=numer/denom # dim=LL*n*TT
piyil=tf$reduce_sum(pijl*mult_tf,2L) # dim=LL*n

Xbet_max=tf$reduce_max(Xbet_tf)
# eXbet_max=tf$exp(Xbet_max)
Xbet_tf_minus=tf$transpose(Xbet_tf,c(2L,0L,1L))-Xbet_max #dim=n*TT*K
#eXbet_tf_minus=tf$exp(tf$transpose(Xbet_tf,c(2L,0L,1L)))-tf$exp(Xbet_max) #dim=n*TT*K

temp1=tf$exp(tf$transpose(Xbet_tf,c(2L,0L,1L)))*eta_tf*Ik_tf # dim=LL*n*TT*K
p1=tf$reduce_sum(temp1, c(2L,3L)) # for uncensored dim=LL*n
p0=tf$exp(-st_tf* tf$reduce_sum(tf$exp(tf$transpose(Xbet_tf,c(2L,0L,1L)))*eta_tf*Ik_tf,c(2L,3L)))# for both uncensored and censored dim=LL*n
pli=p0*p1*(tf$constant(1.0, dtype=tf$float64)-Is_tf)

loss_bet=-tf$reduce_mean( tf$log(tf$reduce_mean(piyil,0L))*Is_tf)- tf$reduce_mean(tf$log(tf$reduce_mean(p0,0L))) - tf$reduce_mean(tf$log(tf$reduce_mean(p1,0L))*Is_tf)+ tf$multiply( tf$constant( (a0+0.5)/n, dtype=tf$float64), tf$reduce_sum(tf$log1p(tf$square(bet_tf)*(0.5/a0) )) )

if(r_random)
{
    loss_r_numer=tf$reduce_mean(tf$exp(tf$log(piyil)*Is_m+tf$log(p1)*Is_m+tf$log(p0))*tf$reduce_sum(-tf$lgamma(tf$exp(til_r_tf))+(tf$exp(til_r_tf))*tf$log(eta_tf),c(2L,3L) ), 0L)
    loss_r_denom=tf$reduce_mean(tf$exp(tf$log(piyil)*Is_m+tf$log(p1)*Is_m+tf$log(p0)),0L)

    #loss_r_til=-(1/n)*tf$reduce_sum((gam0/K-1)*til_r_tf-tf$exp(til_r_tf)*c0)-tf$reduce_mean(loss_r_numer/loss_r_denom)
    loss_r_til=-tf$constant(1/n,dtype=tf$float64)*tf$reduce_sum(tf$multiply(tf$constant(gam0/K-1, dtype=tf$float64),til_r_tf)-tf$exp(til_r_tf)*tf$constant(c0,dtype=tf$float64))-tf$reduce_mean(loss_r_numer/loss_r_denom)
}

train_step_bet <- tf$train$AdamOptimizer(learning_rate=stepsize_bet, epsilon=1e-5)$minimize(loss_bet, var_list=list(bet_tf) )
if(r_random)    
{
    train_step_til_r=tf$train$AdamOptimizer(learning_rate=stepsize_r, epsilon=1e-5)$minimize(loss_r_til, var_list=list(til_r_tf) )
}

sess <- tf$Session()
sess$run(tf$global_variables_initializer())
nbatch=ceiling(n/batch_size)
for(s in 1:ns)
{
    #for(sb in 1:10) 
    {
        if(SGD)
        {
            if(sb%%nbatch==1 || nbatch==1)
            {
                mb=mini_batch(n, batch_size)
            }
            sbsb=ifelse(sb%%nbatch==0, nbatch, sb%%nbatch)
            X_batch=X[mb[[sbsb]],]
            y_batch=y[mb[[sbsb]]]
            mult_batch=mult[mb[[sbsb]],]
            st_batch=s_t[mb[[sbsb]]]
            Is_batch=Is[mb[[sbsb]]]
        }else
        {
            X_batch=X 
            y_batch=y
            mult_batch=mult 
            st_batch=s_t 
            Is_batch=Is 
        }
        
        # MAP solution for bet
        if(T)
        {
            for(j in 1:TT) for(k in 1:K) eta[,,j,k]=matrix(rgamma(LL*batch_size,shape=r[j,k]),LL)
            eta[eta<10^-200]=10^-200
            #for(iii in 1:5)
            {
                if(K>1) fd=dict(x_tf=X_batch, eta_tf=eta, mult_tf=mult_batch, Ik_tf=Ik, Is_tf=Is_batch, st_tf=st_batch, Is_m_tf=Is_m) else fd=dict(x_tf=X_batch, eta_tf=eta, mult_tf=mult_batch, Is_tf=Is_batch, st_tf=st_batch, Is_m_tf=Is_m)



                temp2=sess$run(list(train_step_bet,loss_bet), feed_dict=fd) 
                if(r_random)
                {
                    temp3=sess$run(list(train_step_til_r,loss_r_til), feed_dict=fd) 
                }
                    
            }
                
            #sess$run(clip_op)
            temp4=sess$run(bet_tf)
            if(sum(is.na(temp4))==p*TT*K)
            {
                cat("bet all NA") # there can be numerical problem 
                break
            }
            temp4[is.na(temp4)]=0.00001#bet[is.na(temp)]
            bet=temp4
            betbet[,,,s-burnin]=bet
            if(r_random)
            {
                temp4=exp(sess$run(til_r_tf))
                if(sum(is.na(temp4))==TT*K)
                {
                    cat("r all NA")# there can be numerical problem
                    break
                }
                Ik[is.na(temp4)]=0
                #temp4[is.na(temp4)]=r[is.na(temp)]
                temp4[is.na(temp4)]=0.00001
                r=temp4
                rr[,,s-burnin]=r
            }
        }
    }
}
