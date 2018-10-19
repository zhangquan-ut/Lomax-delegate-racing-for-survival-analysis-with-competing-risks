# 11/25/2017
# This is the latest inference of Lomax survival analysis using the same augmented q_i as in classification. Note that q_i=t_i and no need to sample it if t_i is observed as the event time.
##########################################################################
library(survival)
library(cmprsk)
library(threg)
library(pec)
library(riskRegression)
library(coda)
library(Rcpp)
library(RcppArmadillo)
library(RNG)
library(doParallel)
library(foreach)
library(foreign)
library(nnet)
library(ggplot2)
library(reshape2)
library(MASS)
library(fields)
library(pracma)
#library(BayesLogit)
#options(warn=2)
#par(mfrow=c(3,4))
#sourceCpp("~/Dropbox/fall_2016/GIG/q_sampler/cal_dlt.cpp")
#sourceCpp("~/Dropbox/fall_2016/GIG/q_sampler/c_cdf_q_nonint.cpp")

################## find all non-negative integer solutions to x_1+...+x_TT=n
nnintsol=function(TT, n) # return a matrix with dim= num(solution)*TT
{
    if(TT==1) return(n) else
    {
        tt=TT-1
        A=numeric()
        for(k in 0:n)
        {
            B=nnintsol(tt,n-k)
            
            A=rbind(A, cbind(B, repmat(k, size(B)[1],1 )))
        }    
        return(A)
    }
}

###########################################

#calculate q_itk
log_qq_1_sampler=function(r, xbeta,expert)
{
    nn=dim(xbeta)[1]
    log_new_theta=array(-Inf,dim=c(nn,TT,K,Tex))
    for(j in 1:TT)
    for(k in expert[[j]])
    {
        if(T)
        {
            temp=rgamma(nn, shape=r[j,k],rate=1)
            log_new_theta[temp==0,j,k,Tex]=-Inf
            log_new_theta[temp>0,j,k,Tex]=xbeta[temp>0,j,k,Tex+1]+log(temp[temp>0])
        }
        #new_theta[,j,k,Tex]=expxbeta[,j,k,Tex+1]*rgamma(nn, shape=r[j,k],rate=1)
        if( sum(is.na(log_new_theta[,j,k,Tex]))>0) 
        {
            
            cat(j,k, sum(is.na(log_new_theta[,j,k,Tex])),"nanana")
            stop("!")
        }
    }
    if(max(log_new_theta[,,,1])==-Inf) stop("all lambda are 0")
    log_theta_max=apply(log_new_theta[,,,1],1,max)
    temp2=log_new_theta[,,,1]-log_theta_max
    temp=log_theta_max+log1p(apply(exp(temp2),1,sum)-1)
    log_qq_1=log(rgamma(nn, shape=1,rate=1))-temp
    return(list(log_qq_1=log_qq_1, log_new_theta=log_new_theta))#, expxbeta=expxbeta))
}
qq_1_sampler=function(r, xbeta)
{
    #expxbeta=exp(xbeta-max(xbeta)) # dim=n*TT*K*(Tex+1)
    expxbeta=exp(xbeta)
    nn=dim(xbeta)[1]
    new_theta=array(0,dim=c(nn,TT,K,Tex))
    # sample new_theta^(T)
    for(j in 1:TT)
    	for(k in 1:K)
    	{
            if(T)
            {
                temp=rgamma(nn, shape=r[j,k],rate=1)
                new_theta[temp==0,j,k,Tex]=0
                new_theta[temp>0,j,k,Tex]=expxbeta[temp>0,j,k,Tex+1]*temp[temp>0]
            }
            #new_theta[,j,k,Tex]=expxbeta[,j,k,Tex+1]*rgamma(nn, shape=r[j,k],rate=1)
    		if( sum(is.na(new_theta[,j,k,Tex]))>0) 
    		{
    			
    			cat(j,k, sum(is.na(new_theta[,j,k,Tex])),"nanana")
    			stop("!")
    		}
    	}
    #new_theta[,,,Tex]=expxbeta[,,,Tex+1]*array(rgamma(nn*TT*K, shape=rep(c(r), each=nn), rate=1) , dim=c(nn,TT,K))
    if(Tex>1)
    for(tex in (Tex-1):1)
    {
    	for(j in 1:TT)
    		for(k in 1:K)
    		{
    			new_theta[,j,k,tex]=expxbeta[,j,k,tex+1]*rgamma(nn, shape=r[j,k],rate=1)
    		}
        #new_theta[,,,tex]=expxbeta[,,,tex+1]*array(rgamma(nn*TT*K, shape=c(new_theta[,,,tex+1]), rate=1) , dim=c(nn,TT,K))
    }
    temp=apply(new_theta[,,,1],1,sum) 
    if(sum(temp==0)>0) stop("sum lambda=0")
    qq_1=rgamma(nn, shape=1,rate=1)/temp
    return(list(qq_1=qq_1, new_theta=new_theta, expxbeta=expxbeta))
}



q_recur=function(qq_1, xbeta, expert) # qq_1 is the augmented variable~Gam(n_i, \sum \theta^1)
{
    qq=array(1, dim(xbeta)) # dim=n*TT*K*(Tex+1)
    for(j in 1:TT) for(k in expert[[j]])
    	qq[,j,k,1]=qq_1
    #qq[,,,1]=qq_1
    for(tex in 2:(Tex+1))
    {
        qq[,,,tex]=log1pexp(xbeta[,,,tex]+log(qq[,,,tex-1]))
    }
    return(qq)    
}
q_recur_old=function(xbeta)
{
    qq=array(1, dim(xbeta))
    for(tex in 2:(Tex+1))
    {
        qq[,,,tex]=log1pexp(xbeta[,,,tex]+log(qq[,,,tex-1]))
    }
    return(qq)    
}

###########################################
#############################################

pred_prob_grid=function(r, xbeta, p_g_cond, p_g_marg, cif.t, expert) # need to change for different models
{
    ng=nrow(p_g_cond)
    temp2=log_qq_1_sampler(r, xbeta, expert)
    new_theta=exp(temp2$log_new_theta-apply(temp2$log_new_theta,1,max))
    temp3=apply(new_theta,c(1,2),sum)
    p_g_cond_uncum=temp3/apply(temp3,1,sum)
    p_g_cond_cum=p_g_cond+p_g_cond_uncum

    { # p_g_marg in this part is only for softplus, not for lomax
        temp4=array(-Inf,c(ng, TT, K))
        for(j in 1:TT)
        for(k in expert[[j]])
        {
            temp4[,j,k]=log(r[j,k])+xbeta[,j,k,Tex+1]-log1pexp(temp2$log_qq_1+xbeta[,j,k,Tex+1])
        }
        temp4=exp(temp4-apply(temp4,1,max))
        temp3=apply( temp4 ,c(1,2),sum) # dim=nn*TT
        p_g_marg_uncum=temp3/apply(temp3,1,sum)
        p_g_marg_cum=p_g_marg+p_g_marg_uncum  
    }
    
    temp4=apply(exp(temp2$log_new_theta),1,sum)
    for(et in 1:length(eval_time))
    {
        temp5=1-exp(-eval_time[et]*temp4)
        for(j in 1:TT)
            cif.t[,j,et]=cif.t[,j,et]+temp5*p_g_cond_uncum[,j]
    }

    return(list(p_g_marg_cum= p_g_marg_cum, p_g_marg_uncum=p_g_marg_uncum, p_g_cond_cum= p_g_cond_cum, p_g_cond_uncum=p_g_cond_uncum, cif.t=cif.t
        #, X=X,X.t=xgrid
        ) )   
}

   
    
contour_plot=function(r, x, bet, p_g_cond, p_g_marg,cif.t, cum=T, cntur=F, plot=T, s, burnin=1000, expert, ...)
{    
    if(s <= burnin) return(0)
    pointcolor=c("blue","black","grey", "orange", "pink")
    #calculate predictive prob for each grid point
    ng=nrow(x)
    xgridbet=array(0,c(ng, TT, K, Tex+1))
    for(k in 1:K)
        for(j in 1:TT)
            xgridbet[,j,k,]=x%*%bet[,j,k,]  # construct Xbet with dim=n*(TT-1)*K*(Tex+1)
    #pred_qTex_p1=array(0,c(ng,tt,K))  
    #qq_1=qq_1_sampler(r,xgridbet)$qq_1      
    #pred_qTex_p1[,1:tt,]=q_recur(qq_1, xgridbet)[,label[1:tt],,Tex+1]
    pppp=pred_prob_grid(r, xbeta=xgridbet, p_g_cond=p_g_cond, p_g_marg=p_g_marg, cif.t=cif.t, expert)
    if(cum)
    {
    	p_g_cond=pppp$p_g_cond_cum 
    	p_g_marg=pppp$p_g_marg_cum
    }else 
    {
    	p_g_cond=pppp$p_g_cond_uncum 
    	p_g_marg=pppp$p_g_marg_uncum
    }
    ## contour plot
    return(pppp)        
}


###########################################################################################################
# main function
###########################################################################################################

SDS_MultReg=function(mult)
#function(mult, y, X, ns, TT=ncol(mult),  pred_plot=F, prune=F, Kchange=F )  # K is the truncated number of polytopes that includes each category; label is the category sequence in stick-breaking; tt is the first tt labels to use in stick-breaking, in other words, (tt+1) is the number of broken sticks 
{
    tt=TT-1
    n=length(y)
    #if(aug) X=cbind(X,log1pexp(X%*%new_bet))
    p=ncol(X)
    # set prior for beta, N(b, Sig=diag(alpha))
    mub=rep(0,p)
    # set values for hyper parameters, see notes "stick breaking, sum-softplus multinomial regression"
    # set prior for gam0_t, Gam(a0_t, 1/b0_t). For now, just set a0_t and b0_t all equal to 0.01
    a0=0.01; b0=0.01
    # set prior for c0_t, Gam(e0_t, 1/f0_t). For now, just set e0_t=0.01; f0_t=0.01 for all t
    e0=0.01; f0=0.01
    # gibbs sampler variables

    #bb=array(0, c(n,tt,ns))
    #m=matrix(1, nrow=n, ncol=TT)  # latent poisson variable m_it for b_it, no lab_seq
    #mm=array(0, c(n,tt,ns) )
    sub_m=array(1, c(n, TT, K, Tex+1) )  # m_itk, no lab_seq
    #sub_mm=array(1, c(n,tt,K, Tex+1,ns) )   
    r=matrix(1, nrow=TT, ncol=K)  # r_tk, no lab_seq
    rr=array( 0, c(TT, K, ns-burnin) )   
    c0=rep(2,TT)                  # r_tk~gamma(gam0_t/K, rate=c0_t), no lab_seq
    #cc0=matrix(0,nrow=tt, ncol=ns)
    gam0=rep(2, TT)               # r_tk~gamma(gam0_t/K, rate=c0_t), no lab_seq
    #gamgam0=matrix(0, nrow=tt, ncol=ns)
    theta=array(1,c(n,TT,K,Tex+1))      # latent poisson parameter theta_itk for m_itk, no lab_seq
    for(j in 1:TT) for(k in 1:K) theta[,j,k,Tex+1]=r[j,k]
    tau=theta
    #thetatheta=array(1,c(n,TT,K,ns-burnin))
    bet=array(0,c(p, TT, K, Tex+1) )     # coefficient beta_tk, no lab_seq
    Bet=array(0,c(p, TT, K) )
    #betbet=array(0, c(p,TT,K,ns-burnin))
    w=array(2,c(n, TT,K,Tex+1) )                   # w_itk, Polya gamma data augmentation, no lab_seq
    #ww=array(0,c(n,tt,K,ns))
    alpha=array(0.01, c(p,TT,Tex+1))
    if(no_share) alpha=array(0.01, c(p, TT, K, Tex+1))#precision of beta, no lab_seq
    if(K_share) alpha=array(0.01, c(p,TT,Tex+1)) # alpha are same for all experts in each layer
    if(Tex_share) alpha=array(0.01, c(p,TT,K)) # alpha are same for all layers in each expert
    if(category_share) alpha=matrix(0, p,TT) # alpha are same for all experts and layers in each category
    if(all_share) alpha=numeric(p)
    Xbet=array(0,c(n,TT,K,Tex+1))   # no lab_seq
    for(k in 1:K)
        for(j in 1:TT)
            Xbet[,j,k,]=X%*%bet[,j,k,]  # construct Xbet with dim=n*(TT-1)*K*(Tex+1)

    expert=list()
    for(j in 1:TT) 
        expert[[j]]=1:K # no lab_seq
    #expert[[1]]=1
    #expert[[2]]=1

    qq_1=s_t
    qitk=q_recur(qq_1, Xbet, expert)  # dim=n*TT*K*(Tex+1)
    logq=log(qitk)      # need update in each iter
    #qq_1=rgamma(n, shape=apply(mult,1,sum),rate=1)/apply(theta[,,,1],1,sum)
    qitk[,,,1]=qq_1
    logq[,,,1]=log(qq_1)
    qitk[,,,2]=log1pexp(Xbet[,,,2]+logq[,,,1])
    logq[,,,2]=log(qitk[,,,2])

    
    ldottk=matrix(0,TT,K)
    ldottk[1:TT, 1:K]=apply(array(sub_m[,,,Tex+1], c(n, TT, K)),c(2,3),sum) #need update in each iter
    til_l=matrix(1,nrow=TT,ncol=K)      # til_l~CRT(l_itk, gam0_t), no lab_seq
    #tiltil_l=array(0,c(tt,K,ns))
    til_p=matrix(0.5, nrow=TT,ncol=K)
    
    xi=numeric(n)

    
    p_s_marg=matrix(0,nrow=n,ncol=TT)
    p_s_cond=p_s_marg

    #p_s_list=array(0,c(n,TT,ns-burnin))
    loglh_marg=numeric(ns)
    loglh_cond=loglh_marg
    loglh_old_marg=-Inf
    loglh_old_cond=loglh_old_marg
    length_expert=sapply(expert,length)
    
    idx_censor=(yy==99)
    n_censor=sum(idx_censor)

    #for plot
    if(pred_test)
    {
        xgrid=X.t
        #if(aug) xgrid=cbind(xgrid,log1pexp(xgrid%*%new_bet))
        ng=nrow(xgrid)
        xgridbet=array(0,c(ng,TT,K,Tex+1))
        p_g_cond=matrix(0, nrow=ng, ncol=TT)
        #p_g_cond_iter=array(0, c(ng, TT, ns-burnin) )
        p_g_marg=p_g_cond
        p_g_cond_oneiter=p_g_cond
        p_g_marg_oneiter=p_g_cond_oneiter            
        #p_g_list=array(0,c(ng,TT,ns-burnin ) )
        #pred_list_cond=matrix(0, ng, ns-burnin)
        #pred_list_marg=pred_list_cond
    }
    cif=array(0,c(n, TT, length(eval_time)))
    cif.t=array(0,c(ng, TT, length(eval_time)))
    timer=Sys.time()
    ##### start gibbs sampler    
    for (s in 1:ns)
    {
        if(s%%500==1)
        { 
            cat(s," ")
            #if(s%%50==1)  cat("\n")
        }    
        
        # sample missing y[i]
        for(i in 1:n)
        {
            # first determine the cause of censored obs
            if(yy[i]>=99) #100:uncensored data, but cause is unknown; 99: censored data
            {
                tempp=theta[i,,,1] 
                if(sum(tempp)==0)
                    tempp=matrix(1,TT,K)
                if(K>1) tempp=apply(tempp,1,sum) else tempp=c(tempp)
                idx=c(rmultinom(1,size=1,prob=tempp))
                y[i]=which.max(idx)
            }
        }
        mult=matrix(0,nrow=n, ncol=TT)
        for(j in 1:TT)  mult[y==j,j]=1

        # sample m_itk^{tex=1}
        for(i in 1:n)
        {
        	for(j in 1:TT)
        	{
        		tempp=theta[i,j,,1] 
        		if(sum(tempp)==0)
        		    tempp=rep(1,length(tempp))
        	    sub_m[i,j,,1]=c( rmultinom(1, size=mult[i,j], prob=tempp ) )
        	}
        }
        #sub_mm[,,,1,s]=sub_m[,,,1] 
        # prune: has not changed 
        if(prune & s<=burnin )   # has not changed yet
        if(s%%30==0)
        {
            for(j in 1:TT)
            {
                if(length_expert[j]>1)
                {
                    # prune as many as it can at one checkpoint: prune k if sum_i{m_itk}=0 and r[tk]->0
                    expert[[j]]=expert[[j]][(apply(sub_m[,j,expert[[j]],1 ], 2, sum)>0)]#+ (r[j,expert[[j]] ]>10^-10)>0 ]
                    length_expert[j]=length(expert[[j]])
                    if(length(expert[[j]])==0)
                    {
                        expert[[j]]=1
                        length(expert[[j]])=1
                    }
                    # prune one expert for each checkpoint
                    #if( sum(sub_m[,j,length_expert[j] ])==0 )
                    #{
                    #    expert[[j]]=expert[[j]][1:(length_expert[j]-1)]
                    #    length_expert[j]=length_expert[j]-1
                    #}    
                }
            }  
            #for(j in 1:tt)  cat(expert[[j]],"\n")  
        } 
        
        ## downward sampling for theta
        for(j in 1:TT)
        {
            for(k in expert[[j]])
            {
                theta[,j,k,Tex+1]=r[j,k]
                for(tex in Tex:1)
                {
                    tau[,j,k,tex]=rgamma(n, shape=theta[,j,k,tex+1]+sub_m[,j,k,tex],scale=1-exp(-qitk[,j,k,tex+1]))
                    theta[,j,k,tex]=tau[,j,k,tex]/sapply(qitk[,j,k,tex], max, 10^-10)
                    #idx=qitk[,j,k,tex]==0
                    #theta[idx,j,k,tex]=tau[idx,j,k,tex]*(10^10) # numerical
                    #theta[!idx,j,k,tex]=tau[!idx,j,k,tex]/qitk[!idx,j,k,tex]
                    #if(sum(theta[,j,k,tex]==0)>0) stop("theta=0")
                }
                #theta[,,,1:Tex]=tau[,,,1:Tex]/apply(qitk[,,,1:Tex], c(1,2,3,4), max, 10^(-10))  # numerical 
            }
            theta[,j,-expert[[j]],]=0
        }
        #if(s>burnin) thetatheta[,,,s-burnin]=theta[,,,1]
        # sample qq_1 if t_i is missing
        qq_1[idx_censor]=rexp(n_censor, apply(theta[idx_censor,,,1],1,sum))+s_t[idx_censor]
        qitk[,,,1]=qq_1
        logq[,,,1]=log(qq_1)

        if(F)
        {
            qq_1=rgamma(n, shape=apply(mult,1,sum),rate=1)/apply(theta[,,,1],1,sum)
            qitk[,,,1]=qq_1
            logq[,,,1]=log(qq_1)
        }
        
        #qitk=q_recur(qq_1, Xbet)
        #logq=log(qitk)

        # upward sampling
        #cat("upward for sub_m,w,bet","\n")
        for(j in 1:TT)
        {
        for(k in expert[[j]])
        for(tex in 2:(Tex+1))
        {
            #if(j==1 & k==1) next
            # sample m_itk^{tex=2: Tex+1} 
            sub_m[,j,k,tex]=rCRT(n, sub_m[,j,k,tex-1], theta[,j,k,tex])
            if(T)
            {
            # sample w_itk^{tex=2: Tex+1}  
            idx=qitk[,j,k,tex-1]>0
            w[!idx,j,k,tex]=0
           #cat(paste("num=",sum(idx),"\n", sep=""))
            w[idx,j,k,tex]=rPG(sum(idx), h=sub_m[idx,j,k,tex-1]+theta[idx,j,k,tex], z=Xbet[idx,j,k,tex]+logq[idx,j,k,tex-1], trunc=10)
            w[w[,j,k,tex]<0,j,k,tex]=0
            # sample beta_tk^{tex} 
            #if(!(j==1 & k==1 & Tex==2))
            
            xi[!idx]=sub_m[!idx,j,k,tex-1]
            xi[idx]=-w[idx,j,k,tex]*logq[idx,j,k,tex-1]+0.5*(sub_m[idx,j,k,tex-1]-theta[idx,j,k,tex])
            #omega=diag(w[,j,k,tex])
            Prec=diag(alpha[,j,tex])
            if(no_share) Prec=diag(alpha[,j,k,tex])
            if(K_share) Prec=diag(alpha[,j,tex])
            if(Tex_share) Prec=diag(alpha[,j,k])
            if(category_share) Prec=diag(alpha[,j])
            if(all_share) Prec=diag(alpha)
            if(T)
            {
                vvv=Prec+crossprod(sqrt(w[,j,k,tex]) * X)  #=Prec+ t(X)%*%omega%*%X
                mindiag= min(diag(vvv))
                if(mindiag<10^-100)
                {
                    min_idx=which.min(diag(vvv))
                    vvv[min_idx,min_idx]=10^-100
                }
                dg1=ifelse( mindiag>1, mindiag*0.0001 , 10^-4 )
                bet[,j,k,tex]=c_mvrnorm(n=1, mu=apply(X*xi,2,sum),
                    #c( t(X)%*%xi+Prec%*%mub), 
                    sigma=vvv, eps=dg1, add=T, precision=T, cov_mu=T)$x
                if(F)
                {
                incrm=0
                v=try(solve(vvv),silent=T) 
                temp_b=try(mvrnorm(n = 1, mu=c( v%*%(t(X)%*%xi+Prec%*%mub) ), Sigma=v), silent=T)
                while(class(temp_b)=="try-error")
                {
                    incrm=incrm+dg1
                    v=try(solve(vvv+diag(rep(incrm,p))),silent=T)   
                    if(class(v)=="try-error") next 
                    temp_b=try(mvrnorm(n = 1, mu=c( v%*%(t(X)%*%xi+Prec%*%mub) ), Sigma=v), silent=T)
                }
                bet[,j,k,tex]=temp_b
                }
                #if(F)
                #{
                #    v=solve(Prec+t(X)%*%omega%*%X+diag(rep(10^(-4),p)))
                #    bet[,j,k,tex]=mvrnorm(n = 1, mu=c( v%*%(t(X)%*%xi+Prec%*%mub) ), Sigma=v)
                #}
            }
            } 
            
            # calculate Xbet and qitk   
            Xbet[,j,k,]=X%*%bet[,j,k,]  # construct Xbet with dim=n*(TT-1)*K*(Tex+1)
           	qitk[,j,k,tex]=log1pexp(Xbet[,j,k,tex]+logq[,j,k,tex-1])
            logq[,j,k,tex]=log(qitk[,j,k,tex])
        }
            sub_m[,j,-expert[[j]],]=0
        }   
        #if(s>burnin) betbet[,,,s-burnin]=bet[,,,Tex+1]
        if(s>burnin &K>1) Bet=Bet+ bet[,,,Tex+1]
        # sample alpha, the prior precision of beta
        if(no_share) 
        {
            #cat("alpha","\n")
            for(j in 1:TT)
            for(k in expert[[j]])
            for(tex in 2:(Tex+1))
            {
                temp=f0+0.5*bet[,j,k,tex]^2 # only if mub=0
                alpha[,j,k,tex]=rgamma(p, shape=e0+0.5, rate=1 )/temp# this is 1/sig2
            }
        }
        if(K_share)
        {
            for(j in 1:TT)
            for(tex in 2:(Tex+1))
            {
                if(K>1) temp=f0+0.5*apply(bet[,j,,tex]^2,1,sum) else
                temp=f0+0.5*sum(bet[,j,,tex]^2)
                alpha[,j,tex]=rgamma(p, shape=e0+K/2, rate=1)/temp
            }
        }
        if(Tex_share)
        {
            for(j in 1:TT)
            for(k in expert[[j]])
            {
                if(Tex>1) temp=f0+0.5*apply(bet[,j,k,]^2,1,sum) else
                temp=f0+0.5*sum(bet[,j,k,]^2)
                alpha[,j,k]=rgamma(p, shape=e0+Tex/2, rate=1)/temp
            }
        }
        if(category_share)
        {
            for(j in 1:TT)
            {
                temp=f0+0.5*apply(bet[,j,,]^2,1,sum)
                alpha[,j]=rgamma(p, shape=e0+K*Tex/2, rate=1)/temp
            }
            
        }
        if(all_share)
        {
            alpha=rgamma(p, shape=e0+TT*K*Tex/2, rate=1)/(f0+0.5*apply(bet^2,1,sum))
        }
        
        if(r_random)
        {
            # sample c0
            #cat("c0","\n")
            for(j in 1:(TT))
            {
                c0[j]=rgamma(1, shape=gam0[j]+1, rate=1)/(sum(r[j,expert[[j]]])+1)
                if(c0[j]==0) c0[j]=10^-20
            }
            #cc0[,s]=c0
            
            
            # calculate til_p
            #cat("til_p","\n")
            for(j in 1:TT)
            {
                if(K==1)  til_p[j,]=1/(1+c0[j]/sum(qitk[,j,,Tex+1])  ) else
                til_p[j,]=1/(1+c0[j]/apply(qitk[,j,,Tex+1],2,sum)  )
            }
            # calculate ldottk  
            if(K>1) ldottk=apply(sub_m[,,,Tex+1],c(2,3),sum) else
            {
                ldottk=matrix(0,TT,1)
                ldottk[,1]=apply(sub_m[,,1,Tex+1],2,sum)
            }


            #ldottk=apply(array(sub_m[,,,Tex+1], c(n, TT, K)),c(2,3),sum) #need update in each iter
            
            
            # sample til_l
            #cat("til_l","\n")
            if(Kchange)
            {
              # r~Gam(gam0/K+l.k,...). Here we prune K and set K=length(expert) adaptively and non-increasing.
              for(j in 1:(TT))
              {
                temp=length(expert[[j]])
                for(k in expert[[j]])
                    til_l[j,k]=rCRT(1, ldottk[j,k], gam0[j]/temp)
              }      
            }else
            {
              # if an expert is pruned, than it will not come back, but K never changes
              for(j in 1:(TT))
              {
                for(k in expert[[j]])
                    til_l[j,k]=rCRT(1, ldottk[j,k], gam0[j]/K)
              }      
            }
            
            # sample gam0
            #cat("gam0","\n")
            temp=til_p
            #temp[temp>1-10^-15]=1-10^-15 #numerical
            #temp[temp<10^-15]=10^-15
            for(j in 1:TT)
            {
                temp1=sum(log1p(-temp[j,expert[[j]]])) # dim=(TT)*K
                gam0[j]=rgamma(1,shape=a0+sum(til_l[j,expert[[j]]]), rate=1)/(b0-temp1/length(expert[[j]]))
            }
            
            #gamgam0[,s]=gam0
            
            
            # sample r
            #cat("r","\n")
            for(j in 1:TT)
            {
              if(Kchange)     
              {
                # r~Gam(gam0/K+l.k,...). Here we prune K and set K=length(expert) adaptively and non-increasing.
                temp=length(expert[[j]])
                if(temp==1)
                {
                    r[j,expert[[j]]]=rgamma(temp,shape=gam0[j]/temp+ldottk[j,expert[[j]]],rate=1)/(c0[j]+max(10^-6,sum(qitk[,j,expert[[j]], Tex+1])))
                    r[j,r[j,]==Inf]=10^10
                    #r[j,expert[[j]]]=rgamma(temp,shape=gam0[j]/temp+ldottk[j,expert[[j]]],rate=1)/(c0[j]+sum(qitk[,j,expert[[j]], Tex+1]))
                }else
                {
                    r[j,expert[[j]]]=rgamma(temp, shape=gam0[j]/temp+ldottk[j,expert[[j]]],rate=1)/sapply((c0[j]+apply(qitk[,j,expert[[j]], Tex+1],2,sum)), max, 10^-6)
                    r[j,r[j,]==Inf]=10^10
                }
                #r[j,expert[[j]]]=rgamma(temp, shape=gam0[j]/temp+ldottk[j,expert[[j]]],rate=1)/(c0[j]+apply(qitk[,j,expert[[j]], Tex+1],2,sum))
              }
              if(!Kchange)
              {
                temp=length(expert[[j]])
                # if an expert is pruned, than it will not come back, but K never changes
                if(temp==1)
                {
                    #r[j,expert[[j]]]=rgamma(temp,shape=gam0[j]/K+ldottk[j,expert[[j]]],rate=1)/(c0[j]+max(10^-6,sum(qitk[,j,expert[[j]], Tex+1])))
                    r[j,expert[[j]]]=rgamma(temp,shape=gam0[j]/K+ldottk[j,expert[[j]]],rate=1)/(c0[j]+sum(qitk[,j,expert[[j]], Tex+1]))
                }else
                
                #r[j,expert[[j]]]=rgamma(temp, shape=gam0[j]/K+ldottk[j,expert[[j]]],rate=1)/sapply((c0[j]+apply(qitk[,j,expert[[j]], Tex+1],2,sum)), max, 10^-6)
                r[j,expert[[j]]]=rgamma(temp, shape=gam0[j]/K+ldottk[j,expert[[j]]],rate=1)/(c0[j]+apply(qitk[,j,expert[[j]], Tex+1],2,sum))
              }  
                r[j,-expert[[j]]]=0
            }
            for(j in 1:TT) for(k in 1:K) theta[,j,k,Tex+1]=r[j,k]
            if(s>burnin) rr[,,s-burnin]=r
        }
        

      
        if(pred_test & s>burnin)
        {   
            #cat("here")
            temp=contour_plot(r, xgrid,bet, p_g_cond=p_g_cond, p_g_marg=p_g_marg, cif.t=cif.t, cum=cum, cntur, plot=pred_plot,s, burnin=burnin, expert)
            #cat("here2")
            #p_g_list[,,s-burnin]=temp$p_g_uncum
            #pred_list[,s-burnin]=label[apply(temp$p_g_uncum, 1, which.max)]
            #pred_list_cond[,s-burnin]=apply(temp$p_g_cond_uncum, 1, which.max)
            #pred_list_marg[,s-burnin]=apply(temp$p_g_marg_uncum, 1, which.max)
            p_g_marg=temp$p_g_marg_cum
            p_g_cond=temp$p_g_cond_cum
            p_g_marg_oneiter=temp$p_g_marg_uncum
            p_g_cond_oneiter=temp$p_g_cond_uncum
            cif.t=temp$cif.t
            #p_g_cond_iter[,,s-burnin]=temp$p_g_cond_uncum 
        }
        # calculate likelihood
        #temp=contour_plot(r, X, bet, p_g_cond=p_s_cond, p_g_marg=p_s_marg, cum=F, cntur=F, plot=F, s, burnin=0, expert, loglh, px1, px2)
        temp3=apply(theta[,,,1],c(1,2),sum)
        p_s_cond=temp3/apply(temp3,1,sum)

        # pred prob by qq_1, r and xbeta
        temp3=array(0,c(n, TT, K))
        for(j in 1:TT) 
        {
        	for(k in expert[[j]])
        	{
       			temp3[,j,k]=r[j,k]/(qq_1+1/exp(Xbet[,j,k,2]))
        	} 
            temp3[,j,-expert[[j]]]=0
        }		
        temp3=apply( temp3 ,c(1,2),sum) # dim=nn*TT
        for(i in 1:n)
        {
        	if(sum(temp3[i,])==Inf)
        	{
        		temp3[i, temp3[i,]<Inf]=0
        		temp3[i, temp3[i,]>0]=1
        	}
        }
        p_s_marg=temp3/apply(temp3,1,sum)

        if(F)
        if(sum(is.na(p_s_cond))>0 || sum(is.na(p_s_marg))>0)
        {
        	stop("NA!")
        }
        #if(s==ns) p_tr_cum=temp$p_g_cum
        #if(s>burnin) p_s_list[,,s-burnin]=p_s_marg
        temp1=matrix(0,nrow=n,ncol=TT)
        for(j in 1:TT)
        {
            temp1[,j]=(y==j)  
        }
        loglh_cond[s]=sum(log(apply(p_s_cond*temp1,1,sum)))
        loglh_marg[s]=sum(log(apply(p_s_marg*temp1,1,sum)))
        if(loglh_old_marg<loglh_marg[s]) 
        {
            p_ml_marg=p_g_marg_oneiter
            #p_tr_uncum=p_s
        }
        if(loglh_old_cond<loglh_cond[s])
        {
        	p_ml_cond=p_g_cond_oneiter
        }
        loglh_old_marg=loglh_marg[s]
        loglh_old_cond=loglh_cond[s]
        ## main function done.
        if(F & s%%100==99) 
        {
        	cat(mean(apply(p_s_cond,1,which.max)!=y),"  ",
   				 mean(apply(p_s_marg,1,which.max)!=y),"  ", mean(apply(p_g_cond,1,which.max)!=y.t),"  ",
                 mean(apply(p_g_marg,1,which.max)!=y.t),"\n")	
        }
    }
    timer=Sys.time()-timer

    if(F)
    {
        size_bet=apply(betbet,c(1,2,3),effectiveSize)
        size_p=apply(p_s_list,c(1,2),effectiveSize)
    }
    if(F)
    {
    size_r=apply(rr,c(1,2),effectiveSize)
    size_bet=apply(betbet,c(1,2,3),effectiveSize)
    size_theta=apply(thetatheta,c(1,2,3),effectiveSize)
    }
    er_cond=mean(apply(p_g_cond,1,which.max)!=y.t) 
    #er_cond_ml=mean(apply(p_ml_cond,1,which.max)!=y.t) 
    er_marg=mean(apply(p_g_marg,1,which.max)!=y.t) 
    #er_marg_ml=mean(apply(p_ml_marg,1,which.max)!=y.t) 
    return(list(expert=expert
    #,pred_list=pred_list, 
        #p_s_cond=p_s_cond,
    	#p_g_cond =p_g_cond, p_g_marg=p_g_marg
    	 # ,loglh_cond =loglh_cond, loglh_marg=loglh_marg 
    	  
         #       ,p_ml_cond=p_ml_cond, p_ml_marg=p_ml_marg # p_ml is the within sample prob associated with the max loglh
                #,p_g_list=p_g_list
                #,p_tr_uncum=p_tr_uncum
                #,p_tr_cum=p_tr_cum
                #,r=rr
                #,size_r=size_r
          #      ,size_bet=size_bet
          #      ,size_p=size_p
                #,size_lam=size_theta
          #      ,timer=timer
                ,er_cond=er_cond
          #      ,er_cond_ml=er_cond_ml
                ,er_marg=er_marg
          #      ,er_marg_ml=er_marg_ml
                #,p_g_cond_iter=p_g_cond_iter
                ,bet=Bet/(ns-burnin)
          #      ,X=X
          #      ,X.t=X.t
                ,r=r
                ,rr=rr
                ,cif.t=cif.t/(ns-burnin)
                )
          )
}

#save(res,file="/home/zhang/Dropbox/fall_2016/GIG/data_analysis_2_methods/prune/usps_augment_r_randomTRUE_rep5.RData")
###########################################
if(F)
{
    j=1
    apply(sub_m[,j,,2], 2, sum)+gam0[j]/K
    c0[j]+apply(log1pexp(Xbet[,j,,2]*log(qitk[,j,,1])),2,sum)

	x1_range=c(-2.5,2.5); x2_range=c(-2.5,2.5)    # xor and square
    if(dat=="banana"){x1_range=c(-2.5,3); x2_range=c(-2.5,3) }
	if(dat=="dbmoon") {x1_range=c(-14,24); x2_range=c(-8,14)}
	if(dat=="dbswissroll") {x1_range=c(-11,14); x2_range=c(-13,10)}
	if(dat=="swissroll") {x1_range=c(-12,15); x2_range=c(-14,11)}
	if(dat=="3cosine")  {x1_range=c(-6,6); x2_range=c(-1.5,3.5)}
	TT=max(y[y<99])
	tt=TT-1
    mult=matrix(0,nrow=length(y),ncol=TT)
    for(j in 1:TT)  mult[y==j,j]=1
	ns=3000
	burnin=2000
	K=10
	Tex=1
	pred_test=T
	pred_plot=F
	cum=F
	cntur=F
    aug=F
	#x1_num=100; x2_num=100
    x1_num=50; x2_num=50

	prune=T
	Kchange=T
	M=50 # sample M lambda's from gamma(rt, e^xi'beta) in each gibbs iter for E(pit|lambda)
	
    no_share=F
    K_share=F
    Tex_share=F
    category_share=F
	all_share=F

     #no_share=T
    K_share=T

    r_random=F
    prune=F
    
	res=SDS_MultReg(mult)
	save(res, file="/home/zhang/Dropbox/fall_2016/GIG/stampede/augment_prune/usps.RData")
    mean(apply(p_s_cond,1,which.max)!=y) 
    mean(apply(p_s_marg,1,which.max)!=y) 

    mean(apply(res$p_g_cond,1,which.max)!=y.t) 
    mean(apply(res$p_g_marg,1,which.max)!=y.t) 
    
    median(c(c(res$size_bet),c(res$size_r),c(res$size_lam)))

    quantile(c(res$size_bet), c(0,0.5,1))
    quantile(c(res$size_r), c(0,0.5,1))
    quantile(c(res$size_lam), c(0,0.5,1))

}


## res=g_mult_nonpara(mult, X, ns, r_random, prior_vbet_known, pred_single_obs, M, X_pred, mult_pred, pred_plot)

if(F) # transform data using softplus
{
    er=numeric(10)
    er.t=numeric(10)
    for(ii in 1:10)
    {
        cat("iter=",ii,"\n")
        res=SDS_MultReg(mult)
        er[ii]=mean(apply(res$p_s_cond,1,which.max)!=y) 
        er.t[ii]=res$er_cond
        expert=res$expert
        bet=res$bet
        pp=0
        p=nrow(bet)
        new_bet=numeric()
        for(j in 1:TT) for(k in expert[[j]])
        {
            pp=pp+1
            new_bet=cbind(new_bet, bet[,j,k,2])
        }
        X=cbind(X,log1pexp(X%*%new_bet))
        X.t=cbind(X.t, log1pexp(X.t%*%new_bet))
    }
}


