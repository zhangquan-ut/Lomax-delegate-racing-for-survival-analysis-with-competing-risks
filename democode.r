### synthetic data 1
source("new_surv_gibbs.r")
load("synthetic1.RData")
ns=3000 # number of iterations
burnin=2000 # number of burn-in's
K=10 # Maximum number of sub-risks
Tex=1
pred_test=T
pred_plot=F
cum=T
cntur=F
Kchange=T
no_share=F
K_share=F
Tex_share=F
category_share=F
all_share=F
K_share=T
#category_share=T
r_random=T # Learn random weights of sub-risks
prune=F # if TRUE, prune unnecessary sub-risks by step 8 of Appendix B
##### training-testing partition
set.seed(1)
n=length(yd)
idx=which(yd<99) # y=99 indicating censoring
idx.testing=sample(idx, n*0.2)
X.t= Xd[idx.testing,]
X= Xd[-idx.testing,]
y.t=yd[idx.testing]
y=yd[-idx.testing]
s_t.t=s_td[idx.testing]
s_t=s_td[-idx.testing]
XX=data.frame(X)
XX.t=data.frame(X.t)
yy=y
yyy=y.t
TT=max(y[y<99])
tt=TT-1
mult=matrix(0,nrow=length(y),ncol=TT)
for(j in 1:TT)  mult[y==j,j]=1
res=SDS_MultReg(mult)
r=res$r
r
par(mfrow=c(2,1))
plot(1:K, sort(r[1,],decreasing=T), ylim=range(r), xlab="k", ylab=expression(r[jk]),main="synthetic data 1")
points(1:K, sort(r[2,],decreasing=T),pch=2)
legend("topright", pch=c(1,2), legend=c("risk 1","risk 2"))

##############################################################
##############################################################
## synthetic data 2
load("synthetic2.RData")
ns=4500 # number of iterations
burnin=4000 # number of burn-in's
##### training-testing partition
set.seed(1)
n=length(yd)
idx=which(yd<99) # y=99 indicating censoring
idx.testing=sample(idx, n*0.2)
X.t= Xd[idx.testing,]
X= Xd[-idx.testing,]
y.t=yd[idx.testing]
y=yd[-idx.testing]
s_t.t=s_td[idx.testing]
s_t=s_td[-idx.testing]
XX=data.frame(X)
XX.t=data.frame(X.t)
yy=y
yyy=y.t
TT=max(y[y<99])
tt=TT-1
mult=matrix(0,nrow=length(y),ncol=TT)
for(j in 1:TT)  mult[y==j,j]=1
res=SDS_MultReg(mult)
r=res$r
plot(1:K, sort(r[1,],decreasing=T), ylim=range(r), xlab="k", ylab=expression(r[jk]), main="synthetic data 2")
points(1:K, sort(r[2,],decreasing=T),pch=2)
legend("topright", pch=c(1,2), legend=c("risk 1","risk 2"))



