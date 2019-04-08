import numpy as np
from powerlaw import draw_power_binary

def network_hub_line(N):
    '''Create a connectivity network for the N nodes, line plus connected to hub. Returns dictionary of connections'''
    # Create list of links
    listLinks=[]
    for i in range(N)[1:]:
        listLinks.append([0,i])
        if i<(N-1) and i>0:
            listLinks.append([i,i+1])
            #listLinks

            # Create dictionary with connections, each entry is a node
    gi={}
    for i in range(N):
        gi[i]=[]
    for l in listLinks:
        gi[l[0]].append(l[1])
        gi[l[1]].append(l[0])
    #gi
    return gi

def network_local(N,k,xmin=1):
    '''Create a connectivity network for the N nodes, k nearest neighbours'''
    # Create list of links
    listLinks=[]
    for i in np.arange(N)+xmin:
        for j in range(k):
            if (i+j+1) < N+xmin:
                listLinks.append([i,i+j+1])
            #listLinks
            # Create dictionary with connections, each entry is a node
    gi={}
    for i in np.arange(N)+xmin:
        gi[i]=[i]
    for l in listLinks:
        gi[l[0]].append(l[1])
        gi[l[1]].append(l[0])
    #gi
    return gi



def probabilities(gi,N):
    # Gets dictionary of links, returns dictionary of probabilities, each entry is the probability of a link from i to j
    gij={}
    for i in range(N):
        for j in gi[i]:
            gij[i,j]=1/len(gi[i])
    return gij

def p_power_law(N,alpha,xmin=1):
    arr_i = np.arange(N)+xmin
    arr_pi = arr_i**(-alpha)
    arr_pi /= float(np.sum(arr_pi))
    # pilist=[]
    # for i in range(N):
    #     pilist.append((i+xmin)**(-alpha))
    #     # Generate pi
    # pi=np.array(pilist)/sum(pilist)
    return dict(zip(arr_i,arr_pi))


def text(T,N,pi,gi,gij):
    # inital step
    x=np.random.randint(N)
    # Number of Markov steps
    # T=100000
    sample=[x]
    for t in range(T):
        xp=gi[x][np.random.randint(len(gi[x]))]
        a = (gij[xp,x]*pi[xp])/(gij[x,xp]*pi[x])
        if a>=1:
            x=xp
        else:
            if np.random.rand() < a:
                x=xp
        sample.append(x)
    return sample


def textmu(T,N,pi,gi,mu,alpha,xmin):
    # inital step
    listRandomX= draw_power_binary(int(1+T*1.1*mu),xmin,xmin+N-1,alpha) # pre-computed list of random numbers from power-law, drawn in advance for efficiency reasons
    x=listRandomX[0]
    listRandomX=listRandomX[1:]
    # Number of Markov steps
    # T=100000
    sample=[x]
    for t in range(T):
#        print(t,x,);
        # Proposal: generate next x
        if np.random.rand() > mu: # local
            xp=gi[x][np.random.randint(len(gi[x]))]
        else: # pick a random number from power law
            if(len(listRandomX)==0): ## if there are no more random variables;
                listRandomX= draw_power_binary(int(1+T*0.1*mu),xmin,xmin+N-1,alpha)
            xp=listRandomX[0]
            listRandomX=listRandomX[1:] 
        # Acceptance
        if xp not in gi[x]: # if xp is not a neighbour of x
            x = xp # acceptance=1
        else:
            gxptox=mu*pi[x]+(1-mu)/len(gi[xp])
            gxtoxp=mu*pi[xp]+(1-mu)/len(gi[x])
            a = (gxptox*pi[xp])/(gxtoxp*pi[x])
            if a>=1:
                x=xp
            else:
                if np.random.rand() < a:
                    x=xp
 #       print("\t",xp,a)
        sample.append(x)
    return sample

def text_to_counts(sample,N,xmin=1):
    '''Given a text, return counts'''
    estimates={}
    for i in np.arange(N)+xmin:
        fi=sample.count(i)
        estimates[i]=fi
    return estimates

def counts_hubLine(Ntypes,Ntokens,alpha):
    '''Return counts of words in the text hubLine'''
    pi = p_power_law(Ntypes,alpha)
    gi = network_hub_line(Ntypes)
    gij = probabilities(gi,Ntypes)
    sample = text(Ntokens,Ntypes,pi,gi,gij)
    counts = text_to_counts(sample,Ntypes)
    return counts

def counts_muProcess(Ntypes,Ntokens,alpha,mu,k,xmin=1):
    ''' Returns counts of words in the text generated from muProcess'''
    pi = p_power_law(Ntypes,alpha)
    gi = network_local(Ntypes,k)
    sample = textmu(Ntokens,Ntypes,pi,gi,mu,alpha,xmin)
    counts = text_to_counts(sample,Ntypes)
    return counts

def sequence_muProcess(Ntypes,Ntokens,alpha,mu,k,xmin=1):
    ''' Returns counts of words in the text generated from muProcess'''
    pi = p_power_law(Ntypes,alpha)
    gi = network_local(Ntypes,k)
    sample = textmu(Ntokens,Ntypes,pi,gi,mu,alpha,xmin)
    # counts = text_to_counts(sample,Ntypes)
    return sample


