#! /usr/bin/env python3
import sys
import numpy as np
import networkx as nx
import os.path

#Usage:
# ./sampling-nets.py p N
# where p is the prbability of sampling a neighbour and N is the number of points in your network (use N=0 to sample all)

# Comand line input
try:
    p = float(sys.argv[1])
    N = int(sys.argv[2])
except:
    p =0.5
    N = 0
    print("Using p=0.5, and all nodes enter in the command line to change")

def shufflepair(node1,node2): #Shuffle the order of the nodes in a link before adding
    if np.random.randint(2)==1:
        return([node1,node2])
    else:
        return([node2,node1])
    

G=nx.read_adjlist("../data/networks/out.internet",nodetype=int)
print("Network uploaded successfully, number of links to be sampled:",G.number_of_edges())

# If all nodes to be sampled
if N ==0:
    N = 2*G.number_of_edges()+10

#Initialization:
link = list(G.edges())[np.random.randint(0,len(list(G.edges())))] #Choose a random link from 
oLinks = np.array(shufflepair(link[0],link[1])) #Add link to output list
G.remove_edge(link[0],link[1]) # Remove link from network

#Loop

i=1
while G.number_of_edges()>0 and 2*i<N:
    i=i+1
#    print("Links in the network:",len(list(G.edges()))," Links in the output:",len(list(oLinks)))
    r=np.random.rand()
    flag = 0
    if r<p: # With probability p, choose a neighbour
        #Collect all neigbours of both nodes:
        node1=oLinks[len(oLinks)-1] # Last node
        neighbours1=list(nx.neighbors(G,node1)) 
        node2=oLinks[len(oLinks)-2] # Second to last node
        neighbours2=list(nx.neighbors(G,node2))
        neighbours=neighbours1+neighbours2 # Collection of all neighbours of node 1 and node 2
        if len(neighbours)>0: # If there are neighbours, pick one randomly
            rn = np.random.randint(0,len(neighbours)) #Index of neighbour
            nodeB = neighbours[rn]
            if rn < len(neighbours1): #Was a neighbour of node1
                nodeA=node1
            else: #Was a neighbour of node 2
                nodeA=node2            
            oLinks=np.append(oLinks,shufflepair(nodeA,nodeB))    # Add link to output list in random order
            G.remove_edge(nodeA,nodeB) # Remove link from network
            flag=1
    if flag == 0: # If no link was sampled in the chunck above, choose a random link
        link = list(G.edges())[np.random.randint(0,len(list(G.edges())))] #Choose a random link
        oLinks = np.append(oLinks,shufflepair(link[0],link[1])) #Add link to output list
        G.remove_edge(link[0],link[1]) # Remove from network


print("Finished successfully, preparing output")
#output

txt=''
for i in oLinks:
    txt=txt+str(i)+' '
    
rr=1
fileOut="../data/networks/sampling/Snet-p"+str(p)+"-r"+str(rr)+".dat"
while os.path.isfile(fileOut):
    rr=rr+1
    fileOut="../data/networks/sampling/Snet-p"+str(p)+"-r"+str(rr)+".dat"
outf=open(fileOut,"w")
outf.write(txt)
print("Output with "+str(len(oLinks))+" sampled nodes printed to file "+fileOut)
