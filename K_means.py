import random as rand
import math
import numpy as np
import time

    
#K-Means Algorithm
def kmeans(k,datapoints):

    # dim - caculate the dimensionality of Datapoints vectors
    dim = np.size(datapoints,1) 
    points_number=np.size(datapoints,0)
    
    cluster_centers = np.zeros([points_number,dim])
    new_cluster= np.ones([points_number,1])#initliaze arrays
    last_cluster=np.zeros([points_number,1]) 
    
   
    Max_Iterations = 100 #the Limits of our iterations is 100
    Iterations_count = 0
    
    for i in range(0,k):#choose random centers for the clusters
        cluster_centers[i]=(rand.choice(datapoints))
        
   
       
    #check if there is changes between this cluster and the previuse one
    while (not((new_cluster==last_cluster).all()) and (Iterations_count < Max_Iterations))  :
        
        last_cluster= np.copy(new_cluster)
        Iterations_count += 1
    
        #Update Point's Cluster Alligiance
        #p is index of point in our data
        for p in range(0,points_number):#loop thats goes over all our data points
            min_dist = 100.0
            
            #Check min_distance against all centers
            for i in range(0,k):
                dist = np.linalg.norm(datapoints[p]-cluster_centers[i])
                if (dist < min_dist):
                    min_dist = dist  
                    new_cluster[p] = i  # this mean that the point data[p] is moved to cluster k
        
        
        #Update Cluster's Position
        for i in range(0,k):#loop for each cluster center
            new_center = [-1] * dim
            elements = 0
            for p in range(0,points_number):#p is the row now which means a point in data
                if (new_cluster[p] == i): #If this point belongs to the cluster k
                    for j in range(0,dim):
                        new_center[j] += datapoints[p][j]#sum of all elements in cols p that in the same cluster k
                    elements += 1
            
            for j in range(0,dim):
                if elements != 0:
                    new_center[j] = round(new_center[j] / elements) 
                
                
                else: 
                    new_center = rand.choice(datapoints)

            cluster_centers[k] = new_center
                   
    for k in range(0,len(cluster_centers)):
          
            for p in range(0,points_number):#p is the row now which means a point in data
                if (new_cluster[p] == k): #If this point belongs to the cluster
                    print(k+1, datapoints[p])
                          
            
        

if __name__ == "__main__":
     
    start_time = time.time()
    datapoints = np.array([[ 3, 2, 7, 2],[ 2, 2, 3, 3],[1 ,2, 6, 6],[ 0, 1, 8, 8],[ 1, 0, 9, 9],[ 1, 1, 1, 1],[ 5, 6, 6, 5],[ 7, 7, 7, 8],[ 9, 10, 8, 8],[ 11, 13, 14, 14]
    ,[ 12, 12, 15, 14],[ 12, 13, 15, 14],[ 13, 13, 10, 9],[ 53, 53, 50, 59],[ 63, 63, 60, 99],[ 13, 43, 40, 44],[ 14, 13, 10, 4]
    ,[ 19, 19, 9, 9],[ 13, 15, 10, 9],[ 15, 19, 10, 9],[ 18, 18, 10, 9],[ 88, 88, 10, 9],[ 90, 93, 10, 9],[ 73, 73, 10, 9],[ 0, 13, 10, 9],[ 1, 13, 10, 9]
    ,[ 1, 11, 18, 9],[ 43, 13, 40, 9],[ 66, 66, 10, 9],[ 78, 87, 10, 9],[ 19, 17, 144, 9],[ 200, 13, 100, 9],[ 130, 130, 10, 9]])

    k =3 # K - Number of Clusters
      
    kmeans(k,datapoints) 
    print("--- %s seconds ---" % (time.time() - start_time))