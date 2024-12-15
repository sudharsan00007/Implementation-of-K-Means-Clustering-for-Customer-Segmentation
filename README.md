# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: sudharsan
ResgisterNumber:  24009664
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv("C:/Users/sudharshan/Downloads/Mall_Customers.csv")


print(data.head())

# Check the data info and missing values
print(data.info())
print(data.isnull().sum())

# Elbow method to find the optimal number of clusters
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):  # Loop over 1 to 10 clusters
    kmeans = KMeans(n_clusters=i, init="k-means++")
    kmeans.fit(data.iloc[:, [3, 4]])  # Assuming the columns for clustering are 'Annual Income' and 'Spending Score'
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Fit KMeans with the optimal number of clusters (let's assume 5 clusters based on the Elbow plot)
kmeans = KMeans(n_clusters=5)
kmeans.fit(data.iloc[:, [3, 4]])  # Same columns for clustering
y_pred = kmeans.predict(data.iloc[:, [3, 4]])

# Assign the predicted clusters back to the dataframe
data["cluster"] = y_pred

# Creating separate dataframes for each cluster
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]

# Plot the clusters
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="Cluster 0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="black", label="Cluster 1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="blue", label="Cluster 2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="green", label="Cluster 3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="magenta", label="Cluster 4")

# Adding labels and title to the plot
plt.legend()
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

*/
```
## Output:
![Screenshot 2024-12-15 123017](https://github.com/user-attachments/assets/3e6a8498-84da-4fde-9292-001f5cc71792)
![Screenshot 2024-12-15 123027](https://github.com/user-attachments/assets/529d57c8-cdaf-43a9-80d5-535ee332405b)
![Screenshot 2024-12-15 123036](https://github.com/user-attachments/assets/f78832a0-0778-4efe-be1f-9bb60326138b)
![Screenshot 2024-12-15 123044](https://github.com/user-attachments/assets/9e6e7739-e02d-4508-b974-a9aa4e1e22f2)
![Screenshot 2024-12-15 123050](https://github.com/user-attachments/assets/43c99062-54bd-44bf-8ce8-362af6af029b)




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
