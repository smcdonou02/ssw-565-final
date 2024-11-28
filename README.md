# ssw-565-final

## Author
Stephanie McDonough

## Description
Objective: Automate and optimize the code review process in OpenStack repositories by categorizing check-ins based on their architectural attributes.

P1. Code Check-in Clustering:
- [ ] Export a list of more than 500 code check-ins from OpenDev Links to an external site. 
- [ ] Cluster the check-ins based on architectural attributes (e.g., documentation, performance, security). 
- [ ] Analyze the statistics of the clusters, their percentage distribution, and the challenges in detecting certain clusters.

## Method
Used KMeans clustering to categorize code check-ins from OpenDev into architectural attributes like performance and usability. Pandas and numpy were used to statistically analyze the clustering output and matplotlib was used to visualize percentage distribution. 
