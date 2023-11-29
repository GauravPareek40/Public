# Set Covering Problem:
A optimization problem that involves finding the minimum cost set of sets that cover a given set of elements. 

### Script
This Python script solve instances of the Set Covering Problem using greedy algorithm, local search, and tabu search. 
It reads the user-specified file to translate the file into a cost array and a set covering matrix and tires to find a solution. 

### Output
The script provides detailed information during the each search process (Greedy, Local and Tabu), displaying the progress through multiple iterations. 
Each iteration number is logged, and if a better solution is found, the updated cost is indicated. The final output includes the cost of the  solution and the corresponding set(s) that form the optimized solution set.

### Note
The script allows exporting the results to CSV files, but the relevant code for this functionality is currently commented out. Uncomment the code to enable CSV export.
