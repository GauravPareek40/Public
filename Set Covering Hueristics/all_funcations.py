import numpy as np
import pandas as pd
import csv                                                   #To export data to csv

def update_row(row_number,column_numbers,matrix):
    for col_number in column_numbers:                       # Updating elements of current row as per the data 
        matrix[row_number][int(col_number)] = 1             # Using col number as recived, as cols numbers start from 0 in file  
                                                            # Using row number as recived from code as it is gerated within code, hence shifitng by 1 not required
##################################################################################################################################

def build_set_matrix(filename):
    with open(filename, "r") as file:                       # Open file in read mode
        first_line = file.readline().split()                # Read first line of datafile
        rows = int(first_line[0])                           # Extract numbers of rows from first line
        cols = int(first_line[1])                           # Extract numbers of column from first line

        # Initialize empty matrix
        matrix = np.zeros((rows, cols))                     # Blank set covering matrix
        cost_array = np.zeros(cols)                         # Blank cost array

        ###################################################################################################################################
        # COST ARRAY
        elements_count = 0                                                # Variable to loop through to add cost values to array until the 
                                                                            # number of elements in cost array is equal to number of cols
        while elements_count < cols:                                      # While the cost matrix is not filled
            line = file.readline().split()                                      # Read line
            for item in line:                                                   # For each elemnt in the line, add it to the array
                cost_array[elements_count] = item
                elements_count += 1                                             # Increasing counter of values in array
        #print(cost_array)                                                # Printing cost Array to check result
        
        ##################################################################################################################################
        # SET MATRIX            
        filled_rows = []                                                  # Variable to keep track of rows updated
        
        for row_number in range(0,rows):
            count_of_col_to_update = int(file.readline())                                # Number of cols to be updated for the row, # not used
            column_numbers = file.readline().split()                                        # index of cols covering elemnt
            # print(f"line number givn - {count_of_col_to_update}")                                # Checking code
            # print(f"length of cols   - {len(column_numbers)}: {count_of_col_to_update}")         # Checking code
            if count_of_col_to_update==len(column_numbers):                               #Check for number of cols to updated and number of cols provided
                update_row(row_number,column_numbers,matrix)                                    # Update required elemtns of the row as per col numbers 
                filled_rows.append(row_number)                                                  # Updating variable to keep track of rows updated
            else:
                print(f"Difference between number of cols specifed for row {row_number} and actaul umber of cols provided to update.")    
        
        # Check to see if all elemts are covered by atleast 1 subset
        filled_rows = set(filled_rows)                                    # creating set of updated rows, as row can be updated mutiple times
        if len(filled_rows)== len(matrix) :                               # if length of udpated rows = number of rows in matrix
            print("All elemnts are covered by atleast one set.\n")                # All rows have been updated atleast once.
        else:                                                             # if not
            print("All elemnts are not covered.\n")                             # All rows have not been updated, An elemnt is missing from cover.
    return(rows,cols,cost_array,matrix)
##################################################################################################################################

def Greedy_sol(elements_to_cover,cost_array,matrix):
    df = pd.DataFrame(matrix)
    rows = matrix.shape[0]
    cols = matrix.shape[1]

    subset = {}                                                           # Empty dictionary to store the elemnts coverd by each subsets
    #----------------------------------- Simple and vectorized code
    # for col in df.columns:                                                # Iterating over columns of df to create list of elementns covred by the column(set)
    #     subset[col] = {"indices": df.index[df[col]==1].tolist()}          # Filtering row values in each column to only include rows where the value is 1
    subset = df.apply(lambda col: {"indices": df.index[col==1].tolist()}, axis=0).to_dict()      #vetor form of above commented code
    #-----------------------------------
    
    solution_set = []                                                     # Solution set covering all elments required to be covered

    while elements_to_cover.size > 0 :                                    # Repeating process till no elments are left to cover
        # print(f"Size of elemnts left to cover : {elements_to_cover.size}") #Check code
        matches = np.zeros(cols)                                          # Resetting matches array (common element of to cover and each subset array) for iteration

        for key, value in subset.items():                                 # For each subset
            if key not in solution_set:                                   # if set is not already present in the solution set     
                # matches[key] = sum(int(element in value["indices"]) for element in elements_to_cover)  #Count of common element bwteen to cover ans set array
                matches[key] = np.sum(np.isin(elements_to_cover, value["indices"]))  #Count of common element bwteen to cover ans set array
 
        #----------------------------------- Simple and vectorized code
        # set_avg_cost = []                                                 # Calauting avg cost of set
        # for i in range(len(cost_array)):                                  
        #     if matches[i] == 0:                                           # if no match
        #         set_avg_cost.append(0)                                         # cost of set is 0, not useful set
        #     else:                                                         # if common elemnts exist between set and elemnts needed to cover  
        #         set_avg_cost.append(round(cost_array[i] / matches[i],2))       # avg = set cost / cunt of common elements
        
        # Convert matches and cost_array to numpy arrays
        matches_arr = np.array(matches)
        cost_arr = np.array(cost_array)

        # Calculate cost
        nonzero_matches = matches_arr != 0
        set_avg_cost = np.zeros(len(cost_array))
        set_avg_cost[nonzero_matches] = np.round(cost_arr[nonzero_matches] / matches_arr[nonzero_matches], 2)
    #-----------------------------------
    
        # print(np.vstack((matches, set_avg_cost)))                        # matrix summarizing count of common elements and set_avg_cost
        skip_indices = set(solution_set)                                  # Index of sets already present in solution and hence need to be skipped   
        min_value = max(cost_array)+10                                    # Deafult minimum set_avg_cost (kept higher to avoid error/skipping in follwing loop)
        min_index = None                                                  # Deafult index of minimum set_avg_cost
    
    #----------------------------------- Simple and vectorized code
        # for index,cost in enumerate(set_avg_cost):                        # For all sets presernnt inding set with lowest avg cost that is already not present in the solution
        #     if index not in skip_indices:
        #         if cost!=0 and cost < min_value:                          # if cost is not zero (zero useful elemnt sin set) and lower than current best
        #             min_value = cost                                             # Updating min cost to cost of current lowest avg cost
        #             min_index = index                                            # Updaitng index to index of current lowest avg cost

        indices = np.array([index for index, cost in enumerate(set_avg_cost) if index not in skip_indices and cost != 0 and cost < min_value])
        costs = np.array([cost for cost in set_avg_cost if cost != 0 and cost < min_value])

        if len(indices) > 0:
            min_index = indices[np.argmin(costs)]
            min_value = costs.min()
    #-----------------------------------
        #print(f"Selecting set {min_index}")
        solution_set.append(min_index)                                   # Adding the selected subset to solution set
        matched_elements = subset[min_index]['indices']                  # Elements covered by the selected subset 
        #print(f"matched_elements : {[x for x in matched_elements]}")     # List of elemnts coverd by the selected subset

        mask = np.isin(elements_to_cover, matched_elements)              # Mask of Elements covered by the selected subset 
        elements_to_cover = np.delete(elements_to_cover, np.where(mask)) # Removing elements covered by subset in solution from elemts to cover 

    # cost_of_solution = sum([cost_array[i] for i in solution_set])        #Final cost of including the solution set  
    cost_of_solution = cost_array[solution_set].sum()
    #print(f"Solution cost = {cost_of_solution}")
    
    return (solution_set,cost_of_solution)
##################################################################################################################################

### Neigbourhood Combinaton - Alterantive of combination funcation from itertools
def combinations(arr, k):                              #Function to create incomplete neigbourhood sets
    if k == 0:                                              # If number of elemnts to combination set is 0
        return [[]]                                             # Return blank set
    if not arr:                                             # If passed element is empty
        return []                                               # Return blank 
    result = []                                             # Array to poluate combinations of the k length elemnts 
    for i in range(len(arr)):                               #looping  through all elemnts
        elem = arr[i]                                           
        rest = arr[i+1:]
        for c in combinations(rest, k-1):               #Recurive call to find combination for next k-1 swaps
            result.append([elem] + c)                       # Appending combinations of set with {len(solution)-k} elements from exisint sol:
    return result
##################################################################################################################################

### Neigbourhood Genrator  
def genrate_neighbourhood(passed_solution):
    neighbourhood = []                                                              #Array to popuate all partail neigbourhood sets
    for k in range(1,len(passed_solution)):                                      #Looping for
        comb_result = combinations(passed_solution, len(passed_solution)-k)          #possible combinations of set with {len(solution)-k} elements from exisint sol
        # print(f"possible combinations of set with {len(solution)-k} elements from exisint sol: - {len(comb_result)}")
        neighbourhood = neighbourhood + comb_result                                       # Appending combinations of loop to neigbourhood array
    return neighbourhood
##################################################################################################################################

### Neigbourhood Search  
def neigbourhood_search(passsed_elements_to_cover,cost_array,matrix,passed_solution,allowed_iterations):
    original_cost_of_solution = sum([cost_array[i] for i in passed_solution])        #Cost of passed the solution set  
    
    #----------------------------------------------------------------------------------------------------------------------------------
    #Varibles to record of iteration and status of tested solutions at each iteration of loop
    iter_num = 0                                                      #variable to keep record of iterations completed
    new_cost_of_solution = original_cost_of_solution                  #Current best solution found after above iterations
    new_solution_set = passed_solution                                #Cost of the current best solution 
    sets_tried = []                                                   #Neighbourhood set tried (all are potetnial solutions)
    tried_sets_cost = []                                              #Cost of the tried Neighbourhood solution
    
    sets_tried.append(passed_solution)                                # Addding passed solution to tired sets  
    tried_sets_cost.append(original_cost_of_solution)                 # Addding cost of passed solution to tired sets  
    last_best = None                                                  # To keep reocord of last best solution
    #----------------------------------------------------------------------------------------------------------------------------------
    while iter_num < allowed_iterations:
        if new_solution_set == last_best:                             # If no need solution found
            pass                                                               # No changes, use same neighbourhood
        else:                                                         #if Better solution found
            neighbourhood = genrate_neighbourhood(new_solution_set)            # Genrate neighbouhood arounf better solution 
    #----------------------------------------------------------------------------------------------------------------------------------
    # Searching set to complete the neighbourhood set and obtain set with lowest cost  
        for result in neighbourhood:                                                # For each set in neighbourhood 
            if iter_num < allowed_iterations:                                           # If current iteration is less than allowed               
                print(f"Iteration number : {iter_num+1}\r",sep="")                                      
                elements_to_cover = passsed_elements_to_cover                           # Elemnts to be covered by the iteration solution set
                union = np.max(matrix[:, result], axis=1)                               # Binary of elements covered by selected subsets
                exisitbg_subset_elements = np.where(union)[0]                           # Index of Elements covered by selected subsets  
                mask = np.isin(elements_to_cover, exisitbg_subset_elements)             # Mask of Elements covered by the selected subset 
                elements_to_cover = np.delete(elements_to_cover, np.where(mask))        # Elemnts uncovred by iteration solution set 
        
                #---------------------------------------------------------------------------------------------------------------------------
                # Finding set to cover the uncovered elements 
                df = pd.DataFrame(matrix)
                rows = matrix.shape[0]
                cols = matrix.shape[1]

                subset = {}                                                           # Empty dictionary to store the elemnts coverd by each subsets
                subset = df.apply(lambda col: {"indices": df.index[col==1].tolist()}, axis=0).to_dict()
                
                iteration_set = result                                                     # Solution set covering all elments required to be covered
                while elements_to_cover.size > 0 :                                    # Repeating process till no elments are left to cover
                    matches = np.zeros(cols)                                          # Resetting matches array (common element of to cover and each subset array) for iteration

                    for key, value in subset.items():                                 # For each subset
                        if key not in iteration_set:      # if set is not already present in the solution set     
                            matches[key] = np.sum(np.isin(elements_to_cover, value["indices"]))          #Count of common element bwteen to cover and iteration set
                            
                    set_avg_cost = []                                                 # Calauting avg cost of set
                    # Convert matches and cost_array to numpy arrays
                    matches_arr = np.array(matches)
                    cost_arr = np.array(cost_array)

                    # Calculate cost
                    nonzero_matches = matches_arr != 0
                    set_avg_cost = np.zeros(len(cost_array))
                    set_avg_cost[nonzero_matches] = np.round(cost_arr[nonzero_matches] / matches_arr[nonzero_matches], 2)

                    min_value = max(cost_array)+10                                    # Deafult minimum set_avg_cost (kept higher to avoid error/skipping in follwing loop)
                    min_index = None                                                  # Deafult index of minimum set_avg_cost

                    indices = np.array([index for index, cost in enumerate(set_avg_cost) if cost != 0 and cost < min_value])
                    costs = np.array([cost for cost in set_avg_cost if cost != 0 and cost < min_value])

                    if len(indices) > 0:
                        min_index = indices[np.argmin(costs)]
                        min_value = costs.min()

                    # print(f"Selecting set {min_index}")
                    iteration_set.append(min_index)                                   # Adding the selected subset to solution set
                    matched_elements = subset[min_index]['indices']                  # Elements covered by the selected subset 
                    #print(f"matched_elements : {[x for x in matched_elements]}")     # List of elemnts coverd by the selected subset

                    mask = np.isin(elements_to_cover, matched_elements)              # Mask of Elements covered by the selected subset 
                    elements_to_cover = np.delete(elements_to_cover, np.where(mask)) # Removing elements covered by subset in solution from elemts to cover 

                # print(f"Iter sol (Non Increased by 1): {[ele for ele in iteration_set]}")
                cost_of_iteration_solution = sum([cost_array[i] for i in iteration_set])        # Final cost of including the solution set  
                sets_tried.append(iteration_set)                                                # Updating list of tired sets  
                tried_sets_cost.append(cost_of_iteration_solution)
                
                #Notifying user if better solution found
                if cost_of_iteration_solution < new_cost_of_solution:
                    print(f"Better solution found, cost : {cost_of_iteration_solution}")        
                    new_cost_of_solution = cost_of_iteration_solution                           #updating cost threshold to notfy user to new best cost
                    last_best = new_solution_set                                                #tracking last best result to see if neihbouud needs to be updated
                    new_solution_set = iteration_set                                            #Ebst result of the iteration till now
                    break
            iter_num +=1
                

    #Finding best solution of all neighbourhood sets evalauted
    best_cost = min(tried_sets_cost)                                                       #Lowest Cost
    best_cost_indices = [i for i, x in enumerate(tried_sets_cost) if x == best_cost]       #Index of lowest cost in all cost (as multiple sets can have same cost)
    best_solutions = [sets_tried[i] for i in best_cost_indices]                            #Index of solution having the lowest cost.

    best_solutions = list(set(tuple(sorted(c)) for c in best_solutions))                   #return list of anly unique combinations
    print("")

    # zipped_data_cost_set = sorted(zip(tried_sets_cost, sets_tried), key=lambda x: x[0])
    # for cost,set in zipped_data_cost_set:
    #     print(f"{cost}: {set}")

    # print(f"\nMinimum cost :{best_cost}")
    # print(f"Solution sets :{best_solutions}\n")        #prints unincremented col number, need to add 1.
    return (sets_tried,tried_sets_cost,best_cost,best_solutions)

##################################################################################################################################

def tabu_search(passsed_elements_to_cover,cost_array,matrix,passed_solution,allowed_iterations):
    original_cost_of_solution = sum([cost_array[i] for i in passed_solution])        #Cost of passed the solution set  
    
    #----------------------------------------------------------------------------------------------------------------------------------
    #Varibles to record of iteration and status of tested solutions at each iteration of loop
    iter_num = 0                                                      #variable to keep record of iterations completed
    new_cost_of_solution = original_cost_of_solution                  #Current best solution found after above iterations
    new_solution_set = passed_solution                                #Cost of the current best solution 
    sets_tried = []                                                   #Neighbourhood set tried (all are potetnial solutions)
    tried_sets_cost = []                                              #Cost of the tried Neighbourhood solution
    tabu_index = None                                                 #Subsets to skip while searching neigbourhood during iteration

    sets_tried.append(passed_solution)                                # Addding passed solution to tired sets  
    tried_sets_cost.append(original_cost_of_solution)                 # Addding cost of passed solution to tired sets  

    #---------------------------------------------------------------------------------------------------------------------------------
    neighbourhood = genrate_neighbourhood(passed_solution)
    #----------------------------------------------------------------------------------------------------------------------------------
    # Searching set to complete the neighbourhood set and obtain set with lowest cost  
    for result in neighbourhood:                                                # For each set in neighbourhood 
        if iter_num < allowed_iterations:                                           # If current iteration is less than allowed
            print(f"Iteration number : {iter_num+1}\r",sep="")                                      
            elements_to_cover = passsed_elements_to_cover                           # Elemnts to be covered by the iteration solution set
            union = np.max(matrix[:, result], axis=1)                               # Binary of elements covered by selected subsets
            exisitbg_subset_elements = np.where(union)[0]                           # Index of Elements covered by selected subsets  
            mask = np.isin(elements_to_cover, exisitbg_subset_elements)             # Mask of Elements covered by the selected subset 
            elements_to_cover = np.delete(elements_to_cover, np.where(mask))        # Elemnts uncovred by iteration solution set 
    
            #---------------------------------------------------------------------------------------------------------------------------
            # Finding set to cover the uncovered elements 
            df = pd.DataFrame(matrix)
            rows = matrix.shape[0]
            cols = matrix.shape[1]

            subset = {}                                                           # Empty dictionary to store the elemnts coverd by each subsets
            subset = df.apply(lambda col: {"indices": df.index[col==1].tolist()}, axis=0).to_dict()
            
            iteration_set = result                                                     # Solution set covering all elments required to be covered
            tabu_index = list(set(passed_solution) - set(iteration_set))               # Sets to skip from being included in the new solution serach
            # print(f"Sets to skips while looking for sollution: {[x+1 for x in tabu_index]} ")  #Subset to skip
            
            while elements_to_cover.size > 0 :                                    # Repeating process till no elments are left to cover
                matches = np.zeros(cols)                                          # Resetting matches array (common element of to cover and each subset array) for iteration

                for key, value in subset.items():                                 # For each subset
                    if key not in iteration_set and  key not in tabu_index:      # if set is not already present in the solution set     
                        matches[key] = np.sum(np.isin(elements_to_cover, value["indices"]))          #Count of common element bwteen to cover and iteration set
                        
                set_avg_cost = []                                                 # Calauting avg cost of set
                # Convert matches and cost_array to numpy arrays
                matches_arr = np.array(matches)
                cost_arr = np.array(cost_array)

                # Calculate cost
                nonzero_matches = matches_arr != 0
                set_avg_cost = np.zeros(len(cost_array))
                set_avg_cost[nonzero_matches] = np.round(cost_arr[nonzero_matches] / matches_arr[nonzero_matches], 2)

                tabu_indices = set(iteration_set)                                  # Index of sets already present in solution and hence need to be skipped   
                min_value = max(cost_array)+10                                    # Deafult minimum set_avg_cost (kept higher to avoid error/skipping in follwing loop)
                min_index = None                                                  # Deafult index of minimum set_avg_cost

                indices = np.array([index for index, cost in enumerate(set_avg_cost) if index not in tabu_indices and cost != 0 and cost < min_value])
                costs = np.array([cost for cost in set_avg_cost if cost != 0 and cost < min_value])

                if len(indices) > 0:
                    min_index = indices[np.argmin(costs)]
                    min_value = costs.min()

                #print(f"Selecting set {min_index}")
                iteration_set.append(min_index)                                   # Adding the selected subset to solution set
                matched_elements = subset[min_index]['indices']                  # Elements covered by the selected subset 
                #print(f"matched_elements : {[x for x in matched_elements]}")     # List of elemnts coverd by the selected subset

                mask = np.isin(elements_to_cover, matched_elements)              # Mask of Elements covered by the selected subset 
                elements_to_cover = np.delete(elements_to_cover, np.where(mask)) # Removing elements covered by subset in solution from elemts to cover       

            # print(f"Current sol (Non Increased by 1): {[ele+1 for ele in iteration_set]}")
            cost_of_iteration_solution = sum([cost_array[i] for i in iteration_set])        # Final cost of including the solution set  
            sets_tried.append(iteration_set)                                                # Updating list of tired sets  
            tried_sets_cost.append(cost_of_iteration_solution)

            #Notifying user if better solution found
            if cost_of_iteration_solution < new_cost_of_solution:
                print(f"Better solution found, cost : {cost_of_iteration_solution}")        
                new_cost_of_solution = cost_of_iteration_solution                           #updating cost threshold to notfy user to new best cost
            iter_num +=1

    #Finding best solution of all neighbourhood sets evalauted
    best_cost = min(tried_sets_cost)                                                       #Lowest Cost
    best_cost_indices = [i for i, x in enumerate(tried_sets_cost) if x == best_cost]       #Index of lowest cost in all cost (as multiple sets can have same cost)
    best_solutions = [sets_tried[i] for i in best_cost_indices]                            #Index of solution having the lowest cost.

    print("")
    best_solutions = list(set(tuple(sorted(c)) for c in best_solutions))                   #return list of anly unique combinations

    return (sets_tried,tried_sets_cost,best_cost,best_solutions)

##################################################################################################################################
##################################################################################################################################
