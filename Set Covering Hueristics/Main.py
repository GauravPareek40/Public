from all_funcations import *
filename = input(f"\nEnter the name of the file to read: ")

#Reading user specified file to build the coast array and set covering matrix
try:
    rows,cols,cost_array,matrix = build_set_matrix(filename)            # Cost array and set covring matrix  
except FileNotFoundError:                                               # If user specified file not found. 
        print(f"Error: {filename} not found.\n")                            # Print Error
        raise                                                           # Raisng error (Terminating programm)        

print(f'Set Coveing matrix \n{matrix}\n')              #Printing resultant matrix                      
print(f"Elements Cost array: {cost_array}")            #Printing Cost Array

### elements need to be covered by
rows_with_ones = np.any(matrix == 1, axis=1)            # Boolean array summarizing if atleast 1 set cover the element 
elements_to_cover = np.where(rows_with_ones)[0]         # Elements covered by atleast 1 set  
print(f"Elements to cover :  {elements_to_cover}\n")    # elements_to_cover+1 can be used to start the referencinf from 1

#############################################################################
### Greedy Algorithm
solution,solution_cost = Greedy_sol(elements_to_cover = elements_to_cover,
                                    cost_array = cost_array,
                                    matrix = matrix)   
print(f"Solution Cost: {solution_cost}")
print(f"Solution Set : {[x for x in solution]}")            # x+1 can be used to start the referencinf from 1
print(f"-------------------------------------------------\n")

#############################################################################
### Iteration limit
iterations_allowed = input("Please enter the number neighbouring solutions to search : ")
if iterations_allowed == "":                                                     
      print(f"Evalauting till first 50 neighbouring sets")
      iterations_allowed = 50
else:
    iterations_allowed  = int(iterations_allowed)

#############################################################################
### Neighbourhood search
sets_tried,tried_sets_cost,new_cost_of_solution,new_solution_set = neigbourhood_search(passsed_elements_to_cover= elements_to_cover,
                                                                                       cost_array = cost_array,
                                                                                       matrix = matrix,
                                                                                       passed_solution = solution,
                                                                                       allowed_iterations=iterations_allowed)

print(f"Cost of Local Search Solution : {new_cost_of_solution}")                                      # Minimum Cost 
if len(new_solution_set) > 1:                                                                         # If more than 1 optimal solutio found
    print(f"Solution sets from Local Search:")
    for i in range(len(new_solution_set)):                                                            # For each optimal solution
        print(f"Option {i+1} {[x for x in new_solution_set[i]]}")                                     # x+1 can be used to start the referencinf from 1
else:
    print(f"Solution set from Local Search: {[x for x in new_solution_set[0]]}")                      # Set with lowest cost if only 1 found.
                                                                                                      # x+1 can be used to start the referencinf from 1
print("")                                                                                             #Blank line for astheitics

# Export to CSV
# local_search_best_solution = pd.DataFrame({'Cost of solution': new_cost_of_solution, 'Solution set from Local Search': new_solution_set})
# local_search_best_solution.to_csv(f'{filename}_local_search_output.csv', index=False)
# with open(f'local_search_output.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     # write var1 to the first row
#     writer.writerow(['Solution set from Local Search', new_solution_set])

#     # write var2 to the second row
#     writer.writerow(['Cost of solution',new_cost_of_solution])

#############################################################################
#Tabu search
sets_tried,tried_sets_cost,new_cost_of_solution,new_solution_set = tabu_search(passsed_elements_to_cover= elements_to_cover,
                                                                                       cost_array = cost_array,
                                                                                       matrix = matrix,
                                                                                       passed_solution = solution,
                                                                                       allowed_iterations=iterations_allowed)

print(f"Cost of Tabu Search Solution : {new_cost_of_solution}")                                      # Minimum Cost 
if len(new_solution_set) > 1:                                                                         # If more than 1 optimal solution found
    print(f"Solution sets from Tabu Search:")
    for i in range(len(new_solution_set)):                                                            # For each optimal solution
        print(f"Option {i+1} {[x for x in new_solution_set[i]]}")                                      # x+1 can be used to start the referencinf from 1
else:
    print(f"Solution set from Tabu Search: {[x for x in new_solution_set[0]]}")                    # Set with lowest cost if only 1 found.
                                                                                                      # x+1 can be used to start the referencinf from 1
print("")                                                                                             #Blank line for astheitics

#############################################################################
# Export to CSV
# tabu_search_best_solution = pd.DataFrame({'Cost of solution': new_cost_of_solution, 'Solution set from Local Search': new_solution_set})
# tabu_search_best_solution.to_csv(f'{filename}_tabu_search_output.csv', index=False)
# with open(f'tabu_search_output.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     # write var1 to the first row
#     writer.writerow(['Solution set from Local Search', new_solution_set])

#     # write var2 to the second row
#     writer.writerow(['Cost of solution',new_cost_of_solution])
