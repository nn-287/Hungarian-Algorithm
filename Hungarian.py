import copy
import numpy as np
from ortools.graph.python import linear_sum_assignment
from scipy.optimize import linear_sum_assignment
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import matplotlib.pyplot as plt



def Get_min(mat):

    for row_num in range(mat.shape[0]):

        mat[row_num] = mat[row_num] - np.min(mat[row_num])

    for col_num in range(mat.shape[1]):

        mat[:,col_num] = mat[:,col_num] - np.min(mat[:,col_num])


    return mat



def min_zero(zero_mat,mark_zero):
    min_sum = float('inf')
    min_row_index = -1
    zero_index = -1

    for row_num, row in enumerate(zero_mat):

        num_zeros = np.sum(row == True)

        if num_zeros > 0 and num_zeros < min_sum:
            min_sum = num_zeros
            min_row_index = row_num
            zero_index = np.where(row == True)[0][0]

    if min_row_index != -1:
        mark_zero.append((min_row_index, zero_index))
        zero_mat[min_row_index, :] = False
        zero_mat[:, zero_index] = False





def mark_matrix(mat):

    marked_zero = []
    zero_row = []
    zero_col = []

    current_mat = mat
    zero_bool_mat = (current_mat==0)
    zero_bool_mat_copy = zero_bool_mat.copy()



    while(True in zero_bool_mat_copy):

        min_zero(zero_bool_mat_copy, marked_zero)


    for i in range (len(marked_zero)):

        zero_row.append(marked_zero[i][0])
        zero_col.append(marked_zero[i][1])


    non_marked_row = list(set(range(current_mat.shape[0])) - set(zero_row))

    marked_cols = []
    check = True
    while check:

        check = False

        for i in range(len(non_marked_row)):
            row_array = zero_bool_mat[non_marked_row[i],:]
            for j in range(row_array.shape[0]):
                if(row_array[j]==True and j not in marked_cols):
                    marked_cols.append(j)
                    check = True



        for row_num, col_num in marked_zero:

            if row_num not in non_marked_row and col_num in marked_cols:

                non_marked_row.append(row_num)
                check_switch = True

    marked_rows = list(set(range(mat.shape[0])) - set(non_marked_row))
    return (marked_zero, marked_rows, marked_cols)




def final_matrix(mat,cover_rows,cover_cols):

    cur_mat = mat.copy()

    non_covered_rows = [row for row in range(len(cur_mat)) if row not in cover_rows]
    non_covered_cols = [col for col in range(len(cur_mat[0])) if col not in cover_cols]

    non_zero_elements = [cur_mat[row][col] for row in non_covered_rows for col in non_covered_cols]

    min_num = min(non_zero_elements)

    for row in non_covered_rows:
        for col in non_covered_cols:
            cur_mat[row, col] -= min_num

    for row in cover_rows:
        for col in cover_cols:
            cur_mat[row, col] += min_num

    return cur_mat



def hungarian_integration(cost_matrix):

    n = cost_matrix.shape[0]
    cur_mat = copy.deepcopy(cost_matrix)

    cur_mat = Get_min(cost_matrix)

    count_zero_lines = 0

    while count_zero_lines < n:

        ans_pos, marked_rows, marked_cols = mark_matrix(cur_mat)
        count_zero_lines = len(marked_rows) + len(marked_cols)

        if count_zero_lines < n:

            cur_mat = final_matrix(cur_mat, marked_rows,marked_cols)

    return ans_pos




def min_cost(mat,pos):
    total = 0
    ans_mat = np.zeros((mat.shape[0],mat.shape[1]))
    for i in range(len(pos)):
        total+=mat[pos[i][0], pos[i][1]]
        ans_mat[pos[i][0],pos[i][1]] = mat[pos[i][0], pos[i][1]]

    return total,ans_mat





#Driver code

cost_matrix = np.array([[19,28,31],
                        [11,17,16],
                        [12,15,13]])
pos = hungarian_integration(cost_matrix.copy())
ans,ans_mat = min_cost(cost_matrix,pos)

# print(ans,ans_mat)



#Driver code GUI

B = nx.DiGraph()

top_nodes = [1,2,3]
bottom_nodes = [4,5,6]

B.add_nodes_from(top_nodes,bipartite=0,color='#C5E0B4')
B.add_nodes_from(bottom_nodes,bipartite=1,color='#FFE699')

node_pos = nx.bipartite_layout(B,top_nodes)

B.add_edge(1,4,weight=19,color='b',width=1)
B.add_edge(1,5,weight=28,color='b',width=1)
B.add_edge(1,6,weight=31,color='b',width=1)

B.add_edge(2,4,weight=11,color='b',width=1)
B.add_edge(2,5,weight=17,color='b',width=1)
B.add_edge(2,6,weight=16,color='b',width=1)

B.add_edge(3,4,weight=12,color='b',width=1)
B.add_edge(3,5,weight=15,color='b',width=1)
B.add_edge(3,6,weight=13,color='b',width=1)




my_matching = bipartite.matching.minimum_weight_full_matching(B,top_nodes,"weight")
# print(my_matching)

assignments = list(my_matching.items())
edge_colors = ["r" if edge in assignments else '#C4C2C6' for edge in B.edges()]
edge_width = [5 if edge in assignments else 1 for edge in B.edges()]


node_colors = list(nx.get_node_attributes(B,'color').values())
nx.draw(B, pos=node_pos, with_labels=True, font_color='red', node_size=1000, node_color=node_colors, edge_color=edge_colors, width=edge_width)
label1=nx.get_edge_attributes(B,'weight')
nx.draw_networkx_edge_labels(B,node_pos,edge_labels=label1, label_pos=0.85)
# plt.show()