#################################################################
#
#   __author__ = 'yanhe'
#
#   rating_matrix:
#       transfer the rating matrix from dataset
#
#################################################################


import csv
import numpy as np
import scipy.sparse as sparse


# explore the statistics about the training dataset
def matrix_transfer(option):
    train_path = "HW4_data/train.csv"
    row_list = []
    col_list = []
    data_list = []
    with open(train_path) as f:
        train_data = csv.reader(f)
        for row in train_data:
            # store movie_id in row_list
            row_list.append(int(row[0]))
            # store user_id in col_list
            col_list.append(int(row[1]))
            # store rating in data_list
            data_list.append(float(row[2]))
    train_coo_mtx = sparse.coo_matrix((data_list, (row_list, col_list)), dtype=np.float)
    train_mtx = train_coo_mtx.toarray()
    if option == 2:
        train_mtx -= 3
    return train_mtx


# use this line to execute the main function
# if __name__ == "__main__":
#     matrix_transfer(option)
