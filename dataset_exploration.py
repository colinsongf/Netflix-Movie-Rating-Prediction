#################################################################
#
#   __author__ = 'yanhe'
#
#   dataset_exploration:
#       calculate the statistical parameters in training set
#
#################################################################


import csv
import numpy as np
import scipy.sparse as sparse


# explore the statistics about the training dataset
def matrix_transfer():
    train_path = "HW4_data/train.csv"
    row_list = []
    col_list = []
    data_list = []
    with open(train_path) as f:
        train_data = csv.reader(f)
        for row in train_data:
            # store movie_id in row_list
            row_list.append(float(row[0]))
            # store user_id in col_list
            col_list.append(float(row[1]))
            # store rating in data_list
            data_list.append(float(row[2]))
    train_coo_mtx = sparse.coo_matrix((data_list, (row_list, col_list)), dtype=np.float)
    # print train_coo_mtx.shape
    return train_coo_mtx.toarray()


def explorer():
    train_mtx = matrix_transfer()
    # ***** part one: statistics *****
    rating_one = np.where(train_mtx == 1)
    print 'movie with rating 1: ', rating_one[0].size, '\n'
    rating_two = np.where(train_mtx == 2)
    print 'movie with rating 2: ', rating_two[0].size, '\n'
    rating_three = np.where(train_mtx == 3)
    print 'movie with rating 3: ', rating_three[0].size, '\n'
    rating_four = np.where(train_mtx == 4)
    print 'movie with rating 4: ', rating_four[0].size, '\n'
    rating_five = np.where(train_mtx == 5)
    print 'movie with rating 5: ', rating_five[0].size, '\n'
    rating_avg = np.sum(train_mtx) / np.count_nonzero(train_mtx)
    print 'movie rating average: ', rating_avg, '\n'

    # ***** part two: user_id 4321 *****
    curuser = train_mtx[:, 4321]
    movie_num = np.count_nonzero(curuser)
    print 'number of movie rated: ', movie_num, '\n'
    rating_one_num = np.where(curuser == 1)
    print 'movie with rating 1: ', rating_one_num[0].size, '\n'
    rating_three_num = np.where(curuser == 3)
    print 'movie with rating 3: ', rating_three_num[0].size, '\n'
    rating_five_num = np.where(curuser == 5)
    print 'movie with rating 5: ', rating_five_num[0].size, '\n'
    rating_avg_score = np.sum(curuser) / np.count_nonzero(curuser)
    print 'movie rating average: ', rating_avg_score, '\n'

    # ***** part three: movie_id 3 ****
    curmovie = train_mtx[3, :]
    user_num = np.count_nonzero(curmovie)
    print 'number of user rated: ', user_num, '\n'
    rating_one_user = np.where(curmovie == 1)
    print 'movie with rating 1: ', rating_one_user[0].size, '\n'
    rating_three_user = np.where(curmovie == 3)
    print 'movie with rating 3: ', rating_three_user[0].size, '\n'
    rating_five_user = np.where(curmovie == 5)
    print 'movie with rating 5: ', rating_five_user[0].size, '\n'
    rating_avg_user = np.sum(curmovie) / np.count_nonzero(curmovie)
    print 'movie rating average: ', rating_avg_user, '\n'

# use this line to execute the main function
if __name__ == "__main__":
    explorer()
