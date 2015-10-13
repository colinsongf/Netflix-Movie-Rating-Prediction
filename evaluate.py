#################################################################
#
#   __author__ = 'yanhe'
#
#   evaluate:
#       predict rating in the dev set and test set
#
#################################################################


import csv
import numpy as np
import rating_matrix
import user_sim


# extract the movie_id and user_id pair
def pred_pair(pair_path):
    movie_list = []
    user_list = []
    with open(pair_path) as f:
        pair_data = csv.reader(f)
        for row in pair_data:
            movie_list.append(int(row[0]))
            user_list.append(int(row[1]))
    pair = np.column_stack((movie_list, user_list))
    return pair


# predict rating for each movie-user pair
def rating_pred(pair_path, k):
    pair = pred_pair(pair_path)
    train_mtx = rating_matrix.matrix_transfer(2)
    # TODO: diff dict
    user_knn_dict = user_sim.user_dot_sim(train_mtx, k)
    pred_list = []
    for row in pair:
        movie_id = row[0]
        user_id = row[1]
        user_knn_list = user_knn_dict[user_id]
        pred_rating = np.sum(np.take(train_mtx[movie_id, :], user_knn_list.tolist())) / float(k) + 3
        pred_list.append(pred_rating)
    # output the result
    file_writer(pred_list)
    return pred_list


# write the result into txt file
def file_writer(pred_list):
    # write the ranking result into txt file
    f = open('user_dev_pred.txt', 'w')
    num = len(pred_list)
    for idx in range(0, num):
        f.write("{}\n".format(pred_list[num]))


# use this line to execute the main function
if __name__ == "__main__":
    # pass the value of k
    rating_pred("HW4_data/dev.csv", 10)
