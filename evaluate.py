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
import timeit


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
    dot_sim_mtx = user_sim.user_dot_sim(train_mtx)
    pred_list = []
    for row in pair:
        movie_id = row[0]
        user_id = row[1]
        dot_sim_list = dot_sim_mtx[user_id]
        # top k+1 nearest neighbors
        user_knn_list = np.argsort(dot_sim_list)[::-1][0: k+1]
        if user_id in user_knn_list:
            position = np.where(user_knn_list == user_id)
            user_knn_list = np.delete(user_knn_list, position)
        else:
            user_knn_list = np.delete(user_knn_list, len(user_knn_list) - 1)
        pred_rating = np.sum(np.take(train_mtx[movie_id, :], user_knn_list.tolist())) / float(k) + 3
        pred_list.append(pred_rating)
    # output the result
    file_writer(pred_list)
    return pred_list


# write the result into txt file
def file_writer(pred_list):
    # write the ranking result into txt file
    f = open('eval/user_dev_pred.txt', 'w')
    num = len(pred_list)
    for idx in range(0, num):
        f.write("{}\n".format(pred_list[idx]))


# use this line to execute the main function
if __name__ == "__main__":
    # pass the value of k
    start = timeit.default_timer()
    rating_pred("HW4_data/dev.csv", 10)
    end = timeit.default_timer()
    print end - start
