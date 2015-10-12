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


def rating_pred(pair_path):
    pair = pred_pair(pair_path)


# use this line to execute the main function
if __name__ == "__main__":
    rating_pred("HW4_data/dev.csv")
