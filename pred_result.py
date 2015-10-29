#################################################################
#
#   __author__ = 'yanhe'
#
#   pred_result:
#       write the prediction result into txt file
#
#################################################################


# write the result into txt file
def file_writer(pred_list):
    # write the ranking result into txt file
    f = open('eval/predictions.txt', 'w')
    num = len(pred_list)
    for idx in range(0, num):
        f.write("%s\n" % pred_list[idx])
