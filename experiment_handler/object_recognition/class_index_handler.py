import json
import os

CLASS_INDEX = None
CURRENT_CLASS_SET = None


def load_class_index_file(class_set):
    """
    Loads the class index file into memory for the current class set.
    """
    global CLASS_INDEX, CURRENT_CLASS_SET

    dir_path = os.path.dirname(os.path.realpath(__file__))
    class_index_file = os.path.join(dir_path, "class_label_lookup", class_set + "_class_index.json")
    CLASS_INDEX = json.load(open(class_index_file))

    CURRENT_CLASS_SET = class_set


def get_label_for_index(index, class_set):
    """
    Find to which label and index of the prediction array relates to.
    """
    global CLASS_INDEX, CURRENT_CLASS_SET

    if CLASS_INDEX is None or CURRENT_CLASS_SET != class_set:
        load_class_index_file(class_set)

    return CLASS_INDEX[str(index)][1]


def get_top_n_labels(predictions, class_set, top=5):
    """
    Find labels tor the top n predictions

    Arguments:
        predictions (Numpy array): array of predictions (single row)
        class_set (str):  which labelset is used (e.g. imagenet or tvwall-1)
        top (int): how many items should be returned

    Returns:
         list: list of strings of top class predictions '[label_top1, label_top2, label_top3]'
    """
    global CLASS_INDEX, CURRENT_CLASS_SET

    if CLASS_INDEX is None or CURRENT_CLASS_SET != class_set:
        load_class_index_file(class_set)

    top_indices = predictions.argsort()[-top:][::-1]
    result = [CLASS_INDEX[str(i)][1] for i in top_indices]
    return result
