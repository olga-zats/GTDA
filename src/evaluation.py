from tkinter.tix import ROW
import numpy as np


def read_file(path):
    with open(path, "r") as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)

    # if norm:
    #    score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    # else:
    #    score = D[-1, -1]
    # assert m_row == n_col
    score = D[-1, -1] / max(m_row, n_col) 
    return score

def frame_wise(s1, s2, norm=True):
    assert len(s1) == len(s2)
    metric = 1. - np.sum(s1 == s2) / len(s1)
    return metric


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)  


def eval_file(gt_content, recog_content, past_len, classes):
    last_frame = min(len(recog_content), len(gt_content))
    recognized = recog_content[past_len:last_frame]
    ground_truth = gt_content[past_len:last_frame]

    n_errors = 0
    for i in range(len(ground_truth)):
        if not recognized[i] == ground_truth[i]:
            n_errors += 1

    n_T = np.zeros(len(classes))
    n_F = np.zeros(len(classes))
    for i in range(len(ground_truth)):
        if ground_truth[i] == recognized[i]:
            n_T[ground_truth[i]] += 1
        else:
            n_F[ground_truth[i]] += 1

    return n_errors, len(recognized), n_T, n_F, edit_score(recognized, ground_truth)


