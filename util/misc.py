import os
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
import multiprocessing as mp
from fast_soft_sort.pytorch_ops import soft_sort


def set2mask(set_data_list, input_size):
    # init mask
    mask_data = np.zeros((input_size, input_size))

    # for each set
    for i, set_data in enumerate(set_data_list):
        num_elements = len(set_data)
        tmp_mask_data = np.zeros((input_size, 1))
        tmp_mask_data[set_data] = 1.0
        tmp_mask_data /= float(num_elements)
        mask_data[:, set_data] = tmp_mask_data
    return mask_data


def single_pava(z):
    # init parition info
    p_val = []
    p_cnt = []
    p_set = []
    p_idx = -1

    # for each value
    for i, val in enumerate(z):
        # if first, or current value is larger than others
        if i == 0 or val > p_val[p_idx]:
            # add value as new partition
            p_set.append([i])
            p_val.append(val)
            p_cnt.append(1)
            p_idx += 1
            continue
        # if the value is same as the latest one, just insert
        elif val == p_val[p_idx]:
            # only count up
            p_set[p_idx].append(i)
            p_cnt[p_idx] += 1
            continue

        # if current value is smaller than the current value
        assert val < p_val[p_idx]
        # update partition info
        p_set[p_idx].append(i)
        p_val[p_idx] = (p_val[p_idx] * p_cnt[p_idx] + val) / float(p_cnt[p_idx] + 1)
        p_cnt[p_idx] += 1

        # clean up
        while p_idx > 0:
            # if current parition is equal or smaller than the previous partition
            if p_val[p_idx] <= p_val[p_idx - 1]:
                # merge
                p_set[p_idx - 1] += p_set[p_idx]
                p_val[p_idx - 1] = (p_val[p_idx] * p_cnt[p_idx] + p_val[p_idx - 1] * p_cnt[p_idx - 1]) / float(
                    p_cnt[p_idx] + p_cnt[p_idx - 1])
                p_cnt[p_idx - 1] = p_cnt[p_idx] + p_cnt[p_idx - 1]
                p_set.pop(p_idx)
                p_val.pop(p_idx)
                p_cnt.pop(p_idx)
                p_idx -= 1
            else:
                break
    return set2mask(p_set, len(z))


def multi_pava(z_array):
    pool = mp.Pool(processes=mp.cpu_count())
    output_mask = np.stack(pool.map(single_pava, z_array), 0)
    pool.close()
    return output_mask


# forward pava (differentiable)
def pava_forward(input_data):
    # data size
    batch_size, num_quantiles = input_data.size()

    # for each data, make mask
    input_mask = [single_pava(input_data[i].cpu().data.numpy()) for i in range(batch_size)]
    #input_mask = multi_pava(input_data.cpu().data.numpy())
    input_mask = np.stack(input_mask, 0)
    input_mask = torch.Tensor(input_mask).to(input_data.device)

    # based on mask, compute pava output
    output_data = torch.bmm(input_data.unsqueeze(1), input_mask.detach()).squeeze(1)
    return output_data


def isotonic(input_data, quantile_list):
    quantile_list = np.array(quantile_list).reshape(-1)
    batch_size = input_data.shape[0]
    new_output_data = []
    for i in range(batch_size):
        new_output_data.append(IsotonicRegression().fit_transform(quantile_list, input_data[i]))
    return np.stack(new_output_data, 0)


def fix_crossing(predict_data, fix_type=0):
    is_torch = True
    if type(predict_data) is not torch.Tensor:
        is_torch = False
        predict_data = torch.Tensor(predict_data)

    # number of quantiles
    num_quantiles = predict_data.size()[-1]

    # above 50% and below 50%
    if fix_type == 0:
        # split into below 50% and above 50%
        idx_50 = num_quantiles // 2

        # below 50%
        below_50 = predict_data[:, :(idx_50 + 1)].contiguous()
        below_50 = torch.flip(torch.cummin(torch.flip(below_50, [-1]), -1)[0], [-1])

        # above 50%
        above_50 = predict_data[:, idx_50:].contiguous()
        above_50 = torch.cummax(above_50, -1)[0]

        # refined output
        ordered_data = torch.cat([below_50[:, :-1], above_50], -1)
    # from 0% to 100%
    elif fix_type == 1:
        ordered_data = torch.cummax(predict_data, -1)[0]
    # from 0% to 100% and from 100% to 0%
    elif fix_type == 2:
        min_ordered_data = torch.flip(torch.cummin(torch.flip(predict_data, [-1]), -1)[0], [-1])
        max_ordered_data = torch.cummax(predict_data, -1)[0]
        ordered_data = 0.5 * (min_ordered_data + max_ordered_data)
    else:
        ordered_data = predict_data

    if is_torch:
        return ordered_data
    else:
        return ordered_data.data.cpu().numpy()


# forward sorting (differentiable)
def sort_forward(input_data, regularization_strength):
    return soft_sort(input_data, regularization_strength=regularization_strength)

# make directory
def make_dir(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as e:
            raise ValueError(e)
