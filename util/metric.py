import torch
import numpy as np


# numpy pinball loss
def pinball_loss_np(predict_data, target_data, quantiles, mean=False):
    error_data = target_data.reshape(-1, 1) - predict_data
    loss_data = np.maximum(quantiles * error_data, (quantiles - 1) * error_data)
    if mean:
        return loss_data.mean()
    else:
        return loss_data.mean(0)


# pinball loss
def pinball_loss(predict_data, target_data, quantiles):
    error_data = target_data.reshape(-1, 1) - predict_data
    loss_data = torch.max(quantiles * error_data, (quantiles - 1) * error_data)
    return loss_data.mean()


# huber-pinball loss
def huber_loss(predict_data, target_data, quantiles, alpha=0.01):
    if alpha == 0.0:
        return pinball_loss(predict_data, target_data, quantiles)

    error_data = target_data.reshape(-1, 1) - predict_data
    loss_data = torch.where(torch.abs(error_data) < alpha,
                            0.5 * error_data * error_data,
                            alpha * (torch.abs(error_data) - 0.5 * alpha))
    loss_data /= alpha

    scale = torch.where(error_data >= 0,
                        torch.ones_like(error_data) * quantiles,
                        torch.ones_like(error_data) * (1 - quantiles))
    loss_data *= scale
    return loss_data.mean()


# margin loss (between neighbored prediction)
def margin_loss(predict_data, margin_data):
    # number of samples
    batch_size, num_quantiles = predict_data.size()

    # compute margin loss (batch_size x output_size(above) x output_size(below))
    error_data = predict_data.unsqueeze(1) - predict_data.unsqueeze(2)

    # len(np.shape(margin_data)) ==0, means a scalar
    if len(np.shape(margin_data)) == 1:
        # margin data (num_quantiles) ===> (num_quantiles x num_quantiles)
        if type(margin_data) is not torch.Tensor:
            margin_data = torch.tensor(margin_data, device=predict_data.device)

        margin_data = margin_data.reshape(1, -1)
        margin_data = margin_data.permute(1, 0) - margin_data
        margin_data = torch.tril(margin_data, -1).relu()


    loss_data = torch.tril(error_data + margin_data, diagonal=-1)
    loss_data = loss_data.relu()
    loss_data = loss_data.sum() / np.float32(batch_size * (num_quantiles * num_quantiles - num_quantiles) * 0.5)

    # compute accumulated margin
    #if only_neighbored:
    #    loss_data = torch.tril(torch.triu(error_data + margin_data, diagonal=-1), diagonal=-1)
    #    loss_data = loss_data.relu()
    #    loss_data = loss_data.sum() / np.float32(batch_size * (num_quantiles - 1))

    return loss_data


# PICP, percentage of captured points (ratio of true observations falling inside the estimated prediction)
def prediction_interval_coverage_rate(y_target, y_lower, y_upper):
    return np.mean((y_target >= y_lower) & (y_target <= y_upper))


def mean_prediction_interval_coverage_rate(y_target, y_quantile, quantile_list):
    picp_list = []
    error_list = []

    # for each quantile level (from 1% to 99%)
    num_samples = y_quantile.shape[0]
    num_quantiles = len(quantile_list)
    assert num_quantiles == y_quantile.shape[1]
    y_target = y_target.reshape([num_samples, 1])

    # for each interval
    for i in range(num_quantiles // 2):
        # lower and upper index
        lower_idx = i
        upper_idx = -(i + 1)

        # lower and upper quantile
        lower_quantile = quantile_list[lower_idx]
        upper_quantile = quantile_list[upper_idx]

        # get predicted lower and upper values
        y_lower = y_quantile[:, lower_idx].reshape([num_samples, 1])
        y_upper = y_quantile[:, upper_idx].reshape([num_samples, 1])

        # compute picp
        picp = prediction_interval_coverage_rate(y_target=y_target, y_lower=y_lower, y_upper=y_upper)
        interval_size = upper_quantile - lower_quantile

        picp_list.append(picp)
        error_list.append(np.abs(picp - interval_size))

    # mean over all intervals
    return picp_list, np.array(error_list).mean()


# MeanPredictionIntervalWidth (MPIW)
def mean_prediction_interval_width(y_full, y_lower, y_higher):
    # width of intervals
    y_range = y_full.max() - y_full.min()
    return np.abs(y_higher - y_lower).mean() / y_range


def mean_abs_calibration_error(y_target, y_quantile, quantile_list):
    # y_target (batch_size x 1)
    y_target = y_target.reshape(-1, 1)
    num_samples = y_target.shape[0]

    # y_quantile (batch_size x num_quantiles)
    y_quantile = y_quantile.reshape(num_samples, -1)
    num_quantiles = y_quantile.shape[1]
    assert num_quantiles == len(quantile_list)

    # compute coverage (num_quantiles)
    mean_calibration = (y_target <= y_quantile).mean(0)

    # compute error (mean over quantile-levels)
    return mean_calibration.tolist(), np.abs(mean_calibration - quantile_list).mean()


def root_mean_squared_calibration_error(y_target, y_quantile, quantile_list):
    # y_target (batch_size x 1)
    y_target = y_target.reshape(-1, 1)
    num_samples = y_target.shape[0]

    # y_quantile (batch_size x num_quantiles)
    y_quantile = y_quantile.reshape(num_samples, -1)
    num_quantiles = y_quantile.shape[1]
    assert num_quantiles == len(quantile_list)

    # compute coverage (num_quantiles)
    mean_calibration = (y_target <= y_quantile).mean(0)

    # compute error (mean over quantile-levels)
    return np.sqrt(np.mean(np.square(mean_calibration - quantile_list)))


def mean_interval_score(y_target, y_quantile, quantile_list):
    # assume quantile list is symmetry centered in 50%
    # for each quantile level (from 1% to 99%)
    interval_score_list = []
    num_quantiles = len(quantile_list)
    for i in range(num_quantiles // 2):
        # lower and upper quantile
        lower_quantile = quantile_list[i]
        upper_quantile = quantile_list[-(i + 1)]

        # get predicted lower and upper values
        y_lower = y_quantile[:, quantile_list == lower_quantile]
        y_upper = y_quantile[:, quantile_list == upper_quantile]

        # get mask below lower, above upper
        below_lower = (y_lower > y_target).astype('float')
        above_upper = (y_upper < y_target).astype('float')

        # compute score
        interval_score = (y_upper - y_lower)
        interval_score += (1.0 / lower_quantile) * (y_lower - y_target) * below_lower
        interval_score += (1.0 / lower_quantile) * (y_target - y_upper) * above_upper

        # mean over samples
        interval_score = interval_score.mean()
        interval_score_list.append(interval_score)

    # mean over all intervals
    return interval_score_list


def compute_quantile_results(prediction, target, quantile_list):
    check = pinball_loss_np(predict_data=prediction, target_data=target, quantiles=quantile_list)
    interval = mean_interval_score(y_target=target, y_quantile=prediction, quantile_list=quantile_list)
    results_value = check.tolist() + [np.mean(check)] + interval + [np.mean(interval)]
    return results_value


def compute_calibration_results(prediction, target, quantile_list):
    picp_list, mean_error = mean_prediction_interval_coverage_rate(y_target=target, y_quantile=prediction, quantile_list=quantile_list)
    calib_list, mace = mean_abs_calibration_error(y_target=target, y_quantile=prediction, quantile_list=quantile_list)
    results_value = picp_list + [mean_error] + calib_list + [mace]
    return results_value


def compute_mean_results(prediction, target):
    error_data = target - prediction
    results_value = [np.sqrt(np.mean(error_data * error_data)),
                     np.mean(error_data * error_data),
                     np.mean(np.abs(error_data))]
    return results_value

