# GRADED FUNCTION: schedule_lr_decay

def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):
    """
    Calculates updated the learning rate using exponential weight decay.

    Arguments:
    learning_rate0 -- Original learning rate. Scalar
    epoch_num -- Epoch number. Integer.
    decay_rate -- Decay rate. Scalar.
    time_interval -- Number of epochs where you update the learning rate.

    Returns:
    learning_rate -- Updated learning rate. Scalar
    """
    # (approx. 1 lines)
    # learning_rate = ...
    # YOUR CODE STARTS HERE
    learning_rate = (1/(1 + decay_rate*(epoch_num/time_interval)))*learning_rate0

    # YOUR CODE ENDS HERE
    return learning_rate

def update_lr(learning_rate0, epoch_num,decay_rate):
    '''This method is good for small amount of data since the learning rate will quickly go down to zero'''
    learning_rate = (1/(1+decay_rate*epoch_num))*learning_rate0
    return learning_rate