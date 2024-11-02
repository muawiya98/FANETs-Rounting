import tensorflow as tf


def normalized_actions(actions):
    """

    """
    actions = actions[0].numpy()
    neg_actions = actions[actions < 0]

    sum_neg_actions = sum(neg_actions)
    add_amount = sum_neg_actions/len(actions[actions >= 0])
    actions[actions >= 0] += add_amount
    actions[actions < 0] = 0
    actions = actions/sum(actions)
    return tf.convert_to_tensor(actions, dtype=tf.float32)