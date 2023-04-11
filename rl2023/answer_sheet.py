
############################################################################################################
##########################            RL2023 Assignment Answer Sheet              ##########################
############################################################################################################

# **PROVIDE YOUR ANSWERS TO THE ASSIGNMENT QUESTIONS IN THE FUNCTIONS BELOW.**

############################################################################################################
# Question 2
############################################################################################################

def question2_1() -> str:
    """
    (Multiple choice question):
    For the Q-learning algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_2() -> str:
    """
    (Multiple choice question):
    For the First-visit Monte Carlo algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_3() -> str:
    """
    (Multiple choice question):
    Between the two algorithms (Q-Learning and First-Visit MC), whose average evaluation return is impacted by gamma in
    a greater way?
    a) Q-Learning
    b) First-Visit Monte Carlo
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_4() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) as to why the value of gamma affects more the evaluation returns achieved
    by [Q-learning / First-Visit Monte Carlo] when compared to the other algorithm.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "Gamma is a crucial factor in reinforcement learning that balances immediate and future rewards. QL uses optimal value function estimates to update current estimates and higher gamma values result in more accurate long term value estimates. First Visit MC learns from complete episodes and actual rewards so high gamma values have minor impact. Inappropriate gamma values can significantly affect QL by overestimating or underestimating the true value function thus affecting the bias-variance trade off in QL. Hence, gamma has more impact on QL than on MC."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 3
############################################################################################################

def question3_1() -> str:
    """
    (Multiple choice question):
    In Reinforce, which learning rate achieves the highest mean returns at the end of training?
    a) 6e-1
    b) 6e-2
    c) 6e-3
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_2() -> str:
    """
    (Multiple choice question):
    When training DQN using a linear decay strategy for epsilon, which exploration fraction achieves the highest mean
    returns at the end of training?
    a) 0.75
    b) 0.25
    c) 0.01
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_3() -> str:
    """
    (Multiple choice question):
    When training DQN using an exponential decay strategy for epsilon, which epsilon decay achieves the highest
    mean returns at the end of training?
    a) 1.0
    b) 0.75
    c) 0.001
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_4() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of training when employing an exponential decay strategy
    with epsilon decay set to 1.0?
    a) 0.0
    b) 1.0
    c) epsilon_min
    d) approximately 0.0057
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_5() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of  training when employing an exponential decay strategy
    with epsilon decay set to 0.990?
    a) 0.990
    b) 1.0
    c) epsilon_min
    d) approximately 0.0014
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_6() -> str:
    """
    (Short answer question):
    Based on your answer to question3_5(), briefly  explain why a decay strategy based on an exploration fraction
    parameter (such as in the linear decay strategy you implemented) may be more generally applicable across
    different environments  than a decay strategy based on a decay rate parameter (such as in the exponential decay
    strategy you implemented).
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "A decay strategy that uses an exploration fraction parameter may be more suitable for different environments because it allows for more direct control of the exploration rate. On the other hand, the exponential decay strategy depends on the maximum number of time-steps may reduce slowly resulting in a higher exploration rate for longer than intended and may not even reach the minimum exploration rate desired. In contrast, the linear decay strategy allows for a predictable and gradual reduction in exploration rate with each episode resulting in a smooth transition from exploration to exploitation."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


def question3_7() -> str:
    """
    (Short answer question):
    In DQN, explain why the loss is not behaving as in typical supervised learning approaches
    (where we usually see a fairly steady decrease of the loss throughout training)
    return: answer (str): your answer as a string (150 words max)
    """
    answer = "In supervised learning the targets are fixed and independent of the model's predictions while in DQN the targets are also a function of the model's own predictions. This creates a feedback loop between the models predictions and the targets which can result in instability and oscillations in the loss. The loss function compares the predicted and observed Q-values which can vary as the agent learns from past experiences stored in the replay buffer that may not be relevant to the current state. As the model learns from interacting with its environment the target Q-values change making accurate predictions difficult. Q-values of neighbouring states are updated using bootstrapping which are prone to large fluctuations during training. Q-Learning is an implementation of TD learning and its estimates are biased due to maximisation bias. Consequently the loss may not decrease steadily during training as the model struggles to keep up with moving targets."  # TYPE YOUR ANSWER HERE (150 words max)
    return answer


def question3_8() -> str:
    """
    (Short answer question):
    Provide an explanation for the spikes which can be observed at regular intervals throughout
    the DQN training process.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "The spikes observed during DQN training occur due to infrequent hard updates of the target network DQN uses two networks critics and target to ensure stable training. The target network generates Q-values, which are used to train the online network. However, if the target network is not updated often enough with the weights of the online network the Q-values used for training can become outdated leading to unstable training and spikes in the loss. These spikes are more noticeable in DQN due to batch learning where the model is trained on batches of experiences rather than after each experience."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 5
############################################################################################################

def question5_1() -> str:
    """
    (Short answer question):
    Provide a short description (200 words max) describing your hyperparameter turning and scheduling process to get
    the best performance of your agents
    return: answer (str): your answer as a string (200 words max)
    """
    answer = "5 configurations of the network hidden size from Q4 [64, 128], [128, 128], [256, 128], [512, 256], [512, 512] provided the highest mean results. A log-uniform distribution ranging from 2e-4 to 3e-3 was applied for learning rates, ranging from 2e-4 to 3e-3. Higher learning rates converge faster but can be unstable while lower learning rates are more stable but converge slower. Higher critic learning rates improves accuracy and stability while higher policy learning rates leads to quicker learning towards optimal policies. Tau was selected with a uniform distribution from 0.1 to 0.4 balancing the trade-off between more aggressive updates and potential instability or divergence. Batch size 128 is widely used in deep learning and was found to stabilise training. The gamma value of 0.99 was retained. A buffer capacity of 1e7 was picked to store a large number of experiences for training. To promote smooth and gradual transitions and prevent abrupt changes during training a cosine annealing schedule was employed to adjust the learning rates tau and noise standard deviation based on the current and maximum timesteps of the training loop. Hyperparameters were selected using grid and random search based on their expected impact on convergence speed and performance."  # TYPE YOUR ANSWER HERE (200 words max)
    return answer