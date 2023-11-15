"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    Reduced the noise so that it would be unlikely for the agent to get the negative rewards when
    moving left and right, so it will not want to end the game quickly by moving left.
    """

    answerDiscount = 0.9
    answerNoise = 0.0001

    return answerDiscount, answerNoise

def question3a():
    """
    Noise is 0 so that the agent can go left and right near the cliff without risking negative
    payoff. Small discount so that the agent will prefer a reward that is nearer.
    """

    answerDiscount = 0.3
    answerNoise = 0.0
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    Small discount so that the agent will prefer a reward that is nearer. Noise is 0.2 so that
    the agent will avoid going left and right near the cliff.
    """

    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    Large discount so that the agent will prefer a larger reward that is further away. Noise is 0 so
    that the agent risks the cliff.
    """

    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    Large discount so that the agent will prefer a larger reward that is further away. Noise is 0.4
    so that the agent does not risk the cliff.
    """

    answerDiscount = 0.9
    answerNoise = 0.4
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    Noise is 0.4 so that the agent does not risk the cliff. Living reward is 1.0 so that the agent
    wants to continue living and not exit.
    """

    answerDiscount = 0.9
    answerNoise = 0.4
    answerLivingReward = 2.0

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    There are too few episodes to find the optimal policy.
    """
    
    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
