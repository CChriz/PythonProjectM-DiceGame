# Dice Game Agent

## Approach

### Initial Approach

The initial approach involved creating a 'decide-as-you-go' manual agent that employed a basic Markov Decision Process (MDP) model. This agent considered only the current state of the dice and aimed to find the best average reward for every possible action, ultimately selecting the action with the highest expected value. This agent outperformed basic AlwaysHold and Perfectionist agents but had issues with inconsistency and was computationally slow due to recalculating average rewards each turn.

### Final Approach: Value Iteration

To address the limitations of the initial approach, the final agent was implemented using a value iteration method. The agent class included several methods to determine the optimal action for each possible state before playing the game.

#### `perform_single_value_iteration` Method

This method performed value iteration once on all possible states of the current game. It updated the value function `V(s)` and returned the new delta value. The algorithm followed was as follows:


Loop for each s in S:<br>
	v <- V(s)<br>
	V(s) <- max_a SUM _s',_r p(s',r|s,a)[r + gamma * V(s')]<br>
	delta <- max(delta, |v - V(s)|)


#### `value_iteration` Method

The `value_iteration` method repeatedly called `perform_single_value_iteration` until the delta value was less than the predefined theta value. The algorithm was as follows:


Loop:
	delta <- 0
	perform_single_value_iteration()
until delta < theta


#### `get_next_states` Method

To efficiently retrieve next possible states, rewards, and probabilities, the `get_next_states` method was implemented. It overrode the `get_next_states` method from the game. This method tracked state-action combinations already calculated in a dictionary to avoid redundant calculations.

## Testing and Performance

To assess the agent's performance in a standard game (3x6 sided dice), the `extended_tests()` function in the test script was used with the following settings:

- `n = 100,000`
- `game = DiceGame(dice=3, sides=6)`

The agent's performance was tested with varying values of theta and gamma. Here are some results:

### Varying Theta Values

- `theta = 0.005, gamma = 0.9`: Average score: 12.83542, Average time: 0.00016
- `theta = 0.01, gamma = 0.9`: Average score: 12.82066, Average time: 0.00004
- `theta = 0.1, gamma = 0.9`: Average score: 12.81474, Average time: 0.00013

The agent's performance did not show significant changes with varying theta values, so the initial default value of 0.01 was retained.

### Varying Gamma Values

- `theta = 0.01, gamma = 0.95`: Average score: 13.30923, Average time: 0.00023
- `theta = 0.01, gamma = 0.9`: Average score: 12.82066, Average time: 0.00004
- `theta = 0.01, gamma = 0.8`: Average score: 11.79933, Average time: 0.00011

Notably, the agent's average score and time decreased as gamma values were reduced. A gamma value of 0.96 yielded the highest average score but took significantly longer. After finding a balance between score and time, a gamma value of 0.958 was chosen.

## Conclusion

The Dice Game Agent implemented using value iteration and optimized theta and gamma values achieved a high average score of 13.36055 with an average time of 0.00014. This combination balanced performance and efficiency, making it a suitable choice for playing the game.
