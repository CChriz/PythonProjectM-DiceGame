import time
from collections import defaultdict
from dice_game import DiceGame

class MyAgent(DiceGameAgent):
    """an Agent that determines the optimal policy using value iteration"""
    def __init__(self, game):
        super().__init__(game)
        # initialise constants: theta & gamma for the value iteration
        # process of choosing parameters detailed in readme.txt
        self.theta = 0.01
        self.gamma = 0.958
        # tuple store for the 'hold-all dices' action of the game (last action in game.actions array so index of [-1])
        self.hold_all = self.game.actions[-1]
        # dictionary for the value of each possible state of the game - V(s)
        self.values = {}
        # dictionary for the optimal action of each possible state of the game
        self.optimal_actions = {}
        # dictionary for storing results of get next states function
        # as for a set of given action & state in a game, the outcome of the function will be the same
        # (avoid re-calling get_next_states method from game if result has been calculated before)
        self.next_states_data = {}

        # loop cycles through each possible state, initialising the starting V(s) of each state to 0 and the
        # optimal policy for each possible state to an empty tuple (), so that they can be updated later on
        for state in game.states:
            # set initial value of state all set to 0
            # "initialise V(s), for all s in S arbitrarily except that V(terminal) = 0"
            self.values[state] = 0
            # set initial optimal action of state set to empty tuple (re-roll all)
            self.optimal_actions[state] = ()

        # begin value iteration
        self.value_iteration()

    def perform_single_value_iteration(self):
        """calculates and returns delta (measure of convergence), updates V(s) of each possible state and the
        optimal action to be taken at each of those possible states - a single time"""
        # set delta to 0 at the start of value iteration
        # "delta <- 0"
        delta = 0

        # first loop cycles through each possible state in the current game
        # "Loop for each s in S:"
        for state in self.game.states:
            # retrieve the previous value of the current possible state
            # this is used to calculate delta at the end of current iteration
            # "v <- V(s)"
            value = self.values[state]

            # initialise dictionary for values yielded from each possible action from the current possible state {value:action}
            action_from_value = {}

            # "V(s) = MAX_action ( SUM_nextState,reward ( p( nextState,reward | currentState,action ) * [reward + gamma * V(nextState)] ) )"
            # V(terminal) = 0 : game over
            # first calculating the sum of future rewards for each action
            # "( SUM_nextState,reward ( p( nextState,reward | currentState,action ) * [reward + gamma * V(nextState)] ) )"
            # second loop cycles through each possible actions in the current game
            for action in self.game.actions:
                # initialise the sum of next states' rewards to 0 for current possible action
                # retrieve the list of all possible next states & their corresponding probabilities for the current possible action
                # reward = -1, for all actions except terminal state (holding all dices)
                next_states, game_over, next_reward, next_probs = self.get_next_states(action, state)

                # "SUM_nextState,reward ( p( nextState,reward | currentState,action ) * [reward + gamma * V(nextState)] )"
                # if the possible action is holding-all dices (terminal)
                # then there is only one possible next state (the current state)
                # so  p( nextState,reward | currentState,action ) = 1
                # and  next state = current state (terminal), next value = final value
                # V(nextState) = finalValue
                # therefore  "SUM_nextState,reward = ( 1 * [reward + gamma * finalValue] )"
                if action == self.hold_all:
                    next_value = self.game.final_score(state)
                    sum_nextState_reward = next_reward + (self.gamma * next_value)

                # if possible action is not holding-all dices (not terminal)
                else:
                    sum_nextState_reward = 0
                    # third loop cycles through each possible next states & their corresponding possibilities for the current possible action
                    for next_state, next_prob in zip(next_states, next_probs):
                        next_value = self.values[next_state]
                        # "SUM_nextState,reward ( p( nextState,reward | currentState,action ) * [reward + gamma * V(nextState)] )"
                        sum_nextState_reward += next_prob * (next_reward + (self.gamma * next_value))

                # add current sum to dictionary along with its corresponding action
                # {value:action}
                action_from_value[sum_nextState_reward] = action
                
            # find the largest value from the dictionary
            best_value = max(action_from_value)
            # retrieve the action that yields the largest value and update optimal actions for the current possible state with that action
            self.optimal_actions[state] = action_from_value[best_value]
            # update V(s) of the current possible state to that of the best value yielded
            # " V(s) = MAX_action SUM ... "
            self.values[state] = best_value
            # update delta after calculating for current possible state
            # " delta <- max(delta, |v - V(s)|) "
            # square then square root for absolute value of ( v - V(s) )
            delta = max(delta, ((value - self.values[state]) ** 2) ** 0.5)

        # return updated delta
        return delta

    def value_iteration(self, delta=100):
        """continues value iteration (calls perform_single_value_iteration) as long as delta > theta (not yet converged)"""
        # delta set to 100 at the start so value iteration can begin, as delta > theta
        # if (delta > theta) - not yet converged, continue to perform next value iteration
        # " Loop...Until delta < theta "
        while delta > self.theta:
            delta = self.perform_single_value_iteration()
        # loop ends when delta < theta, converged

    def get_next_states(self, action, state):
        """returns results - the possible next states, if game over, reward, probabilities of the possible next states
        given a possible action and current state"""
        # overrides get_next_states method from game
        # first attempt to find if the action & state has been calculated before
        try:
            results = self.next_states_data[(action, state)]
        # upon key error - the results of the action & state combination has not been calculated before
        except KeyError:
            # update dictionary of next states data with the results
            results = self.game.get_next_states(action, state)
            self.next_states_data[(action, state)] = results
        # return the results
        return results
        
    def play(self, state):
        """returns a corresponding action for the given state - tuple"""
        # returns an action from the dictionary of optimal actions given the current state
        return self.optimal_actions[state]
    
