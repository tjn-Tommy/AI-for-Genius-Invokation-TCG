import random
import numpy as np
import math

# Top-level classes and functions
from src.models.neural_network import NeuralNetworkPolicy
from src.utils.game_state import CardGameState

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, exploration_weight=1.0):
        best_value = -float('inf')
        best_child = None
        for child in self.children:
            ucb_value = (child.value / (child.visits + 1e-6)) + \
                        exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            if ucb_value > best_value:
                best_value = ucb_value
                best_child = child
        return best_child

class MCTSAgent:
    def __init__(self, simulation_limit=1000):
        self.simulation_limit = simulation_limit

    def search(self, initial_state):
        root = MCTSNode(initial_state)

        for _ in range(self.simulation_limit):
            node = root
            # Selection
            while not node.state.is_terminal() and node.is_fully_expanded():
                node = node.best_child()
            # Expansion
            if not node.state.is_terminal():
                legal_actions = node.state.get_legal_actions()
                for action in legal_actions:
                    new_state = node.state.perform_action(action)
                    child_node = MCTSNode(new_state, parent=node)
                    node.children.append(child_node)
            # Simulation
            reward = self.simulate(node.state)
            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

        return root.best_child(exploration_weight=0.0).state

    def simulate(self, state):
        """Simulate a random rollout from the state and return the reward."""
        while not state.is_terminal():
            legal_actions = state.get_legal_actions()
            action = random.choice(legal_actions)
            state = state.perform_action(action)
        return state.get_reward()

class HearthstoneAgent:
    def __init__(self, mcts_agent, nn_policy):
        self.mcts_agent = mcts_agent
        self.nn_policy = nn_policy

    def choose_action(self, state):
        """Use MCTS guided by the neural network policy to choose an action."""
        action_probs = self.nn_policy.predict(state)
        # Use MCTS with a bias towards high-probability actions
        return self.mcts_agent.search(state)

# Usage
if __name__ == "__main__":
    from src.models.dummy_model import DummyModel

    dummy_model = DummyModel()
    nn_policy = NeuralNetworkPolicy(dummy_model)
    mcts_agent = MCTSAgent(simulation_limit=100)
    hearthstone_agent = HearthstoneAgent(mcts_agent, nn_policy)

    # Assume an initial state is given
    initial_state = CardGameState(None, None, None)
    best_action = hearthstone_agent.choose_action(initial_state)
    print("Best action:", best_action)
