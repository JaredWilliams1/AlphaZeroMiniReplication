# -*- coding: utf-8 -*-
""" Author: Jared Williams

This program defines a Node, an object used to construct a tree, and
used by the Tree class.

"""
from math import sqrt
from math import log
from random import random
import numpy as np
from game import Game


class Node:
    """a Node obejct, created to store in a Tree.

    One Node represents one possible state of the game
    Connect4, to be used in a tree.
        """
    EPSILON = 1e-6
    def __init__(self, game_state, index, parent):
        """Inits a Node with a GameState.

        One Node is initialized with a unique GameState, and it doesn't
        change.

        Attributes:
            game_state: A GameState is a representation of the game board
                at a current point in the game, independent from the previous
                moves
            index: The location of this Node in its parent's children lists
            parent: The pointer of this Node's parent Node
            children: A list that stores the children Nodes
            children_value: A Numpy array with the values of it's children Nodes
            children_visits: The count of how many times the Node has been visited.
            """
        self.game_state = game_state
        self.index = index
        self.parent = parent
        self.actions = game_state.allowedActions
        self.children = np.empty(len(self.actions), dtype=Node)
        self.children_value = np.zeros(len(self.actions))
        self.children_visits = np.zeros(len(self.actions))
        self.epsilon = 1e-6


    @property
    def visits(self):
        return self.parent.children_visits[self.index]

    @visits.setter
    def visits(self, value):
        self.parent.children_visits[self.index] = value

    @property
    def value(self):
        return self.parent.children_value[self.index]

    @value.setter
    def value(self, value):
        self.parent.children_value[self.index] = value


    def update_stats(self, win_state):
        """Updates the value and visits attributes, called by each visited Node
           during backpropagation, if the game does not end in a tie.

           The visits attribute is increased by one

           If the player won the game, value will be 1, and the value of the Node
           will be increased by one.

           If the player lost the game, the value will not change.

           Tie games will not make it to this code.

            Args:
                win_state: represents the win status of the player
                    1: the player won, value added
                    -1: the player lost, no value added
            """
        if win_state is not -1:
            self.value += 1
        self.visits += 1

    def select(self):
        """Looks at the possible moves from the current Node, and selects one based on
        the Upper Confidence Bound equation, which tries to balance exploration and exploitation,
         based on visits and value of the children Nodes, and the visits of the current Node.

        the children Nodes have visit and value counts even before they are created, so not all
        the children Nodes are necessarily created yet. If the selected GameState's Node has not
        been created, it is created in the Node's children array.

        Other than in the initialization of the Tree, the select function is the only place where
        Nodes are created.

        If the selected Node's GameState is in EndGame, select returns the GameState's value,
        0 if the Game has ended in a tie, and 1 otherwise.

        If a Node is created, or an EndState has been reached, end returns True, indicating to the
        Tree to stop descending.

            Returns:
                The selected child Node, EndGame int value, and the continuation boolean
            """
        best_value = - float("inf")
        selected_index = 0
        end_val = None
        end = False
        for c in range(len(self.children)):
            uct_value = (self.children_value[c] /
                         (self.children_visits[c] + self.epsilon) +
                         sqrt(log(self.visits+1) / (self.children_visits[c] + self.epsilon)) +
                         random() * self.epsilon)
            if uct_value > best_value:
                selected_index = c
                best_value = uct_value
        if self.children[selected_index] is None:
            action = self.game_state.allowedActions[selected_index]
            new_game_state = self.game_state.takeAction(action)[0]
            self.children[selected_index] = Node(new_game_state, selected_index, self)
            end = True
        if self.children[selected_index].game_state.isEndGame:
            end_val = self.children[selected_index].game_state.value[2]
            end = True
        return self.children[selected_index], end_val, end



    def rollout(self):
        """Randomly descends down GameStates without creating the associated
           Nodes, stopping when a GameState is in EndGame.

           Then returns an int representation of who won.

            Returns:
                an int representation of the winner:
                    1: X won
                    -1: O won
                    0: Tie
            """
        state = self.game_state
        while not state.isEndGame:
            actions = state.allowedActions
            a = actions[np.random.randint(len(actions))]
            state, value, done = state.takeAction(a)
        return state.value[2] * state.playerTurn

class Tree:
    """a Tree obejct, created to map out possibilites of the
       game Connect4 and select the best move from the root.
        """

    def __init__(self, root=Node(Game().gameState, index=0,
                                 parent=Node(game_state=Game().gameState, index=None, parent=None))):
        """Inits a Tree with two associated Nodes.

        One Node is the root, and is initialized with a blank GameState.
        Another Node is initialized as the parent of the root, purely to
        store its visit and value count.

        Attributes:
            root: The root Node
            """
        self.root = root
        self.root.parent.children[0] = root

    def iteration(self):
        """Descends down the Tree of Nodes by calling the select function, only stopping
           when and EndState has been selected or a new Node has been created.

           if a new Node was created, that Node does a random rollout, and the visited
           Nodes update their visits and values based on the outcome of that rollout.

           If an EndState was reached, the visited Nodes update their visits and values
           based on the outcome of that EndState.
            """
        visited = [self.root]  # keeps track of what has been visited
        current = self.root  # starts at the root
        end = False
        while not end:
            current, end_val, end = current.select()
            visited.append(current)
        if end_val is None:
            rollout = current.rollout()
            matching = rollout == current.game_state.playerTurn
            tie = rollout == 0
        else:
            tie = end_val == 0
            matching = True
        if tie:
            value_addition = 0
        elif matching:
            value_addition = 1
        else:
            value_addition = -1
        for visits_array in reversed(visited):
            visits_array.update_stats(value_addition)
            value_addition = -value_addition



