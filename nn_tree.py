# -*- coding: utf-8 -*-
""" Author: Jared Williams

This program defines a Node, an object used to construct a tree, and
used by the Tree class.

"""
from math import sqrt
import numpy as np
from game import Game

class Tree:
    """a Tree obejct, created to map out possibilites of the
       game Connect4 and select the best move from the root.
        """

    def __init__(self, nn, gamestate_start=Game().gameState):
        """Inits a Tree with two associated Nodes.

        One Node is the root, and is initialized with a blank GameState.
        Another Node is initialized as the parent of the root, purely to
        store its visit and value count.

        Attributes:
            root: The root Node
            """
        self.nn = nn
        self.root = Node(gamestate_start, index=0, parent=Node(game_state=Game().gameState, index=None, parent=None, nn=self.nn), nn=self.nn)
        self.root.parent.children[0] = self.root

    def iteration(self):  # 0.65% residual time  |  90.99% of total time
        """Descends down the Tree of Nodes by calling the select function, only stopping
           when and EndState has been selected or a new Node has been created.

           if a new Node was created, that Node does a random rollout, and the visited
           Nodes update their visits and values based on the outcome of that rollout.

           If an EndState was reached, the visited Nodes update their visits and values
           based on the outcome of that EndState.
            """
        visited = [self.root]   # keeps track of what has been visited
        current = self.root  # starts at the root
        end = False
        while not end:
            current, end_val, end = current.select()  # cont: 90.34% of total time
            visited.append(current)
        if end_val is None:
            value_addition = current.nn_prediction[1][0]
        else:
            if end_val is 0:
                value_addition = -1
            else:
                value_addition = 1
        for visits_array in reversed(visited):
            visits_array.value += value_addition
            visits_array.visits += 1
            value_addition = -value_addition



class Node:
    """a Node obejct, created to store in a Tree.

    One Node represents one possible state of the game
    Connect4, to be used in a tree.
        """

    def __init__(self, game_state, index, parent, nn):  # 0.87% residual time  |  85.98% of total time
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
        self.children = np.empty(7, dtype=Node)
        self.children_value = np.full(7, -float('inf'))
        mod_indices = [elem % 7 for elem in self.actions]
        self.children_value[mod_indices] = 0
        if len(self.actions) != 0:
            mod_indices, self.actions = zip(*sorted(zip(mod_indices, self.actions)))
        self.actions = list(self.actions)
        for i in np.where(self.children_value == -float('inf'))[0]:
            self.actions.insert(i, -1)
        self.nn = nn
        final_one = np.reshape((game_state.board == 1).astype(int), (6, 7))
        final_two = np.reshape((game_state.board == -1).astype(int), (6, 7))
        if game_state.playerTurn == 1:
            final_three = np.ones((6, 7))
        else:
            final_three = np.zeros((6, 7))
        self.proper_board = np.array([[final_one, final_two, final_three]])
        self.nn_prediction = nn.predict(np.swapaxes(np.swapaxes(self.proper_board, 1, 2), 2, 3))  # cont: 85.11% of total time
        #self.nn_prediction = nn.predict(np.reshape(self.proper_board, (1, 6, 7, 3)))  # cont: 85.11% of total time
        self.children_visits = np.zeros(7)
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

    def select(self):  # 4.36% residual time  |  90.34% of total time
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
        end_val = None
        end = False
        qs = self.children_value / self.children_visits
        qs[np.isnan(qs)] = 0
        qs[self.children_value == -float('inf')] = -float('inf')
        top = sqrt(np.sum(self.children_visits))
        uct_values = qs + self.nn_prediction[0][0] * top / (1.0 + self.children_visits)
        selected_index = np.argmax(uct_values)
        if self.children[selected_index] is None:
            action = self.actions[selected_index]
            new_game_state = self.game_state.takeAction(action)[0]
            self.children[selected_index] = Node(new_game_state, selected_index, self, self.nn)  # cont: 85.98% of total time
            end = True
        if self.children[selected_index].game_state.isEndGame:
            end_val = self.children[selected_index].game_state.value[2]
            end = True
        return self.children[selected_index], end_val, end
