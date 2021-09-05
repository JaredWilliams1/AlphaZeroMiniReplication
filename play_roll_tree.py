from game import Game
from roll_tree import Tree
import numpy as np
from roll_tree import Node
import pickle

def data_gen(number):
    tuple_array = []
    game_count = 0
    for y in range(number):
        state = Game().gameState
        states = []
        while not state.isEndGame:
            states.append(state)
            actions = state.allowedActions
            a = actions[np.random.randint(len(actions))]
            state, value, done = state.takeAction(a)
        rand_gamestate = states[np.random.randint(len(states))]
        tuple_array.append(tree_record(rand_gamestate))
        game_count += 1
        print("Game:", game_count)
    with open("data.pkl", "wb") as file:
        pickle.dump(tuple_array, file)

def tree_record(gamestate):
    game = Game()
    game.gameState = gamestate
    tree = Tree(root=Node(gamestate, index=0, parent=Node(game_state=Game().gameState, index=None, parent=None)))
    game_list = []
    move_list = []
    winner = 0
    while True:
        for y in range(100):
            tree.iteration()
        max_index = np.where(tree.root.children_visits == np.amax(tree.root.children_visits))[0][0]
        tree.root = tree.root.children[max_index]
        game.step(game.gameState.allowedActions[max_index])
        if game.gameState.isEndGame:
            if game.gameState.value[1] != 0:
                winner = 1
            break
        else:
            game_list.append(np.reshape(game.gameState.board, (6,7)))
            move_list_temp = []
            for numbers in game.gameState.allowedActions:
                new_game_state = game.gameState.takeAction(numbers)
                move_list_temp.append(Node.nn_predict(new_game_state))
            move_list.append(move_list_temp)

        max_index = np.where(tree.root.children_visits == np.amax(tree.root.children_visits))[0][0]
        tree.root = tree.root.children[max_index]
        game.step(game.gameState.allowedActions[max_index])
        if game.gameState.isEndGame:
            if game.gameState.value[1] != 0:
                winner = -1
            break
    full_tuple = (np.array(game_list), winner, np.array(move_list))
    return full_tuple

def person_v_tree():
    game = Game()
    tree = Tree()
    while not game.gameState._checkForEndGame():
        print("root", tree.root)
        for x in range(5000):
            tree.iteration()
        print("iterated")
        max_index = np.where(tree.root.children_visits == np.amax(tree.root.children_visits))[0][0]
        print("root children value", tree.root.children_value)
        print("root children visits", tree.root.children_visits)
        print("root choices:", tree.root.children)
        tree.root = tree.root.children[max_index]
        game.step(game.gameState.allowedActions[max_index])
        game.gameState.print_render()
        print("checkDorEndGame: ", game.gameState.isEndGame)
        print("getValue: ", game.gameState._getValue())
        print("getScore: ", game.gameState._getScore())
        if game.gameState._checkForEndGame():
            print("YOU LOSE!")
            break
        print(game.gameState.allowedActions)
        good_inputs = [x % 7 for x in game.gameState.allowedActions]
        print(good_inputs)

        inp = input("slot # of next move")
        while int(inp) not in good_inputs:
            inp = input("slot # of next move")
        step_index = good_inputs.index(int(inp))
        tree.root = tree.root.children[step_index]
        game.step(game.gameState.allowedActions[step_index])

        game.gameState.print_render()
        print("checkDorEndGame: ", game.gameState.isEndGame)
        print("getValue: ", game.gameState._getValue())
        print("getScore: ", game.gameState._getScore())
    print("GAME OVER!")