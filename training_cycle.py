import sys
import os
import numpy as np
import pickle
from res_net import Connect4Model
from data_generation import gen_batch
from tensorflow.keras.models import model_from_json
from game import Game
from nn_tree import Tree
from nn_tree import Node
import tensorflow as tf
from tensorflow import keras
import time


def main():

    # MAIN OPTIONS:
    # create_lineage("A")
    train_lineage("A", 6, 2)
    # print_lineage_report

    sys.exit()


def create_lineage(lineage_name):

    # creates lineage directory
    root_path = os.getcwd()
    lineage_path = root_path + "/lineage_%s" % lineage_name
    first_iter_path = lineage_path + "/iter_" + lineage_name + "0"

    # create process_data and timing_data saved numpy arrays
    process_data_path = lineage_path + "/processes_" + lineage_name + ".npy"
    timing_data_path = lineage_path + "/timing_" + lineage_name + ".npy"
    np.save(process_data_path, np.zeros(3, dtype=int), allow_pickle=False)
    np.save(timing_data_path, np.zeros(2), allow_pickle=False)

    # create challenge_game_results txt file
    open(lineage_path + "/challenge_game_results_" + lineage_name + ".txt", "w+")

    # create model.json and starting weights
    model = Connect4Model()
    model.build()
    open('model.json', 'w')
    model_path = lineage_path + "/model_" + lineage_name + ".json"
    starting_weights_path = first_iter_path + "/weights_" + lineage_name + "0.h5"
    model.save_model(model_path, starting_weights_path)

    return True


def create_iteration(index, lineage_name):

    root_path = os.getcwd()
    lineage_path = root_path + "/lineage_%s" % lineage_name
    iter_path = "%s/iter_%s%s" % (lineage_path, index, lineage_name)
    game_data_path = "%s/game_data_%s%s" % (iter_path, index, lineage_name)
    try:
        os.makedirs(game_data_path)
    except OSError:
        print("Creation of the directory %s failed" % iter_path)
    else:
        print("Successfully created the directory %s " % iter_path)

    return iter_path


def train_lineage(lineage_name, total_games, batches, does_challenge=False):
    games_in_batch = total_games // batches
    root_path = os.getcwd()
    lineage_path = root_path + "/lineage_%s" % lineage_name
    process_data_path = lineage_path + "/processes_" + lineage_name + ".npy"
    process_np = np.load(process_data_path, allow_pickle=False)
    iter_index = process_np[0]


    # load model
    json_file = open(lineage_path + "/model_" + lineage_name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    while True:

        curr_iteration_suffix = ("%s%s") % (lineage_name, iter_index)
        curr_iter_path = lineage_path + "/iter_" + curr_iteration_suffix
        curr_iter_game_data_path = curr_iter_path + "/game_data_%s" % curr_iteration_suffix
        curr_weights_path = curr_iter_path + "/weights_" + curr_iteration_suffix + ".h5"
        batch_index = len([name for name in os.listdir(curr_iter_game_data_path) if
                           os.path.isfile(os.path.join(curr_iter_game_data_path, name))])
        model.load_weights(curr_weights_path)

        while batch_index < batches:
            batch_save_path = "%s/game_%s_%s.pkl" % (curr_iter_game_data_path, curr_iteration_suffix, batch_index)
            gen_batch(model, batch_save_path, batch_size=games_in_batch)
            batch_index += 1

        next_iter_path = create_iteration(iter_index + 1, lineage_name)
        challenger_weights = train_challenger(model, batch_index, lineage_name, curr_iter_path, iter_index, curr_iter_game_data_path, next_iter_path)
        if does_challenge:
            model.load_weights(curr_weights_path)
            chall_model = model_from_json(loaded_model_json)
            chall_model.load_weights(challenger_weights)
            challenge_result = challenge(chall_model, model, 10)
        else:
            model.load_weights(challenger_weights)
            iter_index += 1
        print()
        print(challenge_result)
        print("test finished")
        sys.exit()


def train_challenger(model, batch_index, lineage_name, curr_iter_path, iter_index, data_path, next_iter_path):

    # format imported: list( tuples( array(boards), winner, array(predictions) ) ) )
    # array(boards): (-1, 6, 7, 3)
    # winner: 0 or 1
    # array(predictions): (-1, 1, 7)
    # format for training: array( array(board) )    array( list( array(predictions), winner ) )
    # board: numpy array (6, 7, 3)  predictions: numpy array (1, 7)
    # add documentation for mirrored data

    board_list = []
    prediction_list = []
    winner_list = []

    for index in range(0, batch_index):
        game_path = "%s/game_%s%s_%s.pkl" % (data_path, lineage_name, iter_index, index)
        pickle_off = open(game_path, "rb")
        unformatted_data = pickle.load(pickle_off)
        for game in unformatted_data:  # game is a tuple
            boards = np.swapaxes(np.swapaxes(game[0], 1, 2), 2, 3)
            for board in boards:  # boards is an array of boards
                board_list.append(board)
                board_list.append(np.flip(board, 1))
            for prediction in game[2]:  # game[2] is an array of predictions
                prediction_list.append(prediction[0])
                prediction_list.append(np.flip(prediction[0], 0))
            winner_list_temp = [game[1]] * game[2].shape[0] * 2
            winner_list = winner_list + winner_list_temp

    board_array = np.array(board_list)
    prediction_array = np.array(prediction_list)
    winner_array = np.array(winner_list)

    print(board_array.shape)
    print(prediction_array.shape)
    print(winner_array.shape)

    model.compile(optimizer=tf.keras.optimizers.Nadam(),
                  loss=['categorical_crossentropy', 'mean_squared_error'],
                  metrics=['accuracy'])

    #model.summary()


    print(prediction_array.shape)
    print(winner_array.shape)
    print(board_array.shape)


    # shuffle data
    seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    np.random.shuffle(prediction_array)
    np.random.seed(seed)
    np.random.shuffle(winner_array)
    np.random.seed(seed)
    np.random.shuffle(board_array)

    split_index = int(prediction_array.shape[0] / (5 / 4))

    prediction_array_train, prediction_array_val = prediction_array[:split_index, :], prediction_array[split_index:, :]
    winner_array_train, winner_array_val = winner_array[:split_index], winner_array[split_index:]
    board_array_train, board_array_val = board_array[:split_index, :, :, :], board_array[split_index:, :, :, :]
    #prediction_array = np.reshape(prediction_array, (-1, 7, 1))

    print()
    print(prediction_array_train.shape, prediction_array_val.shape)
    print(winner_array_train.shape, winner_array_val.shape)
    print(board_array_train.shape, board_array_val.shape)

    y_train = {"value_out1":winner_array_train, "policy_out1":prediction_array_train}
    y_val = {"value_out1":winner_array_val, "policy_out1":prediction_array_val}
    print(type(y_train))

    weights_path = "%s/weights_%s%s.h5" % (next_iter_path, lineage_name, iter_index+1)
    checkpoint_cb = keras.callbacks.ModelCheckpoint(weights_path, save_best_only=True)
    tb_path = os.path.join(curr_iter_path, "tensorboard")
    tensorboard_cb = keras.callbacks.TensorBoard(tb_path)

    model.fit(board_array_train, y_train, epochs=3, callbacks=[checkpoint_cb, tensorboard_cb], validation_data=(board_array_val, y_val))

    return weights_path


def challenge(chall_model, champ_model, game_num=400):

    # if first_model wins it returns 1
    # if second_model wins it returns 2
    def challenge_game(first_model, second_model):
        game = Game()
        rand_step_count = np.random.randint(5)
        step_count = 0
        while not game.gameState.isEndGame and not step_count == rand_step_count:
            actions = game.gameState.allowedActions
            a = actions[np.random.randint(len(actions))]
            game.step(a)
            step_count += 1
        second_tree = Tree(second_model, gamestate_start=game.gameState)
        first_tree = Tree(first_model, gamestate_start=game.gameState)
        winner = 0
        #print("starting:")
        #game.gameState.print_render()
        while True:
            for y in range(50):  #100
                first_tree.iteration()

            max_index = np.where(first_tree.root.children_visits == np.amax(first_tree.root.children_visits))[0][0]
            first_tree.root = first_tree.root.children[max_index]
            #print(max_index)
            if second_tree.root.children[max_index] is None:
                second_tree.root.children[max_index] = Node(second_tree.root.game_state.takeAction(second_tree.root.actions[max_index])[0], max_index, second_tree.root, second_tree.nn)
                second_tree.root = second_tree.root.children[max_index]
            else:
                second_tree.root = second_tree.root.children[max_index]

            #second_tree.root.game_state.print_render()

            if first_tree.root.game_state.isEndGame:
                if first_tree.root.game_state.value[1] != 0:
                    winner = 1
                break

            for y in range(50):  #100
                second_tree.iteration()

            max_index = np.where(second_tree.root.children_visits == np.amax(second_tree.root.children_visits))[0][0]
            second_tree.root = second_tree.root.children[max_index]
            #print(max_index)
            if first_tree.root.children[max_index] is None:
                first_tree.root.children[max_index] = Node(first_tree.root.game_state.takeAction(first_tree.root.actions[max_index])[0], max_index, first_tree.root, first_tree.nn)
                first_tree.root = first_tree.root.children[max_index]
            else:
                first_tree.root = first_tree.root.children[max_index]

            #second_tree.root.game_state.print_render()

            if first_tree.root.game_state.isEndGame:
                if first_tree.root.game_state.value[1] != 0:
                    winner = 2
                break
        first_tree.root.game_state.print_render()
        return winner

    results_array = np.zeros((2, 3), dtype=int)
    turn_switch = True
    #start = time.time()

    for i in range(game_num):
        print("GAME:", i)
        if turn_switch:
            result = challenge_game(chall_model, champ_model)
            results_array[0, result] += 1
        else:
            result = challenge_game(champ_model, chall_model)
            results_array[1, result] += 1
        print("turn_switch ", turn_switch)
        print("result ", result)
        print()
        turn_switch = not turn_switch
    return results_array


if __name__ == '__main__':
    main()
