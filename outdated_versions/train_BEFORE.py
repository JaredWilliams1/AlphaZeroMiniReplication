import sys
import os
import time
import numpy as np
import pickle
from res_net import Connect4Model
from play_nn_tree import data_gen
from keras.models import model_from_json
from game import Game
from nn_tree import Tree
from nn_tree import Node
import tensorflow as tf
from tensorflow import keras
import time


def main():
    #    TESTING
    #data_gen(10)
    #test_individual_prediction()
    make_new_nn()
    #data_gen(50)
    #train_challenger()
    #challenge()
    #test_individual_prediction()
    #model = Connect4Model()
    #model.build()
    #model.predict()

def training_cycle():
    while True:
        data_gen(10)
        train_challenger()
        challenge()

def test_mass_predict():
    path = "/Users/jaredwilliams/Documents/AI/Reinforcement/connect4/"
    pickle_off = open(path + "data.pkl", "rb")
    array = pickle.load(pickle_off)
    board = array[0][0]
    print(board.shape)
    model = Connect4Model()
    model.build()
    board = np.reshape(board, (-1, 6, 7, 1))
    print(board.shape)
    prediction = model.model.predict(board, batch_size=9)
    print(len(prediction))
    print(len(prediction[0]))
    print(prediction[0])


def test_individual_prediction():
    list = []
    path = "/Users/jaredwilliams/Documents/AI/Reinforcement/connect4/"
    pickle_off = open(path + "data.pkl", "rb")
    array = pickle.load(pickle_off)
    board = array[0][0][0]
    print(board.shape)
    board = np.reshape(board, (-1, 6, 7, 1))
    model = Connect4Model()
    model.build()
    prediction = model.model.predict(board)
    print(len(prediction))
    print(prediction[0])
    print()
    print(prediction[0].shape)
    print(prediction[1])

def make_new_nn():
    model = Connect4Model()
    model.build()
    #open('model.json', 'w')
    #model.save_model("model.json", "clean_slate.h5")
    #model.save_model("model.json", "champion.h5")


def train_challenger(data_path="data3.pkl"):

    path = "/Users/jaredwilliams/Documents/AI/Reinforcement/connect4/"
    pickle_off = open(path + data_path, "rb")
    unformatted_data = pickle.load(pickle_off)

    # format imported: list( tuples( array(boards), winner, array(predictions) ) ) )
    # format for training: array( array(board) )    array( list( array(predictions), winner ) )
    # board: numpy array (6,7)  predictions: numpy array (1, 7)
    board_list = []
    prediction_list = []
    winner_list = []

    for game in unformatted_data:  # game is a tuple
        boards = np.reshape(game[0], (-1, 6, 7, 3))
        for board in boards:  # boards is an array of boards
            board_list.append(board)
        for prediction in game[2]:  # game[2] is an array of predictions
            prediction_list.append(prediction[0])
        winner_list_temp = [game[1]] * game[2].shape[0]
        winner_list = winner_list + winner_list_temp

    board_array = np.array(board_list)
    prediction_array = np.array(prediction_list)
    winner_array = np.array(winner_list)

    print(board_array.shape)
    print(prediction_array.shape)
    print(winner_array.shape)

    json_file = open("model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("champion.h5")

    #tb_callback = keras.callbacks.TensorBoard(log_dir="./logs",
                                              #histogram_freq=1, write_graph=True, write_images=True)

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=['categorical_crossentropy', 'mean_squared_error'],
                  metrics=['accuracy'])

    model.summary()


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


    #sys.exit()
    checkpoint_cb = keras.callbacks.ModelCheckpoint("challenger.h5", save_best_only=True)
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    root_logdir = os.path.join(os.curdir, "my_logs")
    run_logdir = os.path.join(root_logdir, run_id)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    model.fit(board_array_train, y_train, epochs=5, callbacks=[checkpoint_cb, tensorboard_cb], validation_data=(board_array_val, y_val))


    #model = keras.models.load_model("checkpoint.h5")
    #model.save_weights("challenger.h5")


def challenge():

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
                    winner = -1
                break
        first_tree.root.game_state.print_render()
        return winner

    json_file = open("model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    champ_model = model_from_json(loaded_model_json)
    chall_model = model_from_json(loaded_model_json)
    champ_model.load_weights("champion.h5")   # negative score when ahead
    chall_model.load_weights("challenger.h5")   # positive score when ahead
    game_num = 400
    winning_balance = 0
    turn_switch = True
    start = time.time()
    for i in range(game_num):
        print("GAME:", i)
        if turn_switch:
            winning_balance += challenge_game(chall_model, champ_model)
        else:
            winning_balance -= challenge_game(champ_model, chall_model)
        print(winning_balance)
        print(time.time() - start)
        print()
        turn_switch = not turn_switch
    print("WINNING BALANCE:", winning_balance)
    if winning_balance >= game_num * .05: # if the challenger is 5% better
        # replace the champion
        #os.remove("model.h5")
        #os.rename("challenger.h5", "champion.h5")
        return True

    else:
        # delete the challenger
        #os.remove("challenger.h5")
        return False




if __name__ == '__main__':
    main()
