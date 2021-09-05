from nn_tree import Tree
from game import Game
import numpy as np
from res_net import Connect4Model
import pickle
from res_net import Connect4Model
from data_generation import gen_batch
from data_generation import tree_record
from tensorflow.keras.models import model_from_json
from game import Game
from nn_tree import Tree
from nn_tree import Node
import tensorflow as tf
from tensorflow import keras
import sys


def main():  # 96.22% of total time
    """
    #x = np.random.choice(np.arange(1, 7), p=[0.1, 0.05, 0.05, 0.2, 0.4, 0.2])
    #print (x)
    dist = np.array([0.1390761, 0.14652108, 0.1436286,  0.13272178, 0.15212338, 0.14448448, 0.14144462])
    dirich_dist = np.random.dirichlet(dist, 0.03)

    epsilon = 0.25
    prior_prob = (1-epsilon) * dist + epsilon * dirich_dist
    print(prior_prob)
    #prior_prob = self.nn_prediction[0][0]
    #print()
    """



    #build_new_model()
    #data_gen(3)
    json_file = open('testing_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("testing_weights.h5")
    #new_tree = Tree(model)
    #print(len(new_tree.root.nn_prediction))
    #print(new_tree.root.nn_prediction[0])
    #print(new_tree.root.nn_prediction[1])
    #print(new_tree.root.nn_prediction[0].shape)
    #print(new_tree.root.nn_prediction[1].shape)
    #new_tree.iteration()
    #new_tree.nn.compile(optimizer=tf.keras.optimizers.Nadam(),
                  #loss=['kl_divergence', 'mean_squared_error'],
                  #metrics=['accuracy'])

    #model.compile()
    model.compile(optimizer=tf.keras.optimizers.Nadam(),
              loss=model_loss,
              metrics=['kullback_leibler_divergence', 'mean_squared_error'])
    model.summary()
    #sys.exit()
    pickle_off = open("testing_data.pkl", "rb")
    unformatted_data = pickle.load(pickle_off)

    board_list = []
    prediction_list = []
    winner_list = []

    print("len(unformatted_data)", len(unformatted_data))

    for game in unformatted_data:  # game is a tuple
        #print("len(game)", len(game))
        #print(game[0].shape)
        #print(game[1])
        #print(game[2].shape)
        #sys.exit()
        #boards = np.reshape(game[0], (-1, 6, 7, 3))
        #boards = np.moveaxis(game[0], [0, 1, 2, 3], [0,2,3,1])
        boards = np.swapaxes(np.swapaxes(game[0], 1, 2), 2, 3)
        #print("game[0].shape", game[0].shape)
        #print("boards.shape", boards.shape)
        #print(game[0][15][0])
        #print()
        """
        for y in range(0, 6):
            for x in range(0, 7):
                print(int(boards[15][y][x][0]), end=' ')
            print()
        flipped = np.flip(boards[15], 1)
        print()
        for y in range(0, 6):
            for x in range(0, 7):
                print(int(flipped[y][x][0]), end=' ')
            print()
        sys.exit()
        """
        #print(boards.shape)
        for board in boards:  # boards is an array of boards
            board_list.append(board)
            #board_list.append(np.flip(board, 1))
        for prediction in game[2]:  # game[2] is an array of predictions
            prediction_list.append(prediction[0])
            #prediction_list.append(np.flip(prediction[0], 0))
        winner_list_temp = [game[1]] * game[2].shape[0] # * 2
        winner_list = winner_list + winner_list_temp

    board_array = np.array(board_list)
    prediction_array = np.array(prediction_list)
    winner_array = np.array(winner_list)

    print(board_array.dtype)
    print(prediction_array.dtype)
    print(winner_array.dtype)

    print(board_array.shape)
    print(prediction_array.shape)
    print(winner_array.shape)

    label_array = np.append(winner_array, prediction_array, axis=0)


    print(label_array.shape)

    sys.exit()
    print()
    print()
    board_tensor = tf.convert_to_tensor(board_array, dtype='float64')
    prediction_tensor = tf.convert_to_tensor(prediction_array, dtype='float32')
    winner_tensor = tf.convert_to_tensor(winner_array, dtype='int64')

    print(board_tensor.shape)
    print(prediction_tensor.shape)
    print(winner_tensor.shape)

    print(board_tensor.dtype)
    print(prediction_tensor.dtype)
    print(winner_tensor.dtype)

    sys.exit()

    """
    input = np.array([board_array[16]])
    label1 = np.array([prediction_array[16]])
    label2 = np.array([winner_array[16]])
    print(label1.shape)
    print(label2.shape)
    label = [np.array([prediction_array[16]]), np.array([winner_array[16]])]
    print("THIS: ", prediction_array.shape)
    print()
    prediction = model.predict(input)
    y = label.copy()
    print(type(y))
    print(y[0].shape, y[1].shape)
    print(type(prediction))
    print(prediction[0].shape, prediction[1].shape)
    #print("CHEIF:", combined_fn(y, label))
    #combined_fn(label, prediction)

    #sys.exit()
    print("THIS: ", prediction[0].shape)
    print("THIS: ", prediction[1].shape)

    print("THIS: ", prediction[0])
    print("THIS: ", prediction[1])
    print(prediction_array[16])
    #sys.exit()
    y_true = [[0, 1, 0], [0, 0, 1]]
    y_pred = [[0.0, 1.0, 0], [0.0, 0.0, 1.0]]
    test = np.array([0,0,0,0,0,0,0])
    #cce = tf.keras.losses.CategoricalCrossentropy()
    kl = tf.keras.losses.KLDivergence()
    #print(cce(test, prediction_array[16]))
    #print(cce(y_true, y_pred))
    print(kl(prediction[0][0], prediction_array[16]))


    #y_train = {"policy_out1":prediction_array, "value_out1":winner_array}
    model.fit(board_array, (prediction_array, winner_array), epochs=3)
    """


def model_loss(y_true, y_pred):

    y_true_value = tf.gather(y_true, tf.constant(0))
    y_true_policy = tf.gather(y_true, tf.constant(1))
    y_pred_value = tf.gather(y_pred, tf.constant(0))
    y_pred_policy = tf.gather(y_pred, tf.constant(1))

    value_error = tf.square(tf.math.subtract(y_true_value, y_pred_value))
    #print("value_error", value_error)

    policy_error = tf.linalg.matvec(tf.expand_dims(y_true_policy, axis=0), tf.math.log(y_pred_policy))
    #print("policy_error", policy_error)

    return tf.math.subtract(value_error, policy_error)


def data_gen(number):  # 4.48% residual time  |  96.22% of total time
    tuple_list = []
    game_count = 0
    json_file = open('testing_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("testing_weights.h5")
    for y in range(number):
        tuple_list.append(tree_record(model, Game().gameState))     # cont: 91.74% of total time
        game_count += 1
        print("GAME:", game_count)
    with open("testing_data.pkl", "wb") as file:
        pickle.dump(tuple_list, file)


def build_new_model():
    model = Connect4Model()
    open('testing_model.json', 'w')
    model.save_model("testing_model.json", "testing_weights.h5")
    return model


def gamestate_mutator_test():
    game = Game()
    state = game.gameState
    for x in range(7):
        actions = state.allowedActions
        a = actions[np.random.randint(len(actions))]
        state, value, done = state.takeAction(a)
    gamestate_mutator(state.board)

def gamestate_mutator(board):
    #final = np.array([6, 7, 3])
    final_one = np.reshape((board == 1).astype(int), (6,7))
    final_two = np.reshape((board == -1).astype(int), (6,7))
    final_three = np.zeros((6,7))
    final = np.array([final_one, final_two, final_three])

    print(final[0])
    print()
    print(final[1])
    print()
    print(final[2])
    print()
    print(final.shape)



if __name__ == '__main__':
    main()
