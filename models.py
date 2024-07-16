import functools
import operator
import os
import cv2
import time
from gtts import gTTS
import numpy as np
import extract_features

import config
import model

latent_dim = config.latent_dim
num_encoder_tokens = config.num_encoder_tokens
num_decoder_tokens = config.num_decoder_tokens
time_steps_encoder = config.time_steps_encoder
max_probability = config.max_probability

# models
tokenizer, inf_encoder_model, inf_decoder_model = model.inference_model()
save_model_path = config.save_model_path
test_path = config.test_path
search_type = config.search_type
num = 0


def greedy_search(loaded_array):
    """

    :param f: the loaded numpy array after creating videos to frames and extracting features
    :return: the final sentence which has been predicted greedily
    """
    inv_map = index_to_word()
    states_value = inf_encoder_model.predict(loaded_array.reshape(-1, 80, 4096))
    target_seq = np.zeros((1, 1, 1500))
    final_sentence = ''
    target_seq[0, 0, tokenizer.word_index['bos']] = 1
    for i in range(15):
        output_tokens, h, c = inf_decoder_model.predict([target_seq] + states_value)
        states_value = [h, c]
        output_tokens = output_tokens.reshape(num_decoder_tokens)
        y_hat = np.argmax(output_tokens)
        if y_hat == 0:
            continue
        if inv_map[y_hat] is None:
            break
        if inv_map[y_hat] == 'eos':
            break
        else:
            final_sentence = final_sentence + inv_map[y_hat] + ' '
            target_seq = np.zeros((1, 1, 1500))
            target_seq[0, 0, y_hat] = 1
    return final_sentence


def decode_sequence2bs(input_seq):
    states_value = inf_encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, tokenizer.word_index['bos']] = 1
    beam_search(target_seq, states_value, [], [], 0)
    return decode_seq


def beam_search(target_seq, states_value, prob, path, lens):
    """

    :param target_seq: the array that is fed into the model to predict the next word
    :param states_value: previous state that is fed into the lstm cell
    :param prob: probability of predicting a word
    :param path: list of words from each sentence
    :param lens: number of words
    :return: final sentence
    """
    global decode_seq
    node = 2
    output_tokens, h, c = inf_decoder_model.predict(
        [target_seq] + states_value)
    output_tokens = output_tokens.reshape(num_decoder_tokens)
    sampled_token_index = output_tokens.argsort()[-node:][::-1]
    states_value = [h, c]
    for i in range(node):
        if sampled_token_index[i] == 0:
            sampled_char = ''
        else:
            sampled_char = list(tokenizer.word_index.keys())[
                list(tokenizer.word_index.values()).index(sampled_token_index[i])]
        MAX_LEN = 12
        if sampled_char != 'eos' and lens <= MAX_LEN:
            p = output_tokens[sampled_token_index[i]]
            if sampled_char == '':
                p = 1
            prob_new = list(prob)
            prob_new.append(p)
            path_new = list(path)
            path_new.append(sampled_char)
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index[i]] = 1.
            beam_search(target_seq, states_value, prob_new, path_new, lens + 1)
        else:
            p = output_tokens[sampled_token_index[i]]
            prob_new = list(prob)
            prob_new.append(p)
            p = functools.reduce(operator.mul, prob_new, 1)
            if p > max_probability:
                decode_seq = path
                max_probability = p


def decoded_sentence_tuning(decoded_sentence):
    # tuning sentence
    decode_str = []
    filter_string = ['bos', 'eos']
    uni_gram = {}
    last_string = ""
    for idx2, c in enumerate(decoded_sentence):
        if c in uni_gram:
            uni_gram[c] += 1
        else:
            uni_gram[c] = 1
        if last_string == c and idx2 > 0:
            continue
        if c in filter_string:
            continue
        if len(c) > 0:
            decode_str.append(c)
        if idx2 > 0:
            last_string = c
    return decode_str


def get_test_data(file_name):
    # loads the features array
    file_list = os.listdir(os.path.join(test_path, 'video'))
    # with open(os.path.join(self.test_path, 'testing.txt')) as testing_file:
    # lines = testing_file.readlines()
    # file_name = lines[self.num].strip()
    model = extract_features.model_cnn_load()
    f = extract_features.extract_features(file_name, model)
    return f, file_name


def test():
    X_test, filename = get_test_data()
    # generate inference test outputs
    if search_type == 'greedy':
        sentence_predicted = greedy_search(X_test.reshape((-1, 80, 4096)))
    else:
        sentence_predicted = ''
        decoded_sentence = decode_sequence2bs(X_test.reshape((-1, 80, 4096)))
        decode_str = decoded_sentence_tuning(decoded_sentence)
        for d in decode_str:
            sentence_predicted = sentence_predicted + d + ' '
    # re-init max prob
    max_probability = -1
    return sentence_predicted, filename


def main(filename, caption):
    """

    :param filename: the video to load
    :param caption: final caption
    :return:
    """
    speech = gTTS(text=caption, slow=True)
    speech.save(os.path.join(test_path, 'audio', filename[:-4] + '.mp3'))
    # 1. Initialize reading video object
    cap1 = cv2.VideoCapture(os.path.join(test_path, 'video', filename))
    cap2 = cv2.VideoCapture(os.path.join(test_path, 'video', filename))
    caption = '[' + ' '.join(caption.split()[1:]) + ']'
    # 2. Cycle through pictures
    while cap1.isOpened():
        ret, frame = cap2.read()
        ret2, frame2 = cap1.read()
        if ret:
            imS = cv2.resize(frame, (480, 300))
            cv2.putText(imS, caption, (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                        2, cv2.LINE_4)
            cv2.imshow("VIDEO CAPTIONING", imS)
        if ret2:
            imS = cv2.resize(frame, (480, 300))
            cv2.imshow("ORIGINAL", imS)
        else:
            break

        # Quit playing
        key = cv2.waitKey(25)
        if key == 27:  # Button esc
            break

    # 3. Free resources
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


def index_to_word():
    # inverts word tokenizer
    index_to_word = {value: key for key, value in tokenizer.word_index.items()}
    return index_to_word


def generate_video_caption(video_path):
    model = extract_features.model_cnn_load()
    features = extract_features.extract_features(video_path, model)
    # Placeholder function for your model that generates video captions
    # generate inference test outputs
    if search_type == 'greedy':
        sentence_predicted = greedy_search(features.reshape((-1, 80, 4096)))
    else:
        sentence_predicted = ''
        decoded_sentence = decode_sequence2bs(features.reshape((-1, 80, 4096)))
        decode_str = decoded_sentence_tuning(decoded_sentence)
        for d in decode_str:
            sentence_predicted = sentence_predicted + d + ' '
    return sentence_predicted
