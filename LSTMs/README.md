# LSTM
Exploring ASL recognition using LSTMs

Run the main.py file for training the LSTM model. Make sure filename of the files in the data folder are in the following format - nameOfUser.phrase.id.ark. The words in the phrase must be seperated by underscores. No grammar or dict needed.

| User | Num. Phrases | Features | Lr | Hidden Layer Dim | Epochs | Sentence Accuracy |
| --- | --- | --- | --- | -- | -- | -- |
| Ravi | 3 - 5 | 20 Elements | 1e-4 | 1024 | ~20 | 90.79754601226994% |
| Prerna | 3 | Best | 1e-4 | 1024 | 20 | 100% |


Best feature set is - ['left_hand_x', 'left_hand_y', 'left_hand_w', 'left_hand_h', 'right_hand_x', 'right_hand_y', 'right_hand_w', 'right_hand_h', 'horizontal_hand_dist', 'vertical_hand_dist', 'delta_horizontal_hand_dist', 'delta_vertical_hand_dist']

