def display_dp(dp):
    for line in dp:
        print(str(line) + "\n")

def get_string_alignment(sent_1, sent_2):
    penalty_space = 1 ##insertion/deletion err weight
    penalty_sub = 1 ##subsititution err weight

    sentence_1_arr = sent_1.split(" ")
    sentence_2_arr = sent_2.split(" ")

    num_sent_1 = len(sentence_1_arr) + 1
    num_sent_2 = len(sentence_2_arr) + 1

    dp_matrix = [[0 for i in range(num_sent_2)] for j in range(num_sent_1)]

    for row in range(0, num_sent_1):
        for col in range(0, num_sent_2):
            if row == 0 or col == 0:
                dp_matrix[row][col] = (row + col) * penalty_space
            else:
                if sentence_1_arr[row-1] ==  sentence_2_arr[col-1]:
                    dp_matrix[row][col] = min(dp_matrix[row-1][col-1], penalty_space + min(dp_matrix[row-1][col], dp_matrix[row][col-1]))
                else:
                    dp_matrix[row][col] = min(penalty_sub + dp_matrix[row-1][col-1], penalty_space + min(dp_matrix[row-1][col], dp_matrix[row][col-1]))

    display_dp(dp_matrix)

    aligned_sent_1 = []
    aligned_sent_2 = []
    row = num_sent_1 - 1
    col = num_sent_2 - 1

    while row != 0 and col != 0:
        if sentence_1_arr[row - 1] == sentence_2_arr[col - 1] and dp_matrix[row][col] == dp_matrix[row - 1][col - 1]:
            aligned_sent_1.insert(0, sentence_1_arr[row -1])
            aligned_sent_2.insert(0, sentence_2_arr[col - 1])
            row -= 1
            col -= 1
        elif sentence_1_arr[row - 1] != sentence_2_arr[col - 1] and  dp_matrix[row][col] == (dp_matrix[row - 1][col - 1] + penalty_sub):
            aligned_sent_1.insert(0, sentence_1_arr[row - 1])
            aligned_sent_2.insert(0, sentence_2_arr[col - 1])
            row -= 1
            col -= 1
        elif dp_matrix[row][col] == (dp_matrix[row][col - 1] + penalty_space):
            aligned_sent_1.insert(0, "_")
            aligned_sent_2.insert(0, sentence_2_arr[col - 1])
            col -= 1
        elif dp_matrix[row][col] == (dp_matrix[row - 1][col] + penalty_space):
            aligned_sent_1.insert(0, sentence_1_arr[row - 1])
            aligned_sent_2.insert(0, "_")
            row -= 1

    while row > 0 or col > 0:
        if row > 0:
            aligned_sent_1.insert(0, sentence_1_arr[row - 1])
            aligned_sent_2.insert(0, "_")
            row -= 1
        else:
            aligned_sent_1.insert(0, "_")
            aligned_sent_2.insert(0, sentence_2_arr[col - 1])
            col -= 1
    
    print("First sentence  = " + str(aligned_sent_1))
    print("Second sentence  = " + str(aligned_sent_2))
