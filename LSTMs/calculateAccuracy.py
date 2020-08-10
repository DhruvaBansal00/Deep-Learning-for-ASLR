import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(encoder, decoder, sentence, output_lang, sil0, sil1, max_length=470):
    with torch.no_grad():
        input_tensor = sentence
        input_length = len(sentence)
        encoder_hidden, encoder_cell = encoder.initHidden()


        for ei in range(input_length):
            _, encoder_hidden, encoder_cell = encoder(input_tensor[ei],
                                                     encoder_hidden, encoder_cell)

        decoder_input = torch.tensor([[sil0]], device=device)

        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        
        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_cell = decoder(
                decoder_input, decoder_hidden, decoder_cell)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == sil1:
                decoded_words.append('sil1')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

def calculateTrainingAccuracy(encoder, decoder, pairs, output_lang, sil0, sil1, file_name=None, write = True):
    total = 0
    correct = 0
    results = None
    if write:
        results = open(file_name, 'w')
    for pair in pairs:
        output_words = evaluate(encoder, decoder, pair[0], output_lang, sil0, sil1)
        output_sentence = ' '.join(output_words)
        sent = [output_lang.index2word[i.item()] for i in pair[1]]
        true_sentence = ' '.join(sent)
        if write:
            print('Predicted Sentence: ', output_sentence, file=results)
            print('True Sentence: ' , true_sentence, file=results)
        answer = None
        if output_sentence == true_sentence:
            correct += 1
            answer = "CORRECT"
        else:
            answer = "INCORRECT"
        total += 1
        if write:
            print('Result: ', answer, file=results)
    if write:
        print('Recognition Total: ', str(correct/total), file=results)
        results.close()
    return correct/total

