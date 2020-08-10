import torch
import torch.nn as nn
import random 
import time
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from calculateAccuracy import calculateTrainingAccuracy
import copy

teacher_forcing_ratio = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, sil0, sil1):
    encoder_hidden, encoder_cell = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = len(input_tensor)
    target_length = target_tensor.size(0)

    loss = 0

    for ei in range(input_length):
        _, encoder_hidden, encoder_cell = encoder(
            input_tensor[ei], encoder_hidden, encoder_cell)

    decoder_input = torch.tensor([[sil0]], device=device)

    decoder_hidden, decoder_cell = encoder_hidden, encoder_cell

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_cell = decoder(
                decoder_input, decoder_hidden, decoder_cell)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_cell = decoder(
                decoder_input, decoder_hidden, decoder_cell)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == sil1:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def testSetLoss(encoder, decoder, input_tensor, target_tensor, criterion, sil0, sil1):
    with torch.no_grad():
        input_tensor = input_tensor
        input_length = len(input_tensor)
        target_length = target_tensor.size(0)

        loss = 0

        encoder_hidden, encoder_cell = encoder.initHidden()
        for ei in range(input_length):
            _, encoder_hidden, encoder_cell = encoder(input_tensor[ei],
                                                     encoder_hidden, encoder_cell)

        decoder_input = torch.tensor([[sil0]], device=device)

        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_cell = decoder(
                decoder_input, decoder_hidden, decoder_cell)
            topv, topi = decoder_output.data.topk(1)
            loss += criterion(decoder_output, target_tensor[di])
            if topi.item() == sil1:
                break
            decoder_input = topi.squeeze().detach()

        return loss.item() / target_length

def trainIters(encoder, decoder, epochs, train_set, test_set, sil0, sil1, output_lang, lr=1e-4, lr_decay=1, lr_drop_epoch=10, l2_penalty = 0):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    test_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay = l2_penalty)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr, weight_decay = l2_penalty)

    best_test_acc = -1
    best_encoder = None
    best_decoder = None

    criterion = nn.NLLLoss()

    for iter in range(1, epochs + 1):
        if iter == lr_drop_epoch:
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr * (lr_decay)**(iter), weight_decay = l2_penalty)
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr * (lr_decay)**(iter), weight_decay = l2_penalty)

        for pairs in train_set:
            input_tensor = pairs[0]
            target_tensor = pairs[1]
            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion, sil0, sil1)
            print_loss_total += loss

        for pair in test_set:
            input_tensor = pair[0]
            target_tensor = pair[1]
            test_loss_total += testSetLoss(encoder, decoder, input_tensor, target_tensor, criterion, sil0, sil1)

        print_loss_avg = print_loss_total / len(train_set)
        test_loss_avg = test_loss_total / len(test_set)
        print_loss_total = 0
        test_loss_total = 0
        test_acc = calculateTrainingAccuracy(encoder, decoder, test_set, output_lang, sil0, sil1, write=False)
        train_acc = calculateTrainingAccuracy(encoder, decoder, train_set, output_lang, sil0, sil1, write=False)
        print('%s (%d %d%%) train loss: %.4f train acc: %.4f test loss: %.4f test acc: %.4f' % (timeSince(start, iter / epochs),
                                        iter, iter / epochs * 100, print_loss_avg, train_acc, test_loss_avg, test_acc))
        
        if test_acc > best_test_acc:
            bet_test_acc = test_acc
            best_encoder = copy.deepcopy(encoder)
            best_decoder = copy.deepcopy(decoder)

        plot_loss_avg = print_loss_avg
        plot_losses.append(plot_loss_avg)

    # showPlot(plot_losses)
    return best_encoder, best_decoder