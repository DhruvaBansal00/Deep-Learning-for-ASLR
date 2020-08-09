# coding: utf-8

"""
Implementation of a mini-batch.
"""

import torch

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, train_iter, src_pad, target_labels, pad_index, use_cuda=False):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        """
        self.src, self.src_lengths = train_iter, torch.tensor([len(video) for video in train_iter])
        self.src_mask = torch.tensor([[frame[0] != src_pad for frame in video] for video in train_iter]).unsqueeze(1)
        self.nseqs = self.src.size(0)
        self.trg_input = None
        self.trg_mask = None
        self.ntokens = None
        self.use_cuda = use_cuda

        trg, trg_lengths = target_labels, [len(label) for label in target_labels]
        # trg_input is used for teacher forcing, last one is cut off
        self.trg_input = trg[:, :-1]
        self.trg_lengths = trg_lengths
        # trg is used for loss computation, shifted by one since BOS
        self.trg = trg[:, 1:]
        # we exclude the padded areas from the loss computation
        self.trg_mask = (self.trg_input != pad_index).unsqueeze(1)
        self.ntokens = (self.trg != pad_index).data.sum().item()

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()

        if self.trg_input is not None:
            self.trg_input = self.trg_input.cuda()
            self.trg = self.trg.cuda()
            self.trg_mask = self.trg_mask.cuda()

    # def sort_by_src_lengths(self):
    #     """
    #     Sort by src length (descending) and return index to revert sort

    #     :return:
    #     """
    #     _, perm_index = self.src_lengths.sort(0, descending=True)
    #     rev_index = [0]*perm_index.size(0)
    #     for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
    #         rev_index[old_pos] = new_pos

    #     sorted_src_lengths = self.src_lengths[perm_index]
    #     sorted_src = self.src[perm_index]
    #     sorted_src_mask = self.src_mask[perm_index]
    #     if self.trg_input is not None:
    #         sorted_trg_input = self.trg_input[perm_index]
    #         sorted_trg_lengths = self.trg_lengths[perm_index]
    #         sorted_trg_mask = self.trg_mask[perm_index]
    #         sorted_trg = self.trg[perm_index]

    #     self.src = sorted_src
    #     self.src_lengths = sorted_src_lengths
    #     self.src_mask = sorted_src_mask

    #     if self.trg_input is not None:
    #         self.trg_input = sorted_trg_input
    #         self.trg_mask = sorted_trg_mask
    #         self.trg_lengths = sorted_trg_lengths
    #         self.trg = sorted_trg

    #     if self.use_cuda:
    #         self._make_cuda()

    #     return rev_index
