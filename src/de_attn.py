# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np





class Decomposable_Attention(nn.Module):
    def __init__(self, vocab_size, device, word_matrix=None, embed_dim=300):
        super(Decomposable_Attention, self).__init__()
        self.embed_num = vocab_size
        self.embed_dim = embed_dim
        self.embed     = nn.Embedding(self.embed_num, self.embed_dim)
        self.device    = device
        
        if word_matrix is not None:
            word_matrix = torch.tensor(word_matrix).to(self.device)
            self.word_embedding.weight.data.copy_(word_matrix)
            self.word_embedding.weight.requires_grad = False
    
    def forward(self, phrases, topic_word_prob):

        phrases_embed = self.embed(phrases)
        topic_word_prob_embed = [self.embed(word_prob[0]) for word_prob in topic_word_prob]
        print phrases.shape
        print topic_word_prob.shape

        # attention matrix by cosine similarity
                

    def forward(self, batched_data):
        sent1_linear = torch.tensor(batched_data[1]).to(self.device)
        sent2_linear = torch.tensor(batched_data[2]).to(self.device)
        sent1_linear_embedding = self.embed(sent1_linear)
        sent2_linear_embedding = self.embed(sent2_linear)
        len1 = sent1_linear_embedding.size(1)
        len2 = sent2_linear_embedding.size(1)

        '''Attend'''
        f1 = self.mlp_f(sent1_linear_embedding.view(-1, self.embed_dim))
        f2 = self.mlp_f(sent2_linear_embedding.view(-1, self.embed_dim))
        f1 = f1.view(-1, len1, self.embed_dim)
        f2 = f2.view(-1, len2, self.embed_dim)
        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2)) 
        # e_{ij} batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, len2)).view(-1, len1, len2)
        # batch_size x len1 x len2
        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, len1)).view(-1, len2, len1)
        # batch_size x len2 x len1

        '''Compare''' 
        sent1_combine = torch.cat(
            (sent1_linear_embedding, torch.bmm(prob1, sent2_linear_embedding)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat(
            (sent2_linear_embedding, torch.bmm(prob2, sent1_linear_embedding)), 2)
        # batch_size x len2 x (hidden_size x 2)

        '''Aggregate'''
        g1 = self.mlp_g(sent1_combine.view(-1, 2 * self.embed_dim))
        g2 = self.mlp_g(sent2_combine.view(-1, 2 * self.embed_dim))
        g1 = g1.view(-1, len1, self.embed_dim)
        # batch_size x len1 x hidden_size
        g2 = g2.view(-1, len2, self.embed_dim)
        # batch_size x len2 x hidden_size
        sent1_output = torch.sum(g1, 1)  # batch_size x 1 x hidden_size
        sent1_output = torch.squeeze(sent1_output, 1)
        sent2_output = torch.sum(g2, 1)  # batch_size x 1 x hidden_size
        sent2_output = torch.squeeze(sent2_output, 1)
        input_combine = torch.cat((sent1_output, sent2_output), 1)
        
        '''Result'''
        result = self.mlp_h(input_combine)
        result = self.final_linear_1(result)
        result = self.final_linear_2(result)
        result = self.final_linear_3(result)
        return result

