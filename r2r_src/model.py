
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
import pdb
import math
import sys
from dyrelu import LangDyReLUC
from fusion import MutanFusion
from utils import attention


class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)

class BEncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1,bert=None,update=False):
        super(BEncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in BEncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        #self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.bert = bert
        self.update = update
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths,att_mask=None,img_feats=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        #embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        assert att_mask is not None
        seq_max_len = att_mask.size(1)
        outputs = self.bert(inputs[:,:seq_max_len], attention_mask=att_mask, img_feats=img_feats)  # (batch, seq_len, embedding_size)
        embeds = outputs[0]
        if not self.update:
            embeds = embeds.detach()
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)

class CEncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1,bert=None,update=False,bert_hidden_size=768):
        super(CEncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in CEncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.linear_in = nn.Linear(bert_hidden_size, embedding_size)
        self.bert = bert
        self.update = update
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths,att_mask=None,img_feats=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        #embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        assert att_mask is not None
        seq_max_len = att_mask.size(1)
        outputs = self.bert(inputs[:,:seq_max_len], attention_mask=att_mask, img_feats=img_feats)  # (batch, seq_len, embedding_size)
        bertembeds = outputs[0]
        if not self.update:
            bertembeds = bertembeds.detach()
        embeds = self.linear_in(bertembeds)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)





class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask.bool(), -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn



class ShiftSoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim, kernel_size=3):
        '''Initialize layer.'''
        super(ShiftSoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.linear_shift = nn.Linear(query_dim, kernel_size)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()
        self.kernel_size = kernel_size
        self.padding_size = kernel_size // 2

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''

        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask.bool(), -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 3, int(attn.size(1)/3))  # batch x 1 x seq_len
        B, _, H = context.shape
        shift_kernel = torch.softmax(self.linear_shift(h),dim=-1).unsqueeze(1) # B, 1, 3 as a kenerl: outdim, 1, kernel_width
        attn3 = torch.cat([attn3[:,:,-self.padding_size:], attn3, attn3[:,:,:self.padding_size]], dim=-1)
        attn3 = attn3.transpose(0,1) # 3, B, L; minibatch, in_channel, input_width
        attn3 = F.conv1d(attn3, shift_kernel, groups=B) # 3, B, L
        attn3 = attn3.transpose(0,1) # B, 3, L
        attn3 = attn3.reshape(B, 1, -1)
        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn




class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        return h_1, c_1, logit, h_tilde

class BAttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4, pred_back=False):
        super(BAttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        if args.use_shift:
            self.feat_att_layer = ShiftSoftDotAttention(hidden_size, feature_size, args.shift_kernel_size)
        else:
            self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size * 2)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.pred_back = pred_back

        if self.pred_back:
            self.back_candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

        self.pred_pm = args.pred_pm
        if self.pred_pm:
            if args.pm_type in ["att", "plain_att"]:
                pm_dim = args.maxInput
            elif args.pm_type in ["att_hid", "plain_att_hid"]:
                pm_dim = args.maxInput + hidden_size
            self.critic = nn.Sequential(
                nn.Linear(pm_dim, 1),
                nn.Sigmoid()
            ).cuda()

        self.input_noise = None
        self.output_noise = None
        # print("embedding_size",embedding_size) # 64
        # print("hidden_size",hidden_size) # 1024
        # print("dropout_ratio",dropout_ratio) # 0.5
        # print("feature_size",feature_size) # 2176
        # featdropout 0.4

    def init_noise(self, shape):
        self.input_noise = self.drop(torch.ones(shape).cuda())
        self.output_noise = self.drop(torch.ones(shape).cuda())

    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''

        # print("action", action.shape) # b,128
        # print("feature",feature.shape) # b, 36, 2176
        # print("cand_feat",cand_feat.shape) # b, 10, 2176
        # print("h_0", h_0.shape) # b, 1024
        # print("prev_h1",prev_h1.shape) # b, 1024
        # print("c_0",c_0.shape) # b,1024
        # print("ctx",ctx.shape) # b, max_seq_len, 2048
        # print("ctx_mask",ctx_mask.shape) # b, max_seq_len
        # print(already_dropfeat,already_dropfeat) #

        aux_outputs = {}

        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        if args.decoder_consistent_drop:
            h_1_drop = h_1 * self.input_noise
        else:
            h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)





        # Adding Dropout
        if args.decoder_consistent_drop:
            h_tilde_drop = h_tilde * self.output_noise
        else:
            h_tilde_drop = self.drop(h_tilde)


        if args.pred_pm:
            if args.pm_type in ["att", "att_hid"]:
                attw = []
                ctx_len = (ctx_mask == False).sum(dim=-1)
                for i_alpha, i_len in zip(alpha, ctx_len):
                    attw.append(F.interpolate(i_alpha[:i_len].unsqueeze(0).unsqueeze(0), args.maxInput, mode='linear', align_corners=True).squeeze(0).squeeze(0))
                attw = torch.stack(attw)
                attw = attw/(attw.sum(-1, keepdims=True) + 1e-10)
            elif args.pm_type in ["plain_att", "plain_att_hid"]:
                B, L = alpha.shape
                if L < args.maxInput:
                    padded_attention = torch.zeros(B, args.maxInput-L).cuda()
                    attw = torch.cat([alpha, padded_attention] , dim=-1)
                else:
                    attw = alpha

            if args.pm_type in ["att", "plain_att"]:
                pm_score = self.critic(attw)
            elif args.pm_type in ["att_hid", "plain_att_hid"]:
                pm_score = self.critic(torch.cat([attw, h_tilde_drop], dim=-1))
            aux_outputs['pm_score'] = pm_score


        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        if self.pred_back:
            if args.back_input == "pre":
                _, back_logit = self.back_candidate_att_layer(prev_h1, cand_feat, output_prob=False)
            elif args.back_input == "cur":
                _, back_logit = self.back_candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)
            aux_outputs['back_logit'] = back_logit

        # print("h_1", h_1.shape) # b, 1024
        # print("c_1", c_1.shape) # b, 1024
        # print("h_tilde", c_1.shape) b, 1024
        # print("cand_feat", cand_feat.shape) # b, max_can_num, 2176
        # print("logit", logit.shape) # b, max_can_num

        return h_1, c_1, logit, h_tilde, aux_outputs



class AdvancedDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4, pred_back=False):
        super(AdvancedDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size * 2)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.pred_back = pred_back
        self.pm_predictor = nn.Linear(80,1)

        if self.pred_back:
            self.back_candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

    def _pack_attw(self, attw_lang):
        B, attw_lang_len = attw_lang.shape
        if attw_lang_len >=args.maxInput:
            return attw_lang
        else:
            padded_attw = torch.zeros(B, args.maxInput -attw_lang_len).to(attw_lang.device)
            attw_lang = torch.cat([attw_lang, padded_attw], dim=1)
            return attw_lang



    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):


        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))


        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)


        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        if self.pred_back:
            _, back_logit = self.back_candidate_att_layer(prev_h1, cand_feat, output_prob=False)
        else:
            back_logit = None
        alpha = self._pack_attw(alpha)
        pred_progress = self.pm_predictor(alpha)


        return h_1, c_1, logit, h_tilde, back_logit, pred_progress




class KVMemAttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4, pred_back=False):
        super(KVMemAttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size * 2)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.pred_back = pred_back
        self.kv = nn.Parameter(nn.init.normal_(torch.Tensor(100, hidden_size)))
        #self.mvalue = nn.Parameter(nn.init.normal_(torch.Tensor(100, hidden_size)))
        self.kv_att_layer = SoftDotAttention(hidden_size, hidden_size)



        if self.pred_back:
            self.back_candidate_att_layer = SoftDotAttention(hidden_size, feature_size)



    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):


        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))


        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
        ext_shape = h_tilde.shape[:1]+self.kv.shape
        h_tilde = h_tilde + self.kv_att_layer(h_tilde, self.kv.expand(ext_shape))[0]




        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        if self.pred_back:
            _, back_logit = self.back_candidate_att_layer(prev_h1, cand_feat, output_prob=False)
        else:
            back_logit = None

        return h_1, c_1, logit, h_tilde, back_logit


class NewAttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4, pred_back=False):
        super(NewAttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+hidden_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size * 2)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.pred_back = pred_back
        self.kv = nn.Parameter(nn.init.normal_(torch.Tensor(100, hidden_size)))
        #self.mvalue = nn.Parameter(nn.init.normal_(torch.Tensor(100, hidden_size)))
        self.kv_att_layer = SoftDotAttention(hidden_size, hidden_size)
        self.visionpose_to_hidden = nn.Linear(feature_size, hidden_size)
        self.language_to_hidden = nn.Linear(2048, hidden_size)



        if self.pred_back:
            self.back_candidate_att_layer = SoftDotAttention(hidden_size, feature_size)



    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):


        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)


        # step: attn vision
        prev_h1_drop = self.drop(prev_h1)
        feature = self.visionpose_to_hidden(feature)
        attn_feat, attw_feat = attention(value=feature, key=feature, query=prev_h1_drop)


        # new hidden
        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))



        # step: attn language
        ctx = self.language_to_hidden(ctx)
        h_1_drop = self.drop(h_1)
        attn_ctx, attw_ctx = attention(value=ctx, key=ctx, query=h_1_drop)
        h_tilde = h_1 + attn_ctx



        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        cand_feat = self.visionpose_to_hidden(cand_feat)


        _, logit = attention(value= cand_feat, key= cand_feat, query=h_tilde_drop,  output_prob=False)

        if self.pred_back:
            _, back_logit = self.back_candidate_att_layer(prev_h1, cand_feat, output_prob=False)
        else:
            back_logit = None

        return h_1, c_1, logit, h_tilde, back_logit


class MutanAttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4, pred_back=False):
        super(MutanAttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size * 2)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.pred_back = pred_back

        if self.pred_back:
            self.back_candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

        mutan_opt = {
            'dim_hv': 1024,
            'dim_hq': 2048,
            'dim_mm': 256,
            'R': 32,
            'dropout_hv': 0.2,
            'dropout_hq': 0.2
        }
        self.mutan = MutanFusion(mutan_opt, False, False)
        self.linear_mutan = nn.Linear(256,1024)
        # print("embedding_size",embedding_size) # 64
        # print("hidden_size",hidden_size) # 1024
        # print("dropout_ratio",dropout_ratio) # 0.5
        # print("feature_size",feature_size) # 2176
        # featdropout 0.4


    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''



        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))


        h_1_drop = self.drop(h_1)
        attended_instr, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask, output_tilde=False)
        # print("attended_instr", attended_instr.shape)
        # print("h_1_drop", h_1_drop.shape)

        h_tilde = self.mutan( h_1_drop, attended_instr)
        h_tilde = self.linear_mutan(h_tilde)




        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        if self.pred_back:
            _, back_logit = self.back_candidate_att_layer(prev_h1, cand_feat, output_prob=False)
        else:
            back_logit = None
        # print("h_1", h_1.shape) # b, 1024
        # print("c_1", c_1.shape) # b, 1024
        # print("h_tilde", c_1.shape) b, 1024
        # print("cand_feat", cand_feat.shape) # b, max_can_num, 2176
        # print("logit", logit.shape) # b, max_can_num

        return h_1, c_1, logit, h_tilde, back_logit


class DoubleBAttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4, pred_back=False):
        super(DoubleBAttnDecoderLSTM, self).__init__()

        self.image_decoder = BAttnDecoderLSTM(embedding_size, hidden_size,
                       dropout_ratio, feature_size, pred_back)
        self.depth_decoder = BAttnDecoderLSTM(embedding_size, hidden_size,
                                              dropout_ratio, feature_size, pred_back)


    def forward(self, action, feature, d_feature, cand_feat, d_cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        i_h_1, i_c_1, i_logit, i_h_tilde, i_back_logit = self.image_decoder(action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False)
        d_h_1, d_c_1, d_logit, d_h_tilde, d_back_logit = self.depth_decoder(action, d_feature, d_cand_feat,
                                          h_0, prev_h1, c_0,
                                          ctx, ctx_mask=None,
                                          already_dropfeat=False)

        h_1 = i_h_1 + d_h_1
        c_1 = i_c_1 + d_c_1
        logit = i_logit + d_logit
        h_tilde = i_h_tilde + d_h_tilde
        if i_back_logit:
            back_logit = i_back_logit + d_back_logit
        else:
            back_logit = None
        return h_1, c_1, logit, h_tilde, back_logit

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.dim = args.critic_dim
        self.state2value = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x

class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


# new 202010

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2




# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y

class MCA_SGA_SGA(nn.Module):
    def __init__(self, __C):
        super(MCA_SGA_SGA, self).__init__()

        self.enc_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc, dec in zip(self.enc_list, self.dec_list):
            x = enc(x, y, x_mask, y_mask)
            y = dec(y, x, y_mask, x_mask)

        return x, y



class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1) # B, L, G

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1) # B, D
            )

        x_atted = torch.cat(att_list, dim=1) # B, D*G
        x_atted = self.linear_merge(x_atted) # B, OUT_D

        return x_atted


class McattEncoder(nn.Module):
    def __init__(self, __C, token_size):
        super(McattEncoder, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.backbone = MCA_SGA_SGA(__C)
        self.attflat_lang = AttFlat(__C)

    def forward(self, seq, seq_mask, seq_lengths, f_t_all):
        """
        :param seq: B, seq_len
        :param seq_mask: B, seq_len
        :param seq_lengths: B
        :param f_t_all: B, V_NUM, V_DIM
        :return:
        """
        B, V_NUM, _ = f_t_all.shape


        seq_mask = seq_mask.unsqueeze(1).unsqueeze(2) # B, 1, 1, L_NUM
        v_mask = torch.zeros(B, 1, 1, V_NUM, dtype=torch.bool).to(seq_mask.device)  # B,1,1,V_NUM (expand to B,H, Query_NUM, V_NUM)

        seq_feat = self.embedding(seq) # B, L_NUM, E_DIM
        seq_feat, _ = self.lstm(seq_feat) #

        # Pre-process Image Feature
        v_feat = self.img_feat_linear(f_t_all)

        # Backbone Framework
        seq_feat, v_feat = self.backbone(
            seq_feat,
            v_feat,
            seq_mask,
            v_mask
        )

        attended_txt = self.attflat_lang(
            seq_feat,
            seq_mask
        )

        attended_v = self.att(v_feat, v_feat, attended_txt.unsqueeze(1))

        # print("seq_feat", seq_feat.shape)
        # print("v_feat", v_feat.shape)
        # print("attended_v", attended_v.shape)

        return seq_feat, attended_txt, v_feat, attended_v


    # Masking
    def make_mask(self, feature):
        """
        :param feature: B, L ,D
        :return: B, 1, 1, L
        """

        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

    def att(self, value, key, query, mask=None):
        """
        :param value: B, V_NUM, D
        :param key:  B, V_NUM, D
        :param query: B, Q_NUM, D
        :param mask: B, 1, V_NUM
        :return:
        """
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k) # B, Q_NUM, V_NUM

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        return torch.matmul(att_map, value) # B, Q_NUM, D


# class McattDecoder(nn.Module):
#
#
#     def __init__(self, embedding_size, hidden_size,
#                        dropout_ratio, feature_size=2048+4):
#         super(McattDecoder, self).__init__()
#         self.embedding_size = embedding_size
#         self.feature_size = feature_size
#         self.hidden_size = hidden_size
#         self.embedding = nn.Sequential(
#             nn.Linear(args.angle_feat_size, self.embedding_size),
#             nn.Tanh()
#         )
#
#         self.lstm = nn.LSTMCell(embedding_size + feature_size, hidden_size)
#         self.drop_env = nn.Dropout(p=args.featdropout)
#
#     def forward(self, action, feature, cand_feat,
#                 h_0, prev_h1, c_0,
#                 ctx, ctx_mask=None,
#                 already_dropfeat=False):
#
#     def forward(self, attended_v, input_a_t, f_t, candidate_feat, already_dropfeat):
#         if not already_dropfeat:
#             candidate_feat[..., :-args.angle_feat_size] = self.drop_env(candidate_feat[..., :-args.angle_feat_size])
#
#         attended_v = torch.cat([attended_v, input_a_t.unsqueeze(1)], dim=-1) # B, 1, D
#         attended_v = self.fc(attended_v) # B, 1, D
#         logits = (attended_v*candidate_feat).sum(dim=-1) # B, max_candidate_num
#
#         # print("attended_v", attended_v.shape)
#         # print("input_a_t", input_a_t.shape)
#         # print("f_t", f_t.shape )
#         # print("candidate_feat", candidate_feat.shape )
#         # print("already_dropfeat", already_dropfeat)
#         # print("logits", logits.shape)
#         return logits



class McattDecoder(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(McattDecoder, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)
        # print("embedding_size",embedding_size) # 64
        # print("hidden_size",hidden_size) # 1024
        # print("dropout_ratio",dropout_ratio) # 0.5
        # print("feature_size",feature_size) # 2176
        # featdropout 0.4


    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''

        # print("action", action.shape) # b,128
        # print("feature",feature.shape) # b, 36, 2176
        # print("cand_feat",cand_feat.shape) # b, 10, 2176
        # print("h_0", h_0.shape) # b, 1024
        # print("prev_h1",prev_h1.shape) # b, 1024
        # print("c_0",c_0.shape) # b,1024
        # print("ctx",ctx.shape) # b, max_seq_len, 2048
        # print("ctx_mask",ctx_mask.shape) # b, max_seq_len
        # print("already_dropfeat",already_dropfeat) #
        

        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))


        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)
        # print("h_1", h_1.shape) # b, 1024
        # print("c_1", c_1.shape) # b, 1024
        # print("h_tilde", c_1.shape) b, 1024
        # print("cand_feat", cand_feat.shape) # b, max_can_num, 2176
        # print("logit", logit.shape) # b, max_can_num

        return h_1, c_1, logit, h_tilde


# def attention(value, key, query, mask=None):
#     d_k = query.size(-1)
#
#     scores = torch.matmul(
#         query, key.transpose(-2, -1)
#     ) / math.sqrt(d_k)
#
#     if mask is not None:
#         scores = scores.masked_fill(mask, -1e9)
#
#     att_map = F.softmax(scores, dim=-1)
#
#     return torch.matmul(att_map, value), att_map


class MTDecoder(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(MTDecoder, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        # self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        # self.attention_layer = SoftDotAttention(hidden_size, hidden_size * 2)
        # self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.v_stop_feat = nn.Parameter(nn.init.normal_(torch.ones(args.feature_size + args.angle_feat_size)))
        self.vemb_to_v = nn.Linear(768, args.feature_size + args.angle_feat_size)
        self.hv_to_upd = nn.Linear(args.d_enc_hidden_size + args.feature_size, args.d_enc_hidden_size)
        self.h_to_ctx = nn.Linear(args.d_enc_hidden_size, 2* args.d_enc_hidden_size)
        self.mlp = MLP(in_size= args.feature_size + args.angle_feat_size + 2* args.d_enc_hidden_size +  self.embedding_size, mid_size=args.d_enc_hidden_size, out_size=1, dropout_r=0., use_relu=False)
        # print("embedding_size",embedding_size) # 64
        # print("hidden_size",hidden_size) # 1024
        # print("dropout_ratio",dropout_ratio) # 0.5
        # print("feature_size",feature_size) # 2176
        # featdropout 0.4


    def forward(self, action, feature, v_emb, cand_feat, cand_idx,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''

        # print("action", action.shape) # b,128
        # print("feature",feature.shape) # b, 36, 2176
        # print("v_emb", v_emb.shape) # b, 36, 768
        # print("cand_feat",cand_feat.shape) # b, 10, 2176
        # print("cand_idx", cand_idx.shape)
        # print("h_0", h_0.shape) # b, 1024
        # print("prev_h1",prev_h1.shape) # b, 1024
        # print("c_0",c_0.shape) # b,1024
        # print("ctx",ctx.shape) # b, max_seq_len, 2048
        # print("ctx_mask",ctx_mask.shape) # b, max_seq_len
        # print(already_dropfeat,already_dropfeat) #


        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)

        ## vision

        ## read current view
        feature = self.vemb_to_v(v_emb) + feature
        mean_v = torch.mean(feature[..., :-args.angle_feat_size], dim=1) # b, 2048

        ## udpate state h
        update_v = self.hv_to_upd(torch.cat([prev_h1, mean_v], dim=1))
        update_score = torch.sigmoid(update_v)
        h = prev_h1*(1-update_score) + update_score*update_v

        ## read instructions
        instr, instr_att = attention(ctx, ctx, self.h_to_ctx(h).unsqueeze(1))


        B, _ , F_DIM = feature.shape

        ## append stop visual token
        feature = torch.cat([feature, self.v_stop_feat.expand(B,1,F_DIM)], dim =-2)

        ## instruction, heading angle, visual token
        instr_angle = torch.cat([instr, action_embeds.unsqueeze(1)], dim=-1)  #
        feature = torch.cat([feature, instr_angle.expand(B, feature.shape[1], instr_angle.shape[2])], dim =-1)
        score = self.mlp(feature).squeeze(-1)
        logit = score.gather(-1, cand_idx)

        return h, h, logit, h




## dyrelu
class DyReluAttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(DyReluAttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size * 2)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.dyrelu1 = LangDyReLUC(2048, 2048)
        # self.dymlp = MLP(2048, 512, 2048)
        # print("embedding_size",embedding_size) # 64
        # print("hidden_size",hidden_size) # 1024
        # print("dropout_ratio",dropout_ratio) # 0.5
        # print("feature_size",feature_size) # 2176
        # featdropout 0.4


    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''

        # print("action", action.shape) # b,128
        # print("feature",feature.shape) # b, 36, 2176
        # print("cand_feat",cand_feat.shape) # b, 10, 2176
        # print("h_0", h_0.shape) # b, 1024
        # print("prev_h1",prev_h1.shape) # b, 1024
        # print("c_0",c_0.shape) # b,1024
        # print("ctx",ctx.shape) # b, max_seq_len, 2048
        # print("ctx_mask",ctx_mask.shape) # b, max_seq_len
        # print(already_dropfeat,already_dropfeat) #

        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))


        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)


        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        max_feat, _ = torch.max(feature[..., :-args.angle_feat_size], dim=1)
        # print("max_feat", max_feat.shape)
        # feat_gate = torch.sigmoid(self.dymlp(max_feat))
        # cand_feat[..., :-args.angle_feat_size] = cand_feat[..., :-args.angle_feat_size] + feat_gate.unsqueeze(1)

        # cand_view = cand_feat[..., :-args.angle_feat_size]
        cand_view = cand_feat[..., :-args.angle_feat_size]
        cand_angle = cand_feat[..., -args.angle_feat_size:]
        cand_view = self.dyrelu1(cand_view, max_feat)
        cand_feat = torch.cat([cand_view, cand_angle], dim=-1)

        # print("feat_gate", feat_gate.shape)
        # print("cand_feat", cand_feat.shape)



        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)
        # print("h_1", h_1.shape) # b, 1024
        # print("c_1", c_1.shape) # b, 1024
        # print("h_tilde", c_1.shape) b, 1024
        # print("cand_feat", cand_feat.shape) # b, max_can_num, 2176
        # print("logit", logit.shape) # b, max_can_num

        return h_1, c_1, logit, h_tilde


# depth guided AdaIN

def calc_mean_std(feat, eps=1e-5, dim=-1):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 3)
    N, C = size[:2]
    feat_var = feat.var(dim=dim, keepdim=True) + eps
    feat_std = feat_var.sqrt()
    feat_mean = feat.mean(dim=dim, keepdim=True)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size() == style_feat.size())
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
