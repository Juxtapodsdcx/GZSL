import torch.nn as nn
import torch


class IntentClassifierNLI(nn.Module):
    def __init__(self, base_model, hidden_size=768, dropout=0.5):
        super(IntentClassifierNLI, self).__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.similarity_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def encode(self, pairs, attention_masks):
        """
        :param pairs: intent x utterance pair ids (bs, n_pairs, max_len)
        :param attention_masks: intent x utterance pair attention masks (bs, n_pairs, max_len)
        :return: pair embeddings (bs, n_pairs, emb_size)
        """
        bs, n_pairs, seq_len = pairs.size()
        flat_pairs = pairs.view(-1, seq_len)  # (batch_size*n_pairs, max_len)
        attention_masks = attention_masks.view(-1, seq_len)

        output = self.base_model(flat_pairs.long(), attention_masks.long(),output_hidden_states=True).hidden_states[-1]  # (batch_size*n_pairs,max_len,emb_size)
        s_tokens = output[:,0,:]

        return s_tokens.reshape(bs, n_pairs, -1)  # (batch_size, n_pairs, emb_size)

    def mlmForward(self, mask_input_ids,right_attention_mask, Y):
        # BERT forward
        outputs = self.base_model(input_ids=mask_input_ids,attention_mask=right_attention_mask,labels=Y)
        return outputs.loss

    def embeddingContrastiveForword(self,uttr_only_ids,uttr_only_attention_mask,right_input,attention_mask):
        # (batch_size, emb_size)
        right_embedding = self.base_model(input_ids=right_input.long(), attention_mask=attention_mask.long(),
                                          output_hidden_states=True).hidden_states[-1][:, 0, :].unsqueeze(
            1)  # (batch_size,1,emb_size)
        cls_tokens_uttr_only = self.encode(uttr_only_ids, uttr_only_attention_mask)

        cos_sim=torch.cosine_similarity(cls_tokens_uttr_only,right_embedding,dim=-1)

        return cos_sim

    def forward(self, pairs, attention_masks):
        # (batch_size, n_pairs, emb_size)
        cls_tokens = self.encode(pairs, attention_masks)

        # batch_size, n_pairs
        similarity = self.similarity_layer(cls_tokens).squeeze(dim=2)  # (batch_size, n_pairs)

        return similarity

