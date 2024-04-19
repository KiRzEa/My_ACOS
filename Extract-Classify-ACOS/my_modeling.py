from transformers import PreTrainedModel, AutoModel, AutoConfig, AutoTokenizer
from torch import nn

class BertForQuadABSA(nn.Module):

    def __init__(self, model_name, config, num_labels=2):
        super(BertForQuadABSA, self).__init__()
        # category-sentiment module parameters
        self.num_labels = [num_labels, 2]
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.crf_num = 6

        #opinion module
        self.crf = CRF(self.crf_num, batch_first=True)
        self.dense = DenseLayer(config)
        self.dense_output = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(768, self.crf_num)
        )
        self.imp_asp_classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.num_labels[1])
        )
        self.imp_opi_classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.num_labels[1])
        )

        self.apply(self.init_bert_weights)

    def forward(self, aspect_input_ids, aspect_labels,
                aspect_token_type_ids, aspect_attention_mask,
                exist_imp_aspect, exist_imp_opinion):

        # for name,parameters in self.state_dict().items():
        #     if parameters.size()[0] < 2:
        #         print(name,':',parameters)

        outputs = self.model(aspect_input_ids, aspect_token_type_ids, aspect_attention_mask)
        pooled_outputs, pooled_output = outputs.last_hidden_state, outputs.pooler_output

        bs = pooled_output.shape[0]
        hidden_size = pooled_output.shape[-1]

        # implicit aspect, opinion classification
        loss_fct = CrossEntropyLoss()
        imp_aspect_exist = self.imp_asp_classifier(pooled_output)
        imp_opinion_exist = self.imp_opi_classifier(pooled_outputs[range(pooled_outputs.shape[0]), torch.sum(aspect_attention_mask, dim=-1)-1])

        imp_aspect_loss = loss_fct(imp_aspect_exist, exist_imp_aspect.view(-1))
        imp_opinion_loss = loss_fct(imp_opinion_exist, exist_imp_opinion.view(-1))

        max_seq_len = aspect_input_ids.size()[1]
        sequence_output = self.dense_output(pooled_outputs)
        sequence_output = sequence_output.view(-1, max_seq_len, self.crf_num)
        ae_loss = - self.crf(sequence_output, aspect_labels, mask=aspect_attention_mask.byte(), reduction='mean')
        pred_tags = self.crf.decode(sequence_output, mask=aspect_attention_mask.byte())

        total_loss = ae_loss + imp_aspect_loss + imp_opinion_loss

        return [total_loss], [pred_tags, imp_aspect_exist, imp_opinion_exist]


class CategorySentiClassification(nn.Module):

    def __init__(self, model_name, config, num_labels=2):
        super(CategorySentiClassification, self).__init__(config)
        # category-sentiment module parameters
        self.output_attentions = output_attentions
        self.num_labels = [num_labels, 2]
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


        self.classifier = nn.Sequential(
            nn.Linear(768*2, num_labels)
        )

        self.apply(self.init_bert_weights)

    def forward(self, tokenizer, _e, aspect_input_ids,
                aspect_token_type_ids, aspect_attention_mask,
                candidate_aspect, candidate_opinion, label_id):

        aspect_seq_len = torch.max(torch.sum(aspect_attention_mask, dim=-1))
        max_seq_len = aspect_seq_len
        aspect_input_ids = aspect_input_ids[:, :max_seq_len].contiguous()
        aspect_token_type_ids = aspect_token_type_ids[:, :max_seq_len].contiguous()
        aspect_attention_mask = aspect_attention_mask[:, :max_seq_len].contiguous()
        candidate_aspect = candidate_aspect[:, :max_seq_len].contiguous()
        candidate_opinion = candidate_opinion[:, :max_seq_len].contiguous()


        outputs = self.model(aspect_input_ids, aspect_token_type_ids, aspect_attention_mask)
        pooled_outputs, pooled_output = outputs.last_hidden_state, outputs.pooler_output


        bs = pooled_output.shape[0]
        hidden_size = pooled_output.shape[-1]
        

        candidate_aspect_sum = torch.sum(candidate_aspect, -1).float()
        aspect_denominator = (candidate_aspect_sum+candidate_aspect_sum.eq(0).float()).unsqueeze(-1).repeat(1, hidden_size)
        candidate_aspect_rep = torch.div(torch.matmul(candidate_aspect.float().unsqueeze(1), pooled_outputs).squeeze(1), aspect_denominator)

        candidate_opinion_sum = torch.sum(candidate_opinion, -1).float()
        opinion_denominator = (candidate_opinion_sum+candidate_opinion_sum.eq(0).float()).unsqueeze(-1).repeat(1, hidden_size)
        candidate_opinion_rep = torch.div(torch.matmul(candidate_opinion.float().unsqueeze(1), pooled_outputs).squeeze(1), opinion_denominator)

        fused_feature = torch.cat([candidate_aspect_rep, candidate_opinion_rep], -1)
        fused_feature = self.classifier(self.dropout(fused_feature))
        cate_loss_fct = BCEWithLogitsLoss()
        loss = cate_loss_fct(fused_feature.view(-1, self.num_labels[0]), label_id.view(-1, self.num_labels[0]).float() )
        # pair_loss = loss_fct(pred_matrix, pair_matrix.view(-1))
        return [loss], [fused_feature]