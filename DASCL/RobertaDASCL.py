# Date Last Updated: 06/06/2022
# Last Editor: PYW
# docstrings are a mix of descriptions from HuggingFace and our own descriptions 

import pandas as pd
import numpy as np
import torch
import transformers
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

class RobertaForSequenceClassificationDASCL(RobertaForSequenceClassification):
    """
    Contains the main class for sequence classification that combines the cross-entropy loss function and the dictionary-assisted supervised contrastive objective. 
    It is a modification of the RobertaForSequenceClassification class from HuggingFace: https://github.com/huggingface/transformers/blob/v4.19.2/src/transformers/models/roberta/modeling_roberta.py#L1161

    Attributes:
    config (['RobertaConfig']):  Model configuration class with all the parameters of the model. Initializing with a 
        config file does not load the weights associated with the model, only the configuration. Check out the 
        [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    temp (float): temperature parameter; if learnable, we initiate a parameter with the temp value 
    learnable_temp (boolean): if learnable_temp is True, the temperature parameter is learned during fine-tuning 
    lamb (float): the lambda parameter which balances the loss functions if set in contrastive mode 
    proj_dim (int): the projection dimension used with the RoBERTa embedding 
    contrastive (boolean): sets the class into contrastive mode; if False, DASCL will not be used
    data_augmentation (boolean): if True, the keyword-simplified text will be used with cross-entropy as well. 
        Akin to traditional data augmentation approaches 
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, 
                 config, 
                 temp=0.07, 
                 learnable_temp=True, 
                 lamb=0.9, 
                 proj_dim=128, 
                 contrastive=True, 
                 data_augmentation=False):
        super().__init__(config)
        
        # Initiate the projection matrix 
        self.proj = nn.Linear(config.hidden_size, proj_dim)

        if learnable_temp: 
            self.temp = nn.Parameter(torch.ones([]) * temp)
        else:
            self.temp = temp
        
        self.lamb = lamb
        
        self.contrastive = contrastive

        self.data_augmentation = data_augmentation 
        
    def forward(
        self,
        input_ids_og=None,
        attention_mask_og=None,
        input_ids_kw=None, 
        attention_mask_kw=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Args:
        input_ids_og (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary using the original text.
            Indices can be obtained using [`RobertaTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask_og (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices using the original text. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        input_ids_kw (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary using the keyword-simplified text.
            Indices can be obtained using [`RobertaTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask_kw (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices using the keyword-simplified text. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss with DASCL), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy with DASCL).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # outputs_og contains the embeddings for the original text
        outputs_og = self.roberta(
            input_ids_og,
            attention_mask=attention_mask_og,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # If keyword-simplified tokens are provided, then outputs_kw contains the embeddings for the 
        # keyword-simplified text 
        if input_ids_kw is not None: 
            outputs_kw = self.roberta(
                input_ids_kw,
                attention_mask=attention_mask_kw
            )

        # We use the 0th embedding, <CLS>, as the document embedding    
        sequence_output_og = outputs_og[0]
        # Get the logits for the original text using the 0th embedding
        logits_og = self.classifier(sequence_output_og)
        
        # Now we do the same thing with the keyword-simplified text, if provided 
        if input_ids_kw is not None: 
            sequence_output_kw = outputs_kw[0]
            if self.data_augmentation:
                logits_kw = self.classifier(sequence_output_kw) 
        
        # This section calculates the loss 
        loss = None
        # loss_mod is the cross-entropy loss function using the original text 
        loss_mod = None
        # if data augmentation is used, loss_kw is the cross-entropy loss function using the 
        # keyword-simplified text 
        if self.data_augmentation: 
            loss_kw = None
        # this section calculates the cross-entropy loss function in a variety of configurations, 
        # such as single label classification, multi-label classification, and regression
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss_mod = loss_fct(logits_og.squeeze(), labels.squeeze())
                else:
                    loss_mod = loss_fct(logits_og, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss_mod = loss_fct(logits_og.view(-1, self.num_labels), labels.view(-1))
                if self.data_augmentation:
                    loss_kw = loss_fct(logits_kw.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss_mod = loss_fct(logits_og, labels)
        
        # This section only runs if the keyword-simplified tokens are provided to the function 
        if input_ids_kw is not None:
            # if set in contrastive mode, we calculate the DASCL loss here 
            if self.contrastive: 
                po = self.proj(sequence_output_og[:,0,:])
                pk = self.proj(sequence_output_kw[:,0,:])
                loss_all = self.dascl_all(mat1=po, mat2=pk, labels=labels)
            
            # if set in contrastive mode but no data augmentation is needed, return just the CE 
            # for the original text, along with the DASCL loss 
            if self.contrastive and not self.data_augmentation: 
                loss = (1-self.lamb)*loss_mod + self.lamb*loss_all

            # if set in contrastive mode and data augmentation is needed, return the CE for both
            # the original text and the keyword-simplified text, along with the DASCL loss
            elif self.contrastive and self.data_augmentation: 
                loss = (1-self.lamb)*(loss_mod + loss_kw) + self.lamb*loss_all 
            # if NOT set in contrastive mode, it is assumed that data augmentation is required
            # if the keyword-simplified tokens are provided
            # in that case, we return the CE for the original text and the keyword-simplified text
            else:
                loss = loss_mod + loss_kw 
        # This section only runs if there are no keyword-simplified tokens provided
        # This is equivalent to vanilla RoBERTa
        else:
            loss = loss_mod 
        
        if not return_dict:
            output = (logits_og,) + outputs_og[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits_og,
            hidden_states=outputs_og.hidden_states,
            attentions=outputs_og.attentions,
        )
    
    def dascl_all(self, mat1, mat2, labels): 
        """
        Defines the DASCL objective 

        Args:
        mat1 (`torch.Tensor` of shape `(batch_size, proj_dim)`): matrix of projected RoBERTa embeddings of the 
            original text
        mat2 (`torch.Tensor` of shape `(batch_size, proj_dim)`): matrix of projected RoBERTa embeddings of the 
            keyword-simplified text
        labels (`torch.LongTensor` of shape `(batch_size,)`): labels for computing the DASCL loss 
        """
        
        # This step calculates the norm of the first matrix and the second matrix
        mat1_norms = torch.linalg.norm(mat1, dim=1) 
        mat2_norms = torch.linalg.norm(mat2, dim=1) 
        # We then divide by the norm to get the normalized vectors for mat1 and mat2
        mat1 = torch.div(mat1, mat1_norms.unsqueeze(-1))
        mat2 = torch.div(mat2, mat2_norms.unsqueeze(-1))
        
        # Convert labels to floats, in order to calculate which entries are considered similar 
        labels = labels.to(torch.float)
        
        # This stacks the two matrices together, so we get a 2N x d matrix 
        mat1_mat2 = torch.cat((mat1, mat2), dim=0)

        # This is to get the pairs of observations that are similar 
        # Observations are similar if they are of the same class 
        # We call this the correspondence matrix 
        # We first stack the labels, because one set of labels belongs to the original text, and the other set of labels belong to the keyword-simplified text. Because they are in the same order, we can simply stack the labels 
        labels_stacked = torch.cat((labels,labels)) 
        # correspondence1 gets all the observations that are both in class 1 (the positive class). 
        # To do this, we multiply correspondence1 (dim: 2N x d) by the transposed version of correspondence1 (dim: d x 2N)
        # This returns a 2N x 2N matrix where any pairs of observations that belong to class 1 
        correspondence1 = torch.matmul(labels_stacked.unsqueeze(-1), labels_stacked.unsqueeze(-1).T)
        # we then invert the labels to get the observations that belong to class 0. In other words, the observations that belong to class 0 are now marked as 1 in inverted_labels  
        inverted_labels = labels_stacked*(-1) + 1
        # we then do the same thing with the class 0 observations. Any pair of observations that are both of class 0 will have a 1 in this similarity matrix.  
        correspondence0 = torch.matmul(inverted_labels.unsqueeze(-1), inverted_labels.unsqueeze(-1).T) 
        # add the two correspondence matrices together to get a final correspondence matrix. 
        correspondence = correspondence1 + correspondence0
        # we subtract any correspondences of an observation with itself because we don't care about the similarity between an observation and itself. 
        correspondence = correspondence - torch.diag(torch.diag(correspondence))

        # The similarity matrix, similarity_matrix, is calculated using a matrix multiplication between the stacked matrix of mat1 and mat2, multiplied by its transpose. This yields a 2N x 2N similarity matrix. 
        # Notice these are all cosine similarities, because of the normalization that occurred before. 
        # We then divide this by the temperature parameter 
        # Then, we exponentiate this entire thing (the numerator of what is inside the log of equation (2) from the paper)
        similarity_matrix = torch.exp(torch.div(torch.matmul(mat1_mat2, mat1_mat2.T), self.temp))

        # To calculate the denominator, we sum across the similarity matrix by row
        # This is the denominator that is inside the log from equation (2) 
        # We subtract out the diagonal of the similarity matrix so it is not counted in the denominator (the indicator function in the denominator of the fraction inside the log of equation (2))
        denominators = torch.sum(similarity_matrix, dim=1) - torch.diag(similarity_matrix) 

        # We then divide by the denominator. This strange combination of tranposes works so that each line of the similarity matrix is divided by the corresponding number from the denominators vector. 
        similarity_matrix_divided = torch.div(similarity_matrix.T, denominators).T 
        
        # We then take the negative log of this entire thing, as done in Equation (2) 
        similarity_matrix_logged = -torch.log(similarity_matrix_divided) 

        # We then sum each row. We only keep observations where observations are in the same class. This is the second indicator function in the inside summation. 
        # We already took care of the first indicator function in the inside summation by forcing the diag values of the correspondence matrix to zero. 
        similarity_matrix_summed = torch.sum(similarity_matrix_logged*correspondence, dim=1) 
        
        # We then calculate the total number of other observations that are in the same class as a given observation. This is used to calculate the 1/(2N-1) part of Equation (2) 
        # Again, we don't need to subtract 1 from this because we forced the diag of the correspondence matrix to 0 
        total_sum_each_row = torch.sum(correspondence, dim=1) 
        
        # Lastly, we divide each row of the similarity_matrix_summed matrix by the total_sum_each_row. This is multiplying the entire inner summation by 1/(2N-1)
        # Then, we take the mean across all the individual contrastive losses
        # This is what the outer summation does 
        # We take the mean instead of simply summing so the losses remain on the same scale when scaling up or down batch sizes. 
        return torch.mean(torch.div(similarity_matrix_summed, total_sum_each_row)) 