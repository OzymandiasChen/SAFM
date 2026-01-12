
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from copy import deepcopy
from glob import glob
import logging
from settings_myadaptor import args, TASK_DICT, SPECIAL_TOKENS, SPECIAL_TOKEN_IDS, FILL_VAL
from settings_myadaptor import TOKENIZER, LEN_FACTOR, DATA_ATTRS, MEMORY_FACTOR,  \
                            MODEL_CONFIG, MODEL_CLASS, CONFIG_CLASS, CONFIG_NAME, METRIC_DICT
import sys, os, json, math
from utils_myadaptor import get_model_dir
logger = logging.getLogger(__name__)


def get_unique_sorted_adapter_array(adapter_function):
    uni_adapter = np.unique(adapter_function)   
    uni_adapter = np.array(sorted(uni_adapter.astype(int))).astype(str)
    return uni_adapter

def frange_cycle_linear(n_iter, start=0.01, stop=1.0,  n_cycle=4, ratio=0.5):
    """
        n_iter = self.args.num_train_epochs[curr_task] * len(train_loader)
        beta_list = frange_cycle_linear(n_iter, start=0.0, n_cycle=self.args.num_cycle, ratio=0.9)
        self.logger.info("Beta list we will use to train"+str(beta_list))
    """
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

def get_kl(p, q, mask, eps=1e-7):
    assert torch.sum(mask).item()>0
    logger.info('p: {}'.format(p))
    logger.info('q: {}'.format(q))
    current_kl_loss = torch.sum(
            F.kl_div(F.log_softmax(p, dim=1), F.softmax(q, dim=1), reduction='none'), 
            dim=1,)  
    logger.info('p_softmax: {}'.format(F.log_softmax(p, dim=1)))
    logger.info('q_softmax: {}'.format(F.softmax(q, dim=1)))
    logger.info('current_kl_loss: {}'.format(current_kl_loss))
    current_kl_loss =  torch.reciprocal(current_kl_loss+eps) * mask     
    batch_kl_loss = torch.sum(current_kl_loss)/torch.sum(mask)
    return batch_kl_loss


def get_cos(p, q, mask, eps=1e-7):
    assert torch.sum(mask).item()>0
    cos_sim = F.cosine_similarity(p, q, dim=1, eps=eps) * mask.squeeze()
    batch_cos_loss = torch.sum(cos_sim)/torch.sum(mask)
    return batch_cos_loss



def get_ce_and_vertical_kl_losses_v0(parallel_model, cqa, Y, gen_X, gen_Y, loss_fct, c_module_list):   
    # we have already change kl_div into cos distance;
    if "lll" in args.seq_train_type or "multilm" in args.seq_train_type or "llewc" in args.seq_train_type:
        qa_output = parallel_model(cqa, output_hidden_states=True)
        lm_output = parallel_model(gen_X, output_hidden_states=True)

        qa_logits = qa_output.logits
        lm_logits = lm_output.logits

        Y = Y.view(-1)
        qa_logits = qa_logits.view(Y.shape[0], -1)
        gen_Y = gen_Y.view(-1)
        lm_logits = lm_logits.view(gen_Y.shape[0], -1)

        qa_loss = loss_fct(qa_logits, Y)
        lm_loss = loss_fct(lm_logits, gen_Y)

        all_hidden_states = qa_output.hidden_states

        kl_loss_along_layers = []
        mask = Y != FILL_VAL
        mask = mask.unsqueeze(1)
        for layer_id in range(0, len(all_hidden_states)-1):
            hidden_dim = all_hidden_states[layer_id+1].shape[-1] 
            hidden_before = all_hidden_states[layer_id].view(-1, hidden_dim)
            hidden_after = all_hidden_states[layer_id+1].view(-1, hidden_dim)
            if c_module_list[layer_id] != '0':
                current_kl_loss = get_cos(hidden_after, hidden_before, mask)
                kl_loss_along_layers.append(current_kl_loss)
        kl_loss = args.cm_v_kl_lambda * torch.mean(torch.stack(kl_loss_along_layers))
        return torch.mean(qa_loss), args.lm_lambda * torch.mean(lm_loss), None, {'kl_loss': kl_loss }


def get_ce_and_vertical_kl_losses(parallel_model, cqa, Y, gen_X, gen_Y, loss_fct, c_module_list, typ=0, task_id=-1):  
    def get_shared_unshared_layers(c_module_list, task_id):
        shared_unshared_flag_list = [False] * len(c_module_list)           
        for layer_id, adapter_name in enumerate(c_module_list):
            if int(adapter_name) == task_id:
                shared_unshared_flag_list[layer_id] = True
        unshared_layer_id_list = []
        for layer_id, shared_unshared_flag in enumerate(shared_unshared_flag_list):
            if shared_unshared_flag: unshared_layer_id_list.append(layer_id)
        return shared_unshared_flag_list, unshared_layer_id_list

    def get_pair_type_1(unshared_layer_id_list):

        processing_pair_lst = []
        for idx, shared_layer_id in enumerate(unshared_layer_id_list):
            if idx==0: continue
            processing_pair_lst.append((unshared_layer_id_list[idx-1], shared_layer_id))
        return processing_pair_lst

    def get_pair_type_2(unshared_layer_id_list):   
        processing_pair_lst = []
        for idx, shared_layer_id in enumerate(unshared_layer_id_list):
            processing_pair_lst.append((shared_layer_id-1, shared_layer_id))
        return processing_pair_lst

    def get_pair_type_3(unshared_layer_id_list):
        processing_pair_lst = [(x, y) for x in unshared_layer_id_list for y in unshared_layer_id_list if x < y]
        return processing_pair_lst

    if "lll" in args.seq_train_type or "multilm" in args.seq_train_type or "llewc" in args.seq_train_type:
        qa_output = parallel_model(cqa, output_hidden_states=True)
        lm_output = parallel_model(gen_X, output_hidden_states=True)

        qa_logits = qa_output.logits
        lm_logits = lm_output.logits

        Y = Y.view(-1)
        qa_logits = qa_logits.view(Y.shape[0], -1)
        gen_Y = gen_Y.view(-1)
        lm_logits = lm_logits.view(gen_Y.shape[0], -1)

        qa_loss = loss_fct(qa_logits, Y)
        lm_loss = loss_fct(lm_logits, gen_Y)

        all_hidden_states = qa_output.hidden_states 

        mask = Y != FILL_VAL
        mask = mask.unsqueeze(1)
        if typ==0:
            kl_loss_along_layers = []
            for layer_id in range(0, len(all_hidden_states)-1):
                hidden_dim = all_hidden_states[layer_id+1].shape[-1] 
                hidden_before = all_hidden_states[layer_id].view(-1, hidden_dim)
                hidden_after = all_hidden_states[layer_id+1].view(-1, hidden_dim)
                if c_module_list[layer_id] != '0':
                    current_kl_loss = get_cos(hidden_after, hidden_before, mask)
                    kl_loss_along_layers.append(current_kl_loss)
            kl_loss = args.cm_v_kl_lambda * torch.mean(torch.stack(kl_loss_along_layers))
        else:
            kl_loss_items = []
            _, unshared_layer_id_list = get_shared_unshared_layers(c_module_list, task_id)
            processing_pair_lst = eval('get_pair_type_'+str(typ))(unshared_layer_id_list)
            for layer_id_before, layer_id_after in processing_pair_lst:
                hidden_dim = all_hidden_states[layer_id_after+1].shape[-1] 
                hidden_before = all_hidden_states[layer_id_before+1].view(-1, hidden_dim)
                hidden_after = all_hidden_states[layer_id_after+1].view(-1, hidden_dim)
                current_kl_loss = get_cos(hidden_after, hidden_before, mask)
                kl_loss_items.append(current_kl_loss)
            if len(kl_loss_items)>0:
                kl_loss = args.cm_v_kl_lambda * torch.mean(torch.stack(kl_loss_items))
            else: kl_loss = torch.tensor(0).to(cqa.device)
        return torch.mean(qa_loss), args.lm_lambda * torch.mean(lm_loss), None, {'kl_loss': kl_loss }


    
def get_ce_and_horizontal_kl_losses(parallel_model, cqa, Y, gen_X, gen_Y, loss_fct, return_ita=False):   
    ita = None
    if "lll" in args.seq_train_type or "multilm" in args.seq_train_type or "llewc" in args.seq_train_type:
        qa_output = parallel_model(cqa, output_hidden_states=True, return_all_hiddens_during_mix_for_each_block=True)
        if return_ita:
            lm_output, ita = parallel_model(gen_X, output_hidden_states=True, return_ita=return_ita,
                                            return_all_hiddens_during_mix_for_each_block=True,
                                            )
        else:
            lm_output = parallel_model(gen_X, output_hidden_states=True, return_all_hiddens_during_mix_for_each_block=True)

        qa_logits = qa_output.logits
        lm_logits = lm_output.logits

        Y = Y.view(-1)
        qa_logits = qa_logits.view(Y.shape[0], -1)

        gen_Y = gen_Y.view(-1)
        lm_logits = lm_logits.view(gen_Y.shape[0], -1)

        qa_loss = loss_fct(qa_logits, Y)
        lm_loss = loss_fct(lm_logits, gen_Y)

        all_hidden_states_list = qa_output.all_hidden_states_list

        kl_loss_tasks_layers = {}
        mask = Y != FILL_VAL
        mask = mask.unsqueeze(1)
        layer_num = len(all_hidden_states_list)
        for layer_id in range(layer_num):     
            kl_loss_tasks_attention = []
            kl_loss_tasks_output = []
            hidden_states_list_attention = all_hidden_states_list[layer_id]['hidden_states_list_attention']
            hidden_states_list_output = all_hidden_states_list[layer_id]['hidden_states_list_output']
            assert len(hidden_states_list_attention) == len(hidden_states_list_output)
            adapter_num = len(hidden_states_list_attention)
            adapter_hidden_attention = hidden_states_list_attention[-1]
            adapter_hidden_output = hidden_states_list_output[-1]
            hidden_dim = adapter_hidden_output.shape[-1]
            adapter_hidden_attention = adapter_hidden_attention.view(-1, hidden_dim)
            adapter_hidden_output = adapter_hidden_output.view(-1, hidden_dim)
            for adapter_id in range(adapter_num-1):
                previous_adapter_hidden_attention = hidden_states_list_attention[adapter_id].view(-1, hidden_dim)
                previous_adapter_hidden_output = hidden_states_list_output[adapter_id].view(-1, hidden_dim)
                output_kl = get_cos(adapter_hidden_output, previous_adapter_hidden_output, mask)   
                kl_loss_tasks_output.append(output_kl)  
            kl_loss_tasks_layers[str(layer_id)] = {'kl_loss_tasks_attention': kl_loss_tasks_attention,
                                                        'kl_loss_tasks_output': kl_loss_tasks_output}
        all_kl = []
        for layer_id in range(layer_num):
            all_kl+=kl_loss_tasks_layers[str(layer_id)]['kl_loss_tasks_attention']
            all_kl+=kl_loss_tasks_layers[str(layer_id)]['kl_loss_tasks_output']
        kl_loss = args.cm_h_kl_lambda * torch.mean(torch.stack(all_kl))
        return torch.mean(qa_loss), args.lm_lambda * torch.mean(lm_loss), ita, kl_loss






def load_teacher_model(task_load):
    model_dir = get_model_dir([task_load])
    model_path = os.path.join(model_dir, 'model-finish')
    config_path = os.path.join(model_dir,CONFIG_NAME)

    model_config = CONFIG_CLASS.from_json_file(config_path)
    model = MODEL_CLASS(model_config).cuda()
    model.resize_token_embeddings(50260 + len(args.tasks))   

    adapter_list = np.load(os.path.join(model_dir, "adapter_list.npy"))
    model.add_adapter_by_list(adapter_list, config=args.adapt_type)  
    model.config.forward_mode = 'Transfer'

    uni_list = get_unique_sorted_adapter_array(adapter_list)
    uni_list = uni_list.tolist()

    if not args.adapterdrop:               
        model.train_adapter(uni_list)
    else:
        model.train_adapter(uni_list, [0, 1, 2])

    state_dict = torch.load(model_path, map_location='cuda:0')
    m, n = model.load_state_dict(state_dict, strict=False)
    logger.info("Missing : {}, Unexpected: {}".format(m, n))
    model.cuda()
    model.eval()

    if not args.fp32:
        logger.warning('The teacher model should be fp32')

    model.ep = 0
    model.model_dir = model_dir
    model.config.testing = True


    return model

def cm_load_one_model(task_load):   
    model_dir = get_model_dir([task_load])
    model_path = os.path.join(model_dir, 'model-finish')
    config_path = os.path.join(model_dir,CONFIG_NAME)

    model_config = CONFIG_CLASS.from_json_file(config_path)
    model = MODEL_CLASS(model_config).cuda()
    model.resize_token_embeddings(50260 + len(args.tasks))   

    adapter_list = np.load(os.path.join(model_dir, "adapter_list.npy"))
    model.add_adapter_by_list(adapter_list, config=args.adapt_type)  
    model.config.forward_mode = 'Transfer'

    uni_list = get_unique_sorted_adapter_array(adapter_list)
    uni_list = uni_list.tolist()

    if not args.adapterdrop:              
        model.train_adapter(uni_list)
    else:
        model.train_adapter(uni_list, [0, 1, 2])

    state_dict = torch.load(model_path, map_location='cuda:0')
    m, n = model.load_state_dict(state_dict, strict=False)
    logger.info("Missing : {}, Unexpected: {}".format(m, n))
    model.cuda()
    model.train()

    if not args.fp32:
        logger.warning('The teacher model should be fp32')

    model.ep = 0
    model.model_dir = model_dir
    model.config.testing = False

    return model



def get_kd_losses(parallel_model, cqa, Y, gen_X, gen_Y, loss_fct_ce, loss_fct_kd, teacher_model, return_ita=False):
    ita = None
    if "lll" in args.seq_train_type or "multilm" in args.seq_train_type or "llewc" in args.seq_train_type:
        qa_output = parallel_model(cqa)
        if return_ita:
            lm_output, ita = parallel_model(gen_X, return_ita=return_ita)
        else:
            lm_output = parallel_model(gen_X)
        with torch.no_grad():
            teacher_output = teacher_model(gen_X)

        qa_logits = qa_output.logits
        lm_logits = lm_output.logits
        teacher_logits = teacher_output.logits

        Y = Y.view(-1)
        qa_logits = qa_logits.view(Y.shape[0], -1)
        gen_Y = gen_Y.view(-1)
        lm_logits = lm_logits.view(gen_Y.shape[0], -1)
        teacher_logits = teacher_logits.view(gen_Y.shape[0], -1)

        qa_loss = loss_fct_ce(qa_logits, Y)
        lm_kd_loss = loss_fct_kd(lm_logits, gen_Y, teacher_logits)   

        if return_ita:
            return torch.mean(qa_loss), args.lm_lambda * torch.mean(lm_kd_loss), ita
        else:
            return torch.mean(qa_loss), args.lm_lambda * torch.mean(lm_kd_loss)
    else:
        pass

class KDLoss(nn.Module):
    def __init__(self, KD_term=0.0, T=1.0, ignore_index=None, weight=None):   # 0.5 2
        super(KDLoss, self).__init__()
        assert 0 <= KD_term <=1
        assert 0 < T
        self.KD_term = KD_term
        self.T = T
        self.ignore_index = ignore_index
        self.weight = weight
            
    def forward(self, output_logits, targets, teacher_logits=None):
        if teacher_logits is None:
            return F.cross_entropy(output_logits, targets, reduction='none',
                    ignore_index=self.ignore_index, weight=self.weight)
        else: 
            KD_loss = F.kl_div(F.log_softmax(output_logits / self.T, dim=1), 
                F.softmax(teacher_logits / self.T, dim=1), reduction='none')
            KD_loss = torch.sum(KD_loss, dim=1)
            CE_loss = F.cross_entropy(output_logits, targets, reduction='none',
                                      ignore_index=self.ignore_index, weight=self.weight)
            return KD_loss * self.KD_term * self.T * self.T + CE_loss * (1 - self.KD_term)   

