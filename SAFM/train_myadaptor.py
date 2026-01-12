import torch
from torch.utils.data import DataLoader
from torch import nn
from mytransformers import AdamW, WEIGHTS_NAME, HoulsbyConfig, get_constant_schedule_with_warmup
import csv
import random
import numpy as np
import os
import copy
import logging
from fp16 import FP16_Module, FP16_Optimizer
from parallel import DataParallelModel, DataParallelCriterion
from collections import OrderedDict
from utils_myadaptor import *
from settings_myadaptor import args, TASK_DICT, init_logging, MODEL_CONFIG, MODEL_CLASS, SPECIAL_TOKENS, CONFIG_CLASS
from settings_myadaptor import TOKENIZER, SPECIAL_TOKEN_IDS, FILL_VAL, SAVE_NAME, FINAL_SAVE_NAME, TOKENS_WEIGHT, CONFIG_NAME
from scheduler import AnnealingLR
from regularizers_myadaptor import REG_TYPES, REG_TYPE_KEYS, Weight_Regularized_AdamW, Weight_Regularized_SGD
from torch.nn import CrossEntropyLoss
import pdb
logger = logging.getLogger(__name__)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

from cm_losses import get_kd_losses, KDLoss, get_ce_and_vertical_kl_losses, get_ce_and_horizontal_kl_losses, \
                    load_teacher_model, get_unique_sorted_adapter_array, cm_load_one_model

def load_old_adapter(model, newname, oldname):
    state_dict = model.state_dict()
    new_state_dict = OrderedDict()
    for i in state_dict:
        if oldname in i:
            new_i = i.replace(oldname, newname)
            new_state_dict[new_i] = state_dict[i].clone().detach()
    m, n = model.load_state_dict(new_state_dict, strict=False)
    logger.info("Load old adapter weight to new adapter weight, Unexpected: {}".format(n))

def load_pre_adapter(model, newname):
    pre_state_dict = torch.load(os.path.join('./PModel/model-pretrain'))
    new_state_dict = OrderedDict()
    for i in pre_state_dict:
        if "pretrain" in i:
            new_i = i.replace("pretrain", newname)
            new_state_dict[new_i] = pre_state_dict[i].clone().detach()
            logger.info("Load from {} to {}".format(i, new_i))
    m, n = model.load_state_dict(new_state_dict, strict=False)
    logger.info("Load old adapter weight to new adapter weight, Unexpected: {}".format(n))

def cal_entropy_loss(ita):
    entropy_loss = torch.tensor(0.0, device="cuda")
    for item in ita:
        item = item / args.select_temp

        dis = torch.nn.functional.softmax(item, dim=0)
        entropy_loss += args.last_dim_coe * (dis[-1][0] - 0.0) ** 2

        log_dis = torch.log(dis)
        entropy_loss += - torch.sum(dis * log_dis)
    return entropy_loss


def freeze_for_mix(model):
    for name, param in model.named_parameters():
        if "ita" not in name: param.requires_grad = False
        else: param.requires_grad = True


def learnable_para_calculate(model, note, printname=False):
    learn_sum = 0
    else_sum = 0
    logger.info("Para requries gradient...")
    param_opt = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_opt.append((name, param))
            if printname: logger.info(name)
            learn_sum += param.nelement()
        else:
            else_sum += param.nelement()
            # """
            if "ita" in name:
                param_opt.append((name, param))
            # """
    logger.info(note + " Number of learned parameter: %.2fM" % (learn_sum/1e6))
    logger.info(note + " Number of else parameter: %.2fM" % (else_sum/1e6))
    logger.info(note + " Ratio: {}".format(1.0 * (learn_sum + else_sum) / else_sum))
    return param_opt


def print_para(model):
    logger.info("Print para")
    printted = [False, False, False, False, False]
    for name, param in model.named_parameters():
        for i in range(5):
            if "adapters." + str(i) in name and not printted[i]:
                logger.info(name)
                logger.info(param)
                printted[i] = True


def swap_name(org_name, seq_distil, ref1):
    if not seq_distil and not ref1: return org_name
    if seq_distil: return org_name.replace("train", "distil")
    if ref1: return org_name.replace("train", "ref1")


def validation(model, valid_dataloader, train_loss_fct):
    cum_loss = 0.0
    cum_qa_loss = 0.0
    cum_lm_loss = 0.0
    cur_n_inputs = 0
    with torch.no_grad():
        model.eval()
        for (_, _, cqa, _, Y, gen_X, gen_Y, task_id, idx) in valid_dataloader:
            torch.cuda.empty_cache()
            n_inputs = cqa[0].shape[0]
            model.config.batch_task_id = task_id[0][0].item()
            qa_loss, lm_loss = get_losses(model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(), train_loss_fct)

            cum_loss += (qa_loss + lm_loss) * n_inputs
            cum_qa_loss += qa_loss * n_inputs
            cum_lm_loss += lm_loss * n_inputs
            cur_n_inputs += n_inputs
    if cur_n_inputs==0:
        logger.warning('line 141 ERROR in calculating loss.')
        return 0, 0, 0
    return cum_loss / cur_n_inputs, cum_qa_loss / cur_n_inputs, cum_lm_loss / cur_n_inputs

def clear(model):
    old = model
    model = copy.deepcopy(old)
    del old
    torch.cuda.empty_cache()
    return model


def load_model(model_dir):
    from mytransformers import GPT2LMHeadModel, GPT2Config

    model_config = GPT2Config.from_json_file(os.path.join(model_dir, "config.json"))
    model = GPT2LMHeadModel(model_config)
    model.resize_token_embeddings(50260 + len(args.tasks))

    adapter_list = np.load(os.path.join(model_dir, "adapter_list.npy"))
    model.add_adapter_by_list(adapter_list, config=args.adapt_type)
    state_dict = torch.load(os.path.join(model_dir, "model-finish"), map_location='cuda:0')
    m, n = model.load_state_dict(state_dict, strict=False)
    logger.info("Missing : {}, Unexpected: {}".format(m, n))
    model.cuda()

    return model


def Mix(task_ids, model):


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = [args.tasks[task_id] for task_id in task_ids]

    logger.info("start to Mix { task: %s, seq train type: %s }" % (tasks, args.seq_train_type))
    model_dir = get_model_dir(tasks)
    make_dir(model_dir)

    train_dataset = [swap_name(TASK_DICT[t]["train"], args.seq_distil, args.ref1) for t in tasks]
    valid_dataset = [TASK_DICT[t]["test"] for t in tasks]

    if args.load_model_for_stage:
        prev_tasks = [args.tasks[task_ids[0]-1]]
        prev_model_dir = get_model_dir(prev_tasks)
        model = load_model(prev_model_dir)
    else:
        logger.info("line179")
        prev_tasks = [args.tasks[task_ids[0]-1]]
        logger.info("prev_tasks: {}".format(prev_tasks) )
        prev_model_dir = get_model_dir(prev_tasks)
        logger.info("prev_model_dir: {}".format(prev_model_dir) )
        load_model(prev_model_dir)

    logger.info('Line 189: In the mixing stage, we are adding new adapters for task {}'.format(task_ids))
    model.config.forward_mode = 'Mix'
    model.config.testing = False
    model.config.mix_ini = args.mix_ini
    model.config.mu = args.cm_mu
    model.add_adapter(str(task_ids[0]), config=args.adapt_type, typ='MLP')
    if args.pretrain_adapter:       
        load_pre_adapter(model, str(task_ids[0]))
    model.train_adapter(str(task_ids[0]))     
    model.cuda()
    adapter_list = model.get_adapter_list()
    logger.info('adapter_list: {}'.format(adapter_list))
    logger.info('Line 200: In the mixing stage, adding new adapters done for task {}'.format(task_ids))

    if args.clear_model:
        model = clear(model)

    param_opt = learnable_para_calculate(model, "whole", True)  

    gen_token = get_gen_token(tasks[0])
    TOKENIZER.add_tokens([gen_token])
    TOKENIZER.save_pretrained(model_dir)
    SPECIAL_TOKENS[tasks[0]] = gen_token
    SPECIAL_TOKEN_IDS[tasks[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
    logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[tasks[0]]))
    MODEL_CONFIG.vocab_size = len(TOKENIZER)
    MODEL_CONFIG.to_json_file(os.path.join(model_dir,CONFIG_NAME))
    global TOKENS_WEIGHT
    logger.info("TOKENS_WEIGHT: {}, {}".format(TOKENS_WEIGHT, TOKENS_WEIGHT.shape,TOKENS_WEIGHT.shape[0]) )
    while 50260 + len(args.tasks) != TOKENS_WEIGHT.shape[0]:
        TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
        logger.info("Add one dim weight!")

    if not args.fp32: 
        model = FP16_Module(model)

    logger.warning("Adapter Mix test, not using extra data now...")
    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]])
    valid_qadata = QADataset(valid_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]])

    max_train_batch_size = args.z_max_batch_size

    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
    if args.cm_test_batch_size != -1:
        valid_dataloader = create_dataloader(valid_qadata, "test", args.cm_test_batch_size)
    else:
        valid_dataloader = create_dataloader(valid_qadata, "test")

    n_train_epochs = args.whole_mix_step
    n_train_optimization_steps = len(train_qadata) * n_train_epochs
    logger.info('len of train dataset: {} , max train batch size {} , num of opt steps: {}'.format(
        len(train_qadata), max_train_batch_size, n_train_optimization_steps))

    if args.whole_optim: param_optimizer = list(model.named_parameters())
    else: param_optimizer = param_opt

    no_decay = ['bias', 'ln_1', 'ln_2', 'ln_f']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    logger.info("USE ARGS.ADAM_EPSILON NOW.....")
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_constant_schedule_with_warmup(optimizer, args.z_warmup_step)

    train_loss_fct = CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT.type(torch.float if args.fp32 else torch.half))

    ita = None
    tot_n_steps = 0
    train_once = TrainStep(model, optimizer, scheduler)

    mix_flag = 0
    
    logger.info('Line 266: n_train_epochs :{}'.format(n_train_epochs))
    adapter_list = model.get_adapter_list()
    logger.info('Lin 268: adapter_list: {}'.format(adapter_list))

    for ep in range(n_train_epochs):
        model.train()
        cum_loss, cum_qa_loss, cum_lm_loss, cur_n_inputs = 0, 0, 0, 0
        cum_en_loss = 0
        cm_kl_loss = 0
        for n_steps, (_, _, cqa, _, Y, gen_X, gen_Y, task_id, idx) in enumerate(train_dataloader):
            n_inputs = cqa[0].shape[0]
            lens = cqa[0].shape[1]
            if lens > 500:
                logger.info(cqa[0].shape)
                continue

            model.config.batch_task_id = task_id[0][0].item()
            if args.cm_kl_h_flag and ep >= args.cm_h_kl_ep-1:
                losses = get_ce_and_horizontal_kl_losses(model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(), 
                                                                                                            train_loss_fct, True)
            else:
                losses = get_losses(model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(), train_loss_fct, True)

            if losses[1].item() == 0: loss = losses[0]
            else: loss = losses[0] + losses[1]

            if args.cm_kl_h_flag and ep >= args.cm_h_kl_ep-1: loss= loss+ losses[3]

            if args.mix_loss_norm and model.config.forward_mode == 'Mix':
                loss /= loss.item()
                loss *= args.mix_loss_coe

            ita = losses[2]
            en_loss = torch.tensor(0.)
            if task_ids[0] > 0 and model.config.forward_mode == 'Mix':
                en_loss = cal_entropy_loss(ita)
                loss += en_loss * args.entropy_coe

            train_once(loss, n_inputs)
            qa_loss = losses[0].item() * n_inputs
            lm_loss = losses[1].item() * n_inputs
            cum_loss += loss.item() * n_inputs
            cum_en_loss += en_loss.item() * args.entropy_coe * n_inputs
            cum_qa_loss += qa_loss
            cum_lm_loss += lm_loss
            cur_n_inputs += n_inputs

            if args.cm_kl_h_flag : 
                cm_kl_loss += losses[3].item() * n_inputs

            if args.constant_sch or task_ids[0] > 0: lr = scheduler.get_lr()[0]
            else: lr = scheduler.get_lr()
            if cur_n_inputs==0:
                logger.info('ep:{}, n_steps: {}, cur_n_inputs: {}'.format(ep, n_steps, cur_n_inputs))
                continue
            if (n_steps + 1) % args.logging_steps == 0:
                logger.info('progress {:.3f} , lr {:.1E} , loss {:.3f} , qa loss {:.3f} , lm loss {:.3f} , en loss {:.3f}, avg batch size {:.1f}'
                            .format(ep + cur_n_inputs/len(train_qadata),
                                    lr, cum_loss/cur_n_inputs,
                                    cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs,
                                    cum_en_loss/cur_n_inputs, cur_n_inputs/(n_steps + 1)))
        if cur_n_inputs==0:
            logger.info('ep:{}, cur_n_inputs: {}'.format(ep, cur_n_inputs))
            continue
        if not args.gradient_debug:
            tot_n_steps += (n_steps + 1)
            val_loss, val_qa_loss, val_lm_loss = validation(model, valid_dataloader, train_loss_fct)
            logger.info('epoch {}/{} done , tot steps {} , loss {:.2f} , qa loss {:.2f} , lm loss {:.2f}, val loss {:.2f}, vqa loss {:.2f}, vlm loss {:.2f}, avg batch size {:.1f}'.format(
                ep+1, n_train_epochs, tot_n_steps,
                cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, 
                cum_lm_loss/cur_n_inputs, val_loss,
                val_qa_loss, val_lm_loss, cur_n_inputs/(n_steps+1)
            ))
        logger.info("ITA:")
        logger.info(ita)

        print_para(model)
        if args.gradient_debug: exit(0)

        if ep == args.warm_mix_step - 1:                        
            logger.info('Line 341: Training Lambda!')
            model.config.forward_mode = 'Mix'
            for name, param in model.named_parameters():
                if "ita" in name: param.requires_grad = True

    if args.layer_debug:
        for i, layer_ita in enumerate(ita):
            if i == args.layer_debug_cnt:
                layer_ita[1] = 1.0

    logger.info('Line 348: Make decision on which adapter to use for the new task (in each layer)')
    cnt_true = model.setup_task_adapter(task_ids[0], args.cm_unshared_adapter_layer_list) 
    logger.info('Line350: [Done] Make decision on which adapter to use for the new task (in each layer)')

    if cnt_true > 0: fit_or_not = True
    else: fit_or_not = False

    current_fit_epoch = None
    trans = True

    torch.save(model.state_dict(), os.path.join(model_dir, SAVE_NAME+"finish"))
    adapter_list = model.get_adapter_list()
    np.save(os.path.join(model_dir, "adapter_list.npy"), adapter_list)
    logger.info("MODEL SAVED!")

    del optimizer
    del scheduler
    torch.cuda.empty_cache()
    logger.info("line355: {}, {} {}".format(fit_or_not, trans, current_fit_epoch) )
    return model, fit_or_not, trans, current_fit_epoch

def Fit(task_ids, model, current_fit_epoch=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = [args.tasks[task_id] for task_id in task_ids]

    logger.info("start to Fit { task: %s, seq train type: %s }" % (tasks, args.seq_train_type))
    model_dir = get_model_dir(tasks)

    train_dataset = [swap_name(TASK_DICT[t]["train"], args.seq_distil, args.ref1) for t in tasks]
    valid_dataset = [TASK_DICT[t]["test"] for t in tasks]

    if args.load_model_for_stage:
        model = load_model(model_dir)
    else:
        load_model(model_dir)

    model.config.forward_mode = 'Fit'
    model.config.testing = False
    model.train_adapter(str(task_ids[0]))
    model.cuda()
    if args.clear_model:
        model = clear(model)
    param_opt = learnable_para_calculate(model, "whole", True)

    gen_token = get_gen_token(tasks[0])
    TOKENIZER.add_tokens([gen_token])
    TOKENIZER.save_pretrained(model_dir)
    SPECIAL_TOKENS[tasks[0]] = gen_token
    SPECIAL_TOKEN_IDS[tasks[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
    logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[tasks[0]]))
    MODEL_CONFIG.vocab_size = len(TOKENIZER)
    MODEL_CONFIG.to_json_file(os.path.join(model_dir,CONFIG_NAME))
    global TOKENS_WEIGHT
    while 50260 + len(args.tasks) != TOKENS_WEIGHT.shape[0]:
        TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
        logger.info("Add one dim weight!")

    if not args.fp32:  
        model = FP16_Module(model)

    logger.warning("In Fit, not using extra data now...")
    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]])
    valid_qadata = QADataset(valid_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]])

    max_train_batch_size = args.z_max_batch_size
    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
    if args.cm_test_batch_size != -1:
        valid_dataloader = create_dataloader(valid_qadata, "test", args.cm_test_batch_size)
    else:
        valid_dataloader = create_dataloader(valid_qadata, "test")

    n_train_epochs = args.fit_epoch
    if current_fit_epoch != None:
        n_train_epochs = current_fit_epoch

    n_train_optimization_steps = len(train_qadata) * n_train_epochs
    logger.info('len of train dataset: {} , max train batch size {} , num of opt steps: {}'.format(
        len(train_qadata), max_train_batch_size, n_train_optimization_steps))

    if args.whole_optim:
        param_optimizer = list(model.named_parameters())
    else:
        param_optimizer = param_opt

    no_decay = ['bias', 'ln_1', 'ln_2', 'ln_f']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    logger.info("USE ARGS.ADAM_EPSILON NOW.....")
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_constant_schedule_with_warmup(optimizer, args.z_warmup_step)
    train_loss_fct = CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT.type(torch.float if args.fp32 else torch.half))

    tot_n_steps = 0
    train_once = TrainStep(model, optimizer, scheduler)

    for ep in range(n_train_epochs):
        model.train()
        cum_loss, cum_qa_loss, cum_lm_loss, cur_n_inputs = 0, 0, 0, 0
        for n_steps, (_, _, cqa, _, Y, gen_X, gen_Y, task_id, idx) in enumerate(train_dataloader):
            n_inputs = cqa[0].shape[0]
            lens = cqa[0].shape[1]
            if lens > 500:
                logger.info(cqa[0].shape)
                continue

            model.config.batch_task_id = task_id[0][0].item()
            losses = get_losses(model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(), train_loss_fct)

            if losses[1].item() == 0: loss = losses[0]
            else: loss = losses[0] + losses[1]

            train_once(loss, n_inputs)

            qa_loss = losses[0].item() * n_inputs
            lm_loss = losses[1].item() * n_inputs
            cum_loss += (qa_loss + lm_loss)
            cum_qa_loss += qa_loss
            cum_lm_loss += lm_loss
            cur_n_inputs += n_inputs

            if args.constant_sch or task_ids[0] > 0: lr = scheduler.get_lr()[0]
            else: lr = scheduler.get_lr()

            if (n_steps + 1) % args.logging_steps == 0:
                logger.info('progress {:.3f} , lr {:.1E} , loss {:.3f} , qa loss {:.3f} , lm loss {:.3f}, avg batch size {:.1f}'
                            .format(ep + cur_n_inputs/len(train_qadata),
                                    lr, cum_loss/cur_n_inputs,
                                    cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs,
                                    cur_n_inputs/(n_steps + 1)))

        if not args.gradient_debug:
            tot_n_steps += (n_steps + 1)
            val_loss, val_qa_loss, val_lm_loss = validation(model, valid_dataloader, train_loss_fct)
            logger.info('epoch {}/{} done , tot steps {} , loss {:.2f} , qa loss {:.2f} , lm loss {:.2f}, val loss {:.2f}, vqa loss {:.2f}, vlm loss {:.2f}, avg batch size {:.1f}'.format(
                ep+1, n_train_epochs, tot_n_steps,
                cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs,
                cum_lm_loss/cur_n_inputs, val_loss,
                val_qa_loss, val_lm_loss, cur_n_inputs/(n_steps+1)
            ))

        print_para(model)

    torch.save(model.state_dict(), os.path.join(model_dir, SAVE_NAME+"finish"))
    adapter_list = model.get_adapter_list()
    np.save(os.path.join(model_dir, "adapter_list.npy"), adapter_list)
    logger.info("MODEL SAVED!")

    del optimizer
    del scheduler
    torch.cuda.empty_cache()

    return model

def Transfer(task_ids, model, fit_bonus=0):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
       
    tasks = [args.tasks[task_id] for task_id in task_ids]

    logger.info("start to transfer { task: %s, seq train type: %s }" % (tasks, args.seq_train_type))
    model_dir = get_model_dir(tasks)

    train_dataset = [swap_name(TASK_DICT[t]["train"], args.seq_distil, args.ref1) for t in tasks]
    logger.info('train_dataset :{}'.format(train_dataset))
    valid_dataset = [TASK_DICT[t]["test"] for t in tasks]
    train_extra_data = []

    c_module_list = []
    if task_ids[0]>0:
        adapter_list = np.load(os.path.join(model_dir, "adapter_list.npy"))
        logger.info("adapter_list[line524]: {} {}".format(adapter_list, len(adapter_list)) )
        for layer_id, layer_list in enumerate(adapter_list):
            c_module_list.append(layer_list[task_ids[0]])
        logger.info("c_module_list: {}".format(c_module_list))
      
    if not args.generate_after and args.cm_lamol_flag:
        logger.info("task_ids[line521]: {}  {}".format(task_ids, len(task_ids)))
        if ("lll" in args.seq_train_type or "llewc" in args.seq_train_type) and task_ids[0] > 1 and not args.pseudo_ablation:
            adapter_list = np.load(os.path.join(model_dir, "adapter_list.npy"))
            logger.info("adapter_list[line524]: {} {}".format(adapter_list, len(adapter_list)) )
            replay = []
            for layer_id, layer_list in enumerate(adapter_list):
                logger.info("layer_list: {} {}".format(layer_list, len(layer_list)))
                c_module = layer_list[task_ids[0]]     
                if c_module != '0':     
                    logger.info("For current task, in layer {}, c_module: {}".format(layer_id, c_module))
                    for i in range(1, task_ids[0]):
                        if layer_list[i] == c_module:
                            if i not in replay:
                                replay.append(i)
            logger.info("replay tasks (task ids): {}".format(replay))

            if len(replay) > 0:
                prev_task = args.tasks[task_ids[0]-1]
                model.config.forward_mode = 'Transfer'
                model.config.testing = False
                with torch.no_grad():
                    create_extra_data(tasks[0], prev_task, model, train_extra_data, None, None, replay)
        logger.info('extra training data size: {}'.format(len(train_extra_data)))

    
    logger.info('model: {}'.format(model))
    if model is None:
        assert(task_ids[0]==0)

        model = MODEL_CLASS.from_pretrained('gpt2')
        model_dir = get_model_dir(tasks)
        make_dir(model_dir)


        torch.manual_seed(42)
        model.resize_token_embeddings(50260 + len(args.tasks))
        logger.info(model.transformer.wte.weight)
        torch.manual_seed(args.seed)
        model.config.forward_mode = 'Transfer'
        model.config.testing = False
        model.add_adapter('0', config=args.adapt_type, typ='NULL')

        if args.pretrain_adapter:
            logger.info("line567, load")
            load_pre_adapter(model, str(task_ids[0]))

        if not args.adapterdrop:
            logger.info("line571: {}".format(str(task_ids[0])) )
            model.train_adapter(str(task_ids[0]))
        else:
            model.train_adapter(str(task_ids[0]), [0, 1, 2])
        model.cuda()
        if args.clear_model:
            model = clear(model)

        param_opt = learnable_para_calculate(model, "whole", True)
        if not args.fp32:
            logger.info("Not support fp32 on mytransformers/adapters now...")
            exit(0)
    else:                      
        if args.load_model_for_stage:
            prev_tasks = [args.tasks[task_ids[0]-1]]
            prev_model_dir = get_model_dir(prev_tasks)
            model = load_model(prev_model_dir)

            adapter_list = np.load(os.path.join(model_dir, "adapter_list.npy"))
            model.update_adapter_list(adapter_list)
        else:
            load_model(model_dir)

        model.config.forward_mode = 'Transfer'
        model.config.testing = False

        if args.partial_learn:
            model.train_adapter(str(task_ids[0]))
        elif args.partial_transfer:
            model.adapter_transfer()         
        else:
            adapter_list = np.load(os.path.join(model_dir, "adapter_list.npy"))
            adapter_names = get_unique_sorted_adapter_array(adapter_list)

            logger.info("line 653 {}".format(adapter_names))
            logger.info("line 654 {}".format(adapter_list))
            model.train_adapter([str(i) for i in adapter_names])    
        model.cuda()
        if args.clear_model:
            model = clear(model)

        param_opt = learnable_para_calculate(model, "whole", True)

    if args.generate_after and args.cm_lamol_flag:  
        if ("lll" in args.seq_train_type or "llewc" in args.seq_train_type) and task_ids[0] > 1 and not args.pseudo_ablation:
            prev_task = args.tasks[task_ids[0]-1]
            model.config.forward_mode = 'Transfer'
            model.config.testing = False
            with torch.no_grad():
                create_extra_data(tasks[0], prev_task, model, train_extra_data)

        logger.info('extra training data size: {}'.format(len(train_extra_data)))

    gen_token = get_gen_token(tasks[0])
    TOKENIZER.add_tokens([gen_token])               
    TOKENIZER.save_pretrained(model_dir)
    SPECIAL_TOKENS[tasks[0]] = gen_token
    SPECIAL_TOKEN_IDS[tasks[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
    logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[tasks[0]]))
    MODEL_CONFIG.vocab_size = len(TOKENIZER)
    MODEL_CONFIG.to_json_file(os.path.join(model_dir,CONFIG_NAME))
    global TOKENS_WEIGHT
    while 50260 + len(args.tasks) != TOKENS_WEIGHT.shape[0]:
        TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
        logger.info("Add one dim weight!")

    if not args.fp32: 
        model = FP16_Module(model)

    logger.warning("Transfer, using extra data now...")
    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
    valid_qadata = QADataset(valid_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]])
    max_train_batch_size = args.z_max_batch_size
    
    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)   
    if args.cm_test_batch_size != -1:
        valid_dataloader = create_dataloader(valid_qadata, "test", args.cm_test_batch_size)
    else:
        valid_dataloader = create_dataloader(valid_qadata, "test")

    if args.gradient_debug:
        n_train_epochs = 1
    elif task_ids[0] == 0:
        n_train_epochs = args.z_train_epochs[task_ids[0]]
    else:
        n_train_epochs = args.z_train_epochs[task_ids[0]] - fit_bonus

    n_train_optimization_steps = len(train_qadata) * n_train_epochs
    logger.info('len of train dataset: {} , max train batch size {} , num of opt steps: {}'.format(
        len(train_qadata), max_train_batch_size, n_train_optimization_steps))

    if args.whole_optim:                 
        param_optimizer = list(model.named_parameters())
    else:
        param_optimizer = param_opt      

    if args.deactivate_reuse_adapter:
        for n, p in param_optimizer:
            if 'attention_adapters.adapters.'+str(task_ids[0]) in n: 
                p.requires_grad = True
            elif 'output_adapters.adapters.'+str(task_ids[0]) in n: 
                p.requires_grad = True
            else:
                p.requires_grad = False 

    finetune_flag = False
    for n, p in param_optimizer:
        if p.requires_grad == True:
            finetune_flag = True                             
    if not finetune_flag:
        n_train_epochs = 0

    no_decay = ['bias', 'ln_1', 'ln_2', 'ln_f']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer_grouped_names = [
        [n for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad==True],
        [n for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad==True]
    ]
    logger.info("name group")
    logger.info(optimizer_grouped_names)

    logger.info("USE ARGS.ADAM_EPSILON NOW.....")
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.z_train_lrs[task_ids[0]], eps=args.adam_epsilon)

    if args.constant_sch:
        logger.info("Start to use constant scheduler!")
        scheduler = get_constant_schedule_with_warmup(optimizer, args.z_warmup_step)
    elif not args.constant_sch and (not args.lamaml or (args.lamaml and task_ids[0] == 0)):
        logger.info("Start to use Annealling scheduler!")
        scheduler = AnnealingLR(optimizer, start_lr=args.z_train_lrs[task_ids[0]], warmup_iter=int(args.n_warmup_ratio*len(train_qadata)), 
                                num_iters=int(n_train_optimization_steps), decay_style=args.decay_style)
    elif not args.constant_sch:
        logger.info("Start to use Annealling scheduler!")
        scheduler = AnnealingLR(optimizer, start_lr=args.z_train_lrs[task_ids[0]], warmup_iter=int(args.n_warmup_ratio*len(train_qadata)), 
                                num_iters=int(train_qadata.get_c_len() * 2 * n_train_epochs) + 100, decay_style=args.decay_style)
    
    if args.cm_kd==True:
        if task_ids[0] > 1:              
            train_loss_fct_kd = KDLoss(KD_term=args.cm_kd_term, T=args.cm_kd_temperature, 
                                            ignore_index=FILL_VAL, weight=TOKENS_WEIGHT.type(torch.float if args.fp32 else torch.half))
            train_loss_fct_ce = CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT.type(torch.float if args.fp32 else torch.half))
            teacher_model = load_teacher_model([args.tasks[task_ids[0]-1]])
        else: 
            train_loss_fct = CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT.type(torch.float if args.fp32 else torch.half))
    elif args.cm_kl_v_flag == True:
        train_loss_fct = CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT.type(torch.float if args.fp32 else torch.half))
    else:
        train_loss_fct = CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT.type(torch.float if args.fp32 else torch.half)) 
    tot_n_steps = 0
    train_once = TrainStep(model, optimizer, scheduler)

    path = [None for i in range(task_ids[0] + 1)]          
    if task_ids[0] > 0:
        path = []
        o_path = model.get_path()
        for i in range(task_ids[0] + 1):
            c_name = []   
            for layer, j in enumerate(o_path):
                c_layer_name = j[i]
                c_name.append('.' + str(layer) + '.attention_adapters.adapters.' + c_layer_name + '.')
                c_name.append('.' + str(layer) + '.output_adapters.adapters.' + c_layer_name + '.')
            path_one = []        
            path_two = []        
            for n, p in param_optimizer:
                if not any(nd in n for nd in no_decay):
                    flag = 0
                    for name in c_name:    
                        if name in n:
                            flag = 1
                            path_one.append(True)
                            break
                    if flag == 0:
                        path_one.append(False)
                else:
                    flag = 0
                    for name in c_name:
                        if name in n:
                            flag = 1
                            path_two.append(True)
                            break
                    if flag == 0:
                        path_two.append(False)
            path.append([path_one, path_two])
        logger.info(path)

        shared = []                   
        for i, c_path in enumerate(path):
            if True in c_path[0] or True in c_path[1]: shared.append(True)
            else: shared.append(False)
        logger.info("shared: {}".format(shared))

    for ep in range(n_train_epochs):
        model.train()
        cum_loss, cum_qa_loss, cum_lm_loss, cur_n_inputs = 0, 0, 0, 0
        cum_kl_v_loss = 0
        for n_steps, (_, _, cqa, _, Y, gen_X, gen_Y, task_id, idx) in enumerate(train_dataloader):
            if cqa is None: continue

            n_inputs = cqa[0].shape[0]
            lens = cqa[0].shape[1]


            if task_ids[0] > 0:
                if not shared[task_id[0][0].item()]:
                    logger.warning('[Warning] no grad reauired for task_id[0][0]: {}'.format(task_id[0][0].item()))
                    continue
      
            model.config.batch_task_id = task_id[0][0].item()

            if args.cm_kd==True:
                if task_ids[0] <=1 :
                    losses = get_losses(model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(), train_loss_fct)
                else:
                    if task_id[0][0].item() == task_ids[0]:
                        losses = get_losses(model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(), train_loss_fct)
                    else:
                        if (n_steps + 1) % args.logging_steps == 0:
                            logger.info('Data_ID/Task_ID: {}/{}, type: {}/{}'.format(
                                            task_id[0][0].item(), task_ids[0], type(task_id[0][0].item()), type(task_ids[0])))
                            logger.info('Using KD Loss')
                        assert task_id[0][0].item() < task_ids[0]
                        teacher_model.config.batch_task_id = task_id[0][0].item()
                        logger.info('setting teacher model path for transfer: {}'.format(teacher_model.config.batch_task_id))
                        losses = get_kd_losses(model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(), 
                                                        train_loss_fct_ce, train_loss_fct_kd, teacher_model)
            elif args.cm_kl_v_flag:
                if task_id[0][0].item() == task_ids[0] and ep >= args.cm_v_kl_ep-1:
                    losses = get_ce_and_vertical_kl_losses(model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(), 
                                                        train_loss_fct, c_module_list, typ=args.cm_v_kl_typ, task_id=task_ids[0])
                else:
                    losses = get_losses(model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(), train_loss_fct)
            else:
                losses = get_losses(model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(), train_loss_fct)

            if losses[1].item() == 0: loss = losses[0] 
            else: loss = losses[0] + losses[1]
            if args.cm_kl_v_flag and task_id[0][0].item() == task_ids[0]:
                loss = loss + losses[3]['kl_loss']

            train_once(loss, n_inputs, path[model.config.batch_task_id])       

            qa_loss = losses[0].item() * n_inputs
            lm_loss = losses[1].item() * n_inputs
            cum_loss += (qa_loss + lm_loss)
            cum_qa_loss += qa_loss
            cum_lm_loss += lm_loss
            cur_n_inputs += n_inputs
            if args.cm_kl_v_flag and task_id[0][0].item() == task_ids[0]: 
                kl_v_loss = losses[3]['kl_loss'].item() * n_inputs
                cum_kl_v_loss += kl_v_loss   
                cum_loss += kl_v_loss

            if args.gradient_debug and task_ids[0] == 0: break

            if args.constant_sch:
                lr = scheduler.get_lr()[0]
            else:
                lr = scheduler.get_lr()

            if cur_n_inputs==0:
                logger.info('ep:{}, n_steps: {}, cur_n_inputs: {}'.format(ep, n_steps, cur_n_inputs))
                continue
            if (n_steps + 1) % args.logging_steps == 0:
                logger.info('progress {:.3f} , lr {:.1E} , loss {:.3f} , qa loss {:.3f} , lm loss {:.3f}, avg batch size {:.1f}'
                            .format(ep + cur_n_inputs/len(train_qadata),
                                    lr, cum_loss/cur_n_inputs,
                                    cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs,
                                    cur_n_inputs/(n_steps + 1)))
                if args.cm_kl_v_flag and task_id[0][0].item() == task_ids[0]: 
                    logger.info('progress {:.3f} , lr {:.1E} , loss {:.3f} , kl_v loss {:.3f}'.format(
                                    ep + cur_n_inputs/len(train_qadata), lr, cum_loss/cur_n_inputs,
                                    cum_kl_v_loss/cur_n_inputs))

        if cur_n_inputs==0:
            logger.info('ep:{}, cur_n_inputs: {}'.format(ep, cur_n_inputs))
            continue
        if not args.gradient_debug:
            logger.info('n_train_epochs {}:')
            tot_n_steps += (n_steps + 1)
            val_loss, val_qa_loss, val_lm_loss = validation(model, valid_dataloader, train_loss_fct)
            logger.info('epoch {}/{} done , tot steps {} , loss {:.2f} , qa loss {:.2f} , lm loss {:.2f}, val loss {:.2f}, vqa loss {:.2f}, vlm loss {:.2f}, avg batch size {:.1f}'.format(
                ep+1, n_train_epochs, tot_n_steps,
                cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, 
                cum_lm_loss/cur_n_inputs, val_loss,
                val_qa_loss, val_lm_loss, cur_n_inputs/(n_steps+1)
            ))

        print_para(model)
        if args.gradient_debug and task_ids[0] > 0: exit(0)

    torch.save(model.state_dict(), os.path.join(model_dir, SAVE_NAME+"finish"))
    adapter_list = model.get_adapter_list()
    np.save(os.path.join(model_dir, "adapter_list.npy"), adapter_list)
    logger.info("MODEL SAVED!")

    del optimizer
    del scheduler
    torch.cuda.empty_cache()

    if args.layer_debug and task_ids[0] == len(args.tasks) - 1:
        model.config.forward_mode = 'Transfer'
        model.config.testing = False
        gen_path = os.path.join(model_dir, "lm-origin-{}-{}.csv".format(args.layer_debug_cnt, args.partial_learn))
        holder = []
        with torch.no_grad(): create_extra_data(tasks[0], tasks[0], model, holder, None, gen_path, [1])

        logger.info("Modifying list")
        model.modify_list(args.layer_debug_cnt, 0, 1)

        gen_path = os.path.join(model_dir, "lm-modified-{}-{}.csv".format(args.layer_debug_cnt, args.partial_learn))
        holder = []
        with torch.no_grad(): create_extra_data(tasks[0], tasks[0], model, holder, None, gen_path, [1])

        exit(0)

    return model


if __name__ == '__main__':

    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)
    
    if args.cm_debug:
        args.whole_mix_step = 0
        args.warm_mix_step = 0
        args.z_max_batch_size=16
        args.cm_test_batch_size=16
        args.cm_h_kl_ep = 1
        if args.cm_kl_h_flag:
            args.z_train_epochs = [1]*len(args.z_train_epochs) 
        elif args.cm_kl_v_flag:
            args.z_train_epochs = [12]*len(args.z_train_epochs) 
            args.cm_v_kl_ep = 0
            args.whole_mix_step = 0
            args.warm_mix_step = 0
        else:
            args.z_train_epochs = [12]*len(args.z_train_epochs) 

    if not args.z_debug:
        make_dir(args.model_dir_root)

        init_logging(os.path.join(args.model_dir_root, 'log_train.txt'))
        logger.info('args = {}'.format(str(args)))

        model = None
        if args.seq_train_type in ["multitask", "multilm"]:
            model = train(list(range(len(args.tasks))), model)
        else:
            args.tasks.insert(0, "NULL_task")
            args.z_train_epochs.insert(0, 0)
            args.z_train_lrs.insert(0, 0)
            if args.unbound:
                TASK_DICT = lll_unbound_setting(split_size=args.unbound)

            if args.cm_continue_from_task != None:
                logger.info('continue training from [{}].'.format(args.cm_continue_from_task))
                start_task_id = args.tasks.index(args.cm_continue_from_task)

            for task_id in range(len(args.tasks)):

                if args.cm_continue_from_task != None:
                    if task_id < start_task_id:
                        gen_token = get_gen_token(args.tasks[task_id])
                        TOKENIZER.add_tokens([gen_token])
                        SPECIAL_TOKENS[args.tasks[task_id]] = gen_token
                        SPECIAL_TOKEN_IDS[args.tasks[task_id]] = TOKENIZER.convert_tokens_to_ids(gen_token)
                        logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[args.tasks[task_id]]))
                        continue
                    elif task_id == start_task_id:
                        # init the model
                        prev_tasks = [args.tasks[task_id-1]]
                        prev_model_dir = get_model_dir(prev_tasks)
                        model = load_model(prev_model_dir)
                        logger.info('Previous model {} laoding'.format(prev_tasks))
                    else:
                        pass
                            
                if task_id == 0:
                    model = Transfer([task_id], model)
                else:
                    if not args.fake_mix_debug:
                        if args.cm_continue_from_task != None and args.cm_ct_n_skip_Mix:
                            # add token;
                            gen_token = get_gen_token(args.tasks[task_id])
                            TOKENIZER.add_tokens([gen_token])
                            SPECIAL_TOKENS[args.tasks[task_id]] = gen_token
                            SPECIAL_TOKEN_IDS[args.tasks[task_id]] = TOKENIZER.convert_tokens_to_ids(gen_token)
                            logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[args.tasks[task_id]]))
                            # init the model
                            model = cm_load_one_model(args.tasks[task_id])
                            Fit_or_Not, trans = False, True            
                            logger.info('Model {}, which has been well trained from the Mix stage, is loaded'.format(args.tasks[task_id]))
                        else:
                            model, Fit_or_Not, trans, current_fit_epoch = Mix([task_id], model)
                        fit_bonus = 0
                        if Fit_or_Not:
                            model = Fit([task_id], model, current_fit_epoch)
                            fit_bonus = 0
                        if trans:
                            model = Transfer([task_id], model, fit_bonus)
                        if args.cm_debug:
                            break
                    else:
                        logger.info("In fake mix debug!")
                        tmp_model = copy.deepcopy(model)
                        tmp_model, Fit_or_Not, trans, current_fit_epoch = Mix([task_id], tmp_model)
                        del tmp_model

                        tasks = [args.tasks[task_id]]
                        model_dir = get_model_dir(tasks)
                        adapter_list = np.load(os.path.join(model_dir, "adapter_list.npy"))
                        model.update_adapter_list(adapter_list)   
                        model = Transfer([task_id], model, 0)
    else:
        init_logging(os.path.join(args.model_dir_root, 'log_train_debug.txt'))
        logger.info('args = {}'.format(str(args)))

        model = None
        if args.z_debug_tsk_num >= 1:
            from mytransformers import GPT2LMHeadModel, GPT2Config

            tasks = [args.tasks[args.z_debug_tsk_num - 1]]
            if args.z_debug_start_from_trans:
                tasks = [args.tasks[args.z_debug_tsk_num]]

            model_dir = get_model_dir(tasks)
            model_config = GPT2Config.from_json_file(os.path.join(model_dir, "config.json"))
            model = GPT2LMHeadModel(model_config)
            model.resize_token_embeddings(50260 + len(args.tasks))

            adapter_list = np.load(os.path.join(model_dir, "adapter_list.npy"))
            model.add_adapter_by_list(adapter_list, config=args.adapt_type)
            state_dict = torch.load(os.path.join(model_dir, "model-finish"), map_location='cuda:0')
            m, n = model.load_state_dict(state_dict, strict=False)
            logger.info("Missing : {}, Unexpected: {}".format(m, n))
            model.cuda()

            global TOKENS_WEIGHT
            tsk_cnt = 0
            while tsk_cnt < args.z_debug_tsk_num:
                TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
                gen_token = get_gen_token(args.tasks[tsk_cnt])
                TOKENIZER.add_tokens([gen_token])
                SPECIAL_TOKENS[args.tasks[tsk_cnt]] = gen_token
                SPECIAL_TOKEN_IDS[args.tasks[tsk_cnt]] = TOKENIZER.convert_tokens_to_ids(gen_token)
                tsk_cnt += 1

            if not args.fp32:
                model = FP16_Module(model)

            while 50260 + len(args.tasks) != TOKENS_WEIGHT.shape[0]:
                TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
                logger.info("Add one dim weight!")

        for task_id in range(args.z_debug_tsk_num, len(args.tasks)):
            if task_id == 0:
                model = Transfer([task_id], model)
            elif task_id == args.z_debug_tsk_num and args.z_debug_start_from_trans:
                fit_bonus = 0
                model.config.forward_mode = 'Transfer'
                model.config.testing = False
                adapter_names = get_unique_sorted_adapter_array(adapter_list)
                model.train_adapter([str(i) for i in adapter_names])

                model = Transfer([task_id], model, fit_bonus)
            else:
                model, Fit_or_Not, trans, current_fit_epoch = Mix([task_id], model)
                fit_bonus = 0
                if Fit_or_Not:
                    model = Fit([task_id], model, current_fit_epoch)
                    fit_bonus = 0
                if trans:
                    model = Transfer([task_id], model, fit_bonus)

    logger.info('[DONE] Training Process Finished!')

