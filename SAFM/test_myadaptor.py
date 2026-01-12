import torch
import csv
import os,sys
import json
import logging
import numpy as np
from fp16 import FP16_Module
import GPUtil
from collections import OrderedDict
from settings_myadaptor import args, MODEL_CLASS, TOKENIZER, SPECIAL_TOKEN_IDS, init_logging
from settings_myadaptor import MEMORY_FACTOR, LEN_FACTOR, TASK_DICT, MODEL_CONFIG, \
                                DATA_ATTRS, SPECIAL_TOKENS, CONFIG_CLASS, CONFIG_NAME, METRIC_DICT
from utils_myadaptor import QADataset, top_k_top_p_filtering, create_dataloader, logits_to_tokens, get_model_dir
from utils_myadaptor import sample_sequence, remove_id, get_gen_token, lll_unbound_setting
from metrics import compute_metrics
import pdb
import random

from cm_losses import get_unique_sorted_adapter_array
logger = logging.getLogger(__name__)


def test_one_to_one(task_load, task_eval, model, score_dict=None, model_task_metric_dict=None, gen_token_task=None):

    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("start to test { task: %s (load) %s (eval), seq train type: %s }" % (task_load, task_eval, args.seq_train_type))
    if hasattr(args, 'extra_e2e'):
        if args.extra_e2e and task_eval=='e2enlg':
            logger.info("USE extra_e2e!", flush=True)
            TASK_DICT[task_eval]["test"] = TASK_DICT[task_eval]["test"].replace('test', 'extra')

    test_qadata = QADataset(                    
                TASK_DICT[task_eval]["test"] if not args.test_training_set else TASK_DICT[task_eval]["train"] , 
                "test", 
                SPECIAL_TOKEN_IDS[task_eval] if gen_token_task is None else SPECIAL_TOKEN_IDS[gen_token_task]
            ).sort()


    max_a_len = test_qadata.max_a_len
    logger.info('max_a_len: {}'.format(max_a_len))


    test_dataloader = create_dataloader(test_qadata, "test", args.cm_test_batch_size)
    n_examples = len(test_qadata)
    logger.info("len of test dataset: {}".format(n_examples))

    need_process = OrderedDict()
    qa_results = [0 for _ in range(n_examples)]
    all_pasts = [[0 for _ in range(n_examples)] for __ in range(MODEL_CONFIG.n_layer)]
    max_tot_lens = [0 for _ in range(n_examples)]

    cnt = 0
    model.config.testing = True
    model.config.batch_task_id = args.tasks.index(task_eval) if gen_token_task is None else args.tasks.index(gen_token_task)
    logger.info(model.config.batch_task_id)
    for n_steps, (cqs, len_cqs, _, _, _, _, _, task_id, _) in enumerate(test_dataloader):
        cqs = cqs[0]
        len_cqs = len_cqs[0]
        n_inputs = cqs.shape[0]

        all_outputs = model(input_ids=cqs.cuda(), use_cache=True)
        outputs = all_outputs.logits
        if args.model_name == "gpt2":
            pasts = all_outputs.past_key_values
        next_logits = outputs[range(n_inputs), len_cqs-1, :] / args.temperature_qa
        next_tokens = logits_to_tokens(next_logits).cpu()

        for i in range(n_inputs):
            max_tot_lens[cnt] = max_a_len + test_qadata[cnt][1]
            qa_results[cnt] = cqs[i][:len_cqs[i]]
            if next_tokens[i] != SPECIAL_TOKEN_IDS["eos_token"]:
                qa_results[cnt] = torch.cat((cqs[i][:len_cqs[i]], next_tokens[i]))
                if len(qa_results[cnt]) not in [max_tot_lens[cnt], args.max_len]:
                    need_process.update([[cnt, None]])
                    if args.model_name == "gpt2":
                        for layer_id in range(MODEL_CONFIG.n_layer):
                            # for 539: transfer to cpu
                            all_pasts[layer_id][cnt] = torch.stack(
                                pasts[layer_id], dim=0)[:, i, ..., :len_cqs[i], :].type(torch.float32 if args.fp32 else torch.half).cpu()
            cnt += 1

        if n_steps % 20 == 0:
            sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)
    sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)

    if task_eval in ['wikisql','woz.en','multinli.in.out']:
        ids = test_qadata.get_indices()
        test_qadata.sort_by_index()
        qa_results = [x[1] for x in sorted([(i, g) for i, g in zip(ids, qa_results)])]
    for i in range(len(test_qadata)):
        _, len_cq, _, _, Y, _, _, hashcode, _, _ = test_qadata[i]
        if task_eval in ['wikisql','woz.en']:
            Y = test_qadata.answers[i] if not args.test_training_set else test_qadata.answers[0]
        else:
            Y = list(filter(lambda x: x != -1, Y))[:-1]  # remove eos
            Y = ' '.join([str(y) for y in Y]).split(str(SPECIAL_TOKEN_IDS["pad_token"]))
            Y = [TOKENIZER.decode(list(map(int, y.split()))) for y in Y]
        if not args.test_training_set:
            try:
                qa_results[i] = [TOKENIZER.decode(qa_results[i].tolist()[len_cq:]), Y]
            except TypeError:
                logger.info("Type error!!!")
                logger.info(qa_results[i].tolist()[len_cq:])
                logger.info(Y)
                qa_results[i] = ["Type error!", Y]
        else:
            qa_results[i] = [TOKENIZER.decode(qa_results[i].tolist()[len_cq:]), Y, hashcode]
   
    if score_dict!=None: score = get_test_score(task_eval, qa_results, score_dict) 

    if model_task_metric_dict!=None: model_task_metric_dict[task_load][task_eval].update(score)

    model_dir = model.model_dir
    ep = model.ep

    if args.layer_debug:
        results_path = "qa_{}_{}".format(task_eval, ep+1) + "-" + str(args.layer_debug_cnt) + "-modified.csv"
    elif args.layer_debug_cnt != -1:
        results_path = "qa_{}_{}".format(task_eval, ep+1) + "-" + str(args.layer_debug_cnt) + "-original.csv"
    else:
        results_path = "qa_{}_{}.csv".format(task_eval, ep+1)

    results_path = os.path.join(model_dir, results_path)

    if args.cm_case_study_flag:
        logger.info('writing to csv {}'.format(results_path))
        with open(results_path, "w", encoding="utf-8") as f:
            qa_writer = csv.writer(f, delimiter=',')
            qa_writer.writerow(["y", "pred"])
            for pred, y in qa_results:
                if task_eval == 'wikisql':
                    y = y["answer"]
                elif task_eval == 'woz.en':
                    y = y[1]
                qa_writer.writerow([y, pred])
        return

    return model, score_dict



def get_test_score(task_eval, qa_results, score_dict):

    score = compute_metrics(
            qa_results,
            rouge= 'rouge' == METRIC_DICT[task_eval],
            bleu= 'bleu' == METRIC_DICT[task_eval],

        )
    score_dict[task_eval] = score
    return score



def test_one_to_many(task_load, model_task_metric_dict, ii_iplus_flag=False):       
    score_dicts = []
    for ep in range(args.n_train_epochs[task_load]-1, args.n_train_epochs[task_load]) if not args.test_all else range(args.n_train_epochs[task_load]):
        model_dir = get_model_dir([task_load])
        model_path = os.path.join(model_dir, 'model-finish')
        config_path = os.path.join(model_dir,CONFIG_NAME)

        model_config = CONFIG_CLASS.from_json_file(config_path)
        model = MODEL_CLASS(model_config).cuda()
        model.resize_token_embeddings(50260 + len(args.tasks))    

        adapter_list = np.load(os.path.join(model_dir, "adapter_list.npy"))
        model.add_adapter_by_list(adapter_list, config=args.adapt_type)   
        model.config.forward_mode = 'Transfer'
        uni_list = get_unique_sorted_adapter_array(adapter_list).tolist()

        if not args.adapterdrop:                
            model.train_adapter(uni_list)
        else:
            model.train_adapter(uni_list, [0, 1, 2])

        state_dict = torch.load(model_path, map_location='cuda:0')
        m, n = model.load_state_dict(state_dict, strict=False)
        logger.info("Missing : {}, Unexpected: {}".format(m, n))
        model.cuda()
        model.eval()

        if args.layer_debug:
            logger.info("Modifying list")
            model.modify_list(args.layer_debug_cnt, 0, 1)

        if not args.fp32:
            model = FP16_Module(model)

        model.ep = ep
        model.model_dir = model_dir
        logger.info("task: {}, epoch: {}".format(task_load, ep+1))
        score_dict = {k:None for k in args.tasks}
        with torch.no_grad():                   
            if ii_iplus_flag:                                                        
                task_id_load = args.tasks.index(task_load)
                for task_id_eval, task_eval in enumerate(args.tasks):
                    if task_id_eval == task_id_load:
                        test_one_to_one(task_load, task_eval, model, score_dict, model_task_metric_dict, gen_token_task=None) 
                        logger.info("score: {}".format(score_dict))
                    elif task_id_eval == task_id_load+1:                     
                        test_one_to_one(task_load, task_eval, model, score_dict, model_task_metric_dict, gen_token_task=task_load) 
                        logger.info("score: {}".format(score_dict))
                    else:
                        pass
            else:
                for task_eval in args.tasks:
                    if task_eval == 'NULL_task': continue
                    test_one_to_one(task_load, task_eval, model, score_dict, model_task_metric_dict)
                    logger.info("score: {}".format(score_dict))
                    if task_load == task_eval: break
        logger.info("score: {}".format(score_dict))
        score_dicts.append(score_dict)

    if args.layer_debug:
        save_name = "metrics" + "-" + str(args.layer_debug_cnt) + "-modified.json"
    elif args.layer_debug_cnt != -1:
        save_name = "metrics" + "-" + str(args.layer_debug_cnt) + "-original.json"
    else:
        save_name = "metrics.json"

    with open(os.path.join(model_dir, save_name),"w") as f:
        json.dump(score_dicts, f)

def init_metric_dict():

    model_task_metric_dict = OrderedDict()
    for model in args.tasks:
        model_task_metric_dict.update({model: OrderedDict()})
        for task in args.tasks:
            model_task_metric_dict[model].update({task: OrderedDict()})
    print(model_task_metric_dict)
    return model_task_metric_dict
            

def case_study(task_load):        
    for ep in range(args.n_train_epochs[task_load]-1, args.n_train_epochs[task_load]) if not args.test_all else range(args.n_train_epochs[task_load]):
        model_dir = get_model_dir([task_load])
        model_path = os.path.join(model_dir, 'model-finish')
        config_path = os.path.join(model_dir,CONFIG_NAME)

        model_config = CONFIG_CLASS.from_json_file(config_path)
        model = MODEL_CLASS(model_config).cuda()
        model.resize_token_embeddings(50260 + len(args.tasks))    
        adapter_list = np.load(os.path.join(model_dir, "adapter_list.npy"))
        model.add_adapter_by_list(adapter_list, config=args.adapt_type)  
        model.config.forward_mode = 'Transfer'
        uni_list = get_unique_sorted_adapter_array(adapter_list).tolist()

        if not args.adapterdrop:                
            model.train_adapter(uni_list)
        else:
            model.train_adapter(uni_list, [0, 1, 2])

        state_dict = torch.load(model_path, map_location='cuda:0')
        m, n = model.load_state_dict(state_dict, strict=False)
        logger.info("Missing : {}, Unexpected: {}".format(m, n))
        model.cuda()
        model.eval()

        if args.layer_debug:
            logger.info("Modifying list")
            model.modify_list(args.layer_debug_cnt, 0, 1)

        if not args.fp32:
            model = FP16_Module(model)

        model.ep = ep
        model.model_dir = model_dir
        logger.info("task: {}, epoch: {}".format(task_load, ep+1))
        with torch.no_grad():
            for task_eval in args.cm_case_study_test_task:     
                logger.info('case generating on {}'.format(task_eval))          
                test_one_to_one(task_load, task_eval, model, score_dict=None, model_task_metric_dict=None)



if __name__ == '__main__':
    if args.n_gpus > 1:
        raise NotImplementedError("test can be run with only one gpu currently!")

    args.tasks.insert(0, "NULL_task")
    args.z_train_epochs.insert(0, 0)
    args.z_train_lrs.insert(0, 0)
    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)
    init_logging(os.path.join(args.model_dir_root, 'log_test.txt'))
    logger.info('args = {}'.format(args))

    # Help with type error - 
    for task_load in args.tasks:
        gen_token = get_gen_token(task_load)
        TOKENIZER.add_tokens([gen_token])
        SPECIAL_TOKENS[task_load] = gen_token
        SPECIAL_TOKEN_IDS[task_load] = TOKENIZER.convert_tokens_to_ids(gen_token)

    logger.info(SPECIAL_TOKEN_IDS)
    
    model_task_metric_dict = init_metric_dict()   

    logger.info('Calculating a[T, i] for evaluation')
    if args.seq_train_type in ["multitask", "multilm"]:
        test_one_to_many('_'.join(args.tasks))
    else:
        if args.cm_debug:
             for task_load in args.cm_test_tasks:
                logger.info('Calculating rows===========')
                test_one_to_many(task_load, model_task_metric_dict)
                break
        if args.cm_case_study_flag:
            case_study(task_load)
            sys.exit()
        if args.unbound:                    
            TASK_DICT = lll_unbound_setting(split_size=args.unbound, data_type="test",test_target="origin")
            for task_load in args.splitted_tasks:
                test_one_to_many(task_load)
        else:
            
            if args.cm_fwt_dwt_flag:
                logger.info('Calculating upper triangular matrix===========')
                for task_load in args.tasks[:-1]:
                    if task_load == 'NULL_task': continue
                    test_one_to_many(task_load, model_task_metric_dict, ii_iplus_flag=True)
                    logger.info("******************{}".format(task_load))

            for task_load in args.cm_test_tasks:
                logger.info('Calculating rows===========')
                test_one_to_many(task_load, model_task_metric_dict)
            
            logger.info('printing metric matrix')
            logger.info(model_task_metric_dict)
            testing_result_dir = get_model_dir(['testing_result'])
            os.makedirs(testing_result_dir, exist_ok=True)
            with open(os.path.join(testing_result_dir, 'rst.json'), 'w') as f:
                json.dump(model_task_metric_dict, f)
            
            logger.info('print adapter list ===================')
            logger.info(np.load(os.path.join(get_model_dir([args.tasks[-1]]), "adapter_list.npy")))          
            
            for task_load in args.cm_test_tasks:
                task_load_id = args.tasks.index(task_load)
                logger.info('Calcualting all metrcis for model-[{}]============'.format(task_load))
                logger.info('Calculating metrics for each specific task**********************')
                all_metrics = []
                for task_eval_id, task_eval in enumerate(args.tasks):
                    if task_eval == 'NULL_task': continue
                    logger.info("model-[%20s] tesing on [%20s]: %s" % (
                                task_load, task_eval, json.dumps(model_task_metric_dict[task_load][task_eval])))
                    all_metrics.append(model_task_metric_dict[task_load][task_eval][METRIC_DICT[task_eval]])
                    if task_load == task_eval: break
                ave = sum(all_metrics) / len(all_metrics)
                logger.info("model-[%20s] AVE: %.2f" % (task_load, ave))
                if args.cm_fwt_dwt_flag:

                    fwt, T = 0.0, task_load_id
                    for i in range(1, T):     
                        fwt += model_task_metric_dict[args.tasks[i]][args.tasks[i+1]][METRIC_DICT[args.tasks[i+1]]]
                    fwt = fwt / (T-1.0)
                    logger.info("model-[%20s] FWT: %.2f" % (task_load, fwt))
                    
                    bwt, T = 0.0, task_load_id
                    for i in range(1, T):      
                        bwt = bwt + (model_task_metric_dict[task_load][args.tasks[i]][METRIC_DICT[args.tasks[i]]] - 
                            model_task_metric_dict[args.tasks[i]][args.tasks[i]][METRIC_DICT[args.tasks[i]]])
                    bwt = bwt / (T-1.0)
                    logger.info("model-[%20s] BWT: %.2f" % (task_load, bwt)) 

