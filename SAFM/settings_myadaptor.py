import os
import json
import argparse
import logging
import datetime
logger = logging.getLogger(__name__)

import GPUtil
from mytransformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig
from mytransformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, CONFIG_NAME 
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILL_VAL = -1

# for preseqlen=10 pretraining
# LEN_FACTOR = 1.08
# original setting
# LEN_FACTOR = 1.163
# lamaml setting
LEN_FACTOR = 1.2
MEMORY_FACTOR = {
    "finetune": 0.58,
    "multitask": 0.58,
    "multilm": 0.35,
    "lll": 0.35,
    "llewc": 0.35,
    "ewc": 0.30,
    "mas": 0.18,
    "gem": 0.50,
}
TURING_ARCHS = {'Tesla V100', '2080 Ti'}
MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig),
}
SAVE_NAME = 'model-'
FINAL_SAVE_NAME = 'model-finish'




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adam_epsilon", default=1e-4, type=float)

    parser.add_argument("--add_task_tokens", action="store_true")
    parser.add_argument("--use_eos_as_sos", action="store_true")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--decay_style", type=str, default="linear")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--real_sample", action="store_true")
    parser.add_argument("--unbound", type=int, default=0)
    parser.add_argument("--gen_lm_sample_percentage", type=float, default=0.05)
    parser.add_argument("--distil", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help='used in decision stage')
    parser.add_argument("--logging_steps", type=int, default=100)

    parser.add_argument("--qa_theta", type=float, default=1.0)
    parser.add_argument("--lm_lambda", type=float, default=0.25)

    parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--max_n_epochs", type=int, default=9)
    parser.add_argument("--min_batch_size", type=int, default=4)
    parser.add_argument("--min_n_steps", type=int, default=1500)
    parser.add_argument("--model_dir_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt2", choices=["gpt2", "openai-gpt"])
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_train_epochs", type=int, default=1)
    parser.add_argument("--dynamic_epochs", action="store_true")
    parser.add_argument("--n_warmup_ratio", type=float, default=0.005)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--use_sep", action="store_true")
    parser.add_argument("--reg_lambda", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_all", action="store_true")
    parser.add_argument("--test_training_set", action="store_true")
    parser.add_argument("--round_robin", action="store_true")
    parser.add_argument("--upsample_data", type=str, default=None)
    parser.add_argument("--seq_distil", action="store_true")
    parser.add_argument("--extra_e2e", action="store_true")
    parser.add_argument("--ref1", action="store_true")
    parser.add_argument("--multitask_specific", action="store_true")
    parser.add_argument("--seq_train_type", type=str, default="lll", choices=["lll","llewc","finetune","multitask","mas","ewc","gem","multilm"])
    parser.add_argument("--tasks", nargs='+', default=["squad2"])
    parser.add_argument("--skip_tasks", nargs='+')
    parser.add_argument("--temperature_kd", type=float, default=2.0)
    parser.add_argument("--temperature_lm", type=float, default=1.0)
    parser.add_argument("--temperature_qa", type=float, default=1.0)
    parser.add_argument("--test_batch_size", type=int, default=0)
    parser.add_argument("--tokens_weight", type=float, default=5)
    parser.add_argument("--top_k_lm", type=int, default=20)
    parser.add_argument("--top_k_qa", type=int, default=20)
    parser.add_argument("--top_p_lm", type=float, default=0.)
    parser.add_argument("--top_p_qa", type=float, default=0.)
    parser.add_argument("--train_batch_size", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--qp_margin", type=float, default=0.5)

    # added args
    parser.add_argument("--z_debug", action="store_true", help='recover from one task')
    parser.add_argument("--z_debug_tsk_num", type=int, default=0, help='recover from one task, which task?')
    parser.add_argument("--z_debug_dir", type=str, default="None", help='recover from one task, which dir?')
    parser.add_argument("--z_debug_model", type=str, default="None", help='recover from one task, which model?')
    parser.add_argument("--ppl_thr", type=float, default=100.0, help='not used')

    parser.add_argument("--z_learning_rate", type=float, default=1e-3)
    parser.add_argument("--z_warmup_step", type=int, default=1000)
    parser.add_argument("--z_step", type=int, default=300)
    parser.add_argument("--preseqlen", type=int, default=10, help='not used')
    parser.add_argument("--mid_dim", type=int, default=256, help='not used')

    parser.add_argument("--gradient_block", action="store_true", help='not used')
    parser.add_argument("--mom", type=float, default=0.1, help='not used')
    parser.add_argument("--no_repara", action="store_true", help='not used')
    parser.add_argument("--imp_p", type=float, default=0.9, help='not used')
    parser.add_argument("--first_only", action="store_true", help='not used')

    parser.add_argument("--lamaml", action="store_true", help='Set to True to use our replay strategy (with high replay frequency)')

    parser.add_argument("--second_order", action="store_true", help='not used')
    parser.add_argument("--learn_lr", action="store_true", help='not used')
    parser.add_argument("--sync_update", action="store_true", help='not used')
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help='not used')
    parser.add_argument('--opt_lr', type=float, default=1e-3, help='not used')
    parser.add_argument('--opt_wt', type=float, default=1e-3, help='not used')
    parser.add_argument('--alpha_init', type=float, default=1e-3, help='not used')
    # parser.add_argument("--glances", default=1, type=int, help="In single pass setting")
    parser.add_argument('--glances', nargs='+', type=int, default=[1, 1, 1, 1, 1], help='not used')

    parser.add_argument("--pre_learning_rate", type=float, default=1e-4, help='not used')
    parser.add_argument("--pre_warmup_step", type=int, default=100, help='not used')
    parser.add_argument("--pre_start_from", type=str, default="None", help='not used')

    parser.add_argument("--pretrained_prefix", type=str, default="None", help='not used')
    parser.add_argument("--dump", action="store_true", help='not used')
    parser.add_argument("--z_max_batch_size", type=int, default=32, help='not used')

    parser.add_argument("--test_skip", type=int, default=0, help='not used')
    parser.add_argument("--meta_last", action="store_true", help='not used')
    parser.add_argument("--observe_type", type=int, default=1, help='not used')
    parser.add_argument("--rep_beta", type=float, default=0.1, help='not used')
    parser.add_argument("--half_assert", action="store_true", help='not used')
    parser.add_argument('--thr', nargs='+', type=float, default=[0, 0, 0, 0, 0], help='not used')
    parser.add_argument("--random_batch", action="store_true", help='not used')
    parser.add_argument("--replay_first", action="store_true", help='not used')
    parser.add_argument("--random_first", action="store_true", help='not used')

    parser.add_argument('--grad_coe', nargs='+', type=float, default=[1.0, 1.0], help='not used')
    parser.add_argument("--return_m_grads", action="store_true", help='not used')
    parser.add_argument("--last_half_is_whole", action="store_true", help='not used')
    parser.add_argument("--first_half_is_old", action="store_true", help='not used')
    parser.add_argument("--maml_qalm", action="store_true", help='not used')
    parser.add_argument("--block_first_meta", action="store_true", help='not used')
    parser.add_argument("--sync_dropout", action="store_true", help='not used')
    parser.add_argument("--same_pass", action="store_true", help='not used')

    # parser.add_argument('--task_test', nargs='+', type=int, default=[1, 2, 3, 4, 5], help='which task to test?')
    parser.add_argument('--task_test', nargs='+', type=int, default=[5], help='which task to test?')
    parser.add_argument("--grad_ob", action="store_true", help='not used')
    parser.add_argument("--z_debug_noshuff", action="store_true", help='not used')
    parser.add_argument("--mask_neg", action="store_true", help='not used')
    parser.add_argument("--mask_pos", action="store_true", help='not used')
    parser.add_argument("--tanh_trans", action="store_true", help='not used')
    parser.add_argument("--alpha_temp", type=float, default=1e-8, help='not used')
    parser.add_argument("--alpha_b", type=float, default=0.0, help='not used')
    parser.add_argument("--meta_block_emb", action="store_true", help='not used')
    parser.add_argument("--no_refresh", action="store_true", help='not used')
    parser.add_argument("--use_momentum", action="store_true", help='not used')
    parser.add_argument("--learn_lr_with_nomul", action="store_true", help='not used')

    parser.add_argument("--alpha_dropout", type=float, default=0.0, help='not used')
    parser.add_argument("--smooth", action="store_true", help='not used')
    parser.add_argument("--pretraining", action="store_true", help='not used')
    parser.add_argument("--prefix_dropout", type=float, default=0.0, help='not used')
    parser.add_argument('--set_p_drop', nargs='+', type=int, default=[0, 1, 0], help='not used')
    parser.add_argument("--use_autoM", action="store_true", help='not used')
    parser.add_argument("--lr_epoch", type=int, default=1, help='not used')
    parser.add_argument("--single_alpha", action="store_true", help='not used')
    parser.add_argument("--rev_gradient", action="store_true", help='not used')
    parser.add_argument("--lr_grad_norm", type=float, default=1.0, help='not used')
    parser.add_argument('--wt_op', nargs='+', type=bool, default=[True, False, False, True], help='not used')

    parser.add_argument("--id", type=int, default=1, help='Exp id')
    parser.add_argument("--adapt_type", type=str, default="houlsby")
    parser.add_argument("--constant_sch", action="store_true", help='what scheduler to use?')
    parser.add_argument("--top_two", action="store_true")
    parser.add_argument("--entropy_coe", type=float, default=0.01, help='entropy loss coe')
    parser.add_argument("--load_old", action="store_true")
    parser.add_argument("--mix_ini", type=float, default=0.2, help='weight initialization')

    parser.add_argument("--gradient_debug", action="store_true")
    parser.add_argument("--whole_optim", action="store_true")
    parser.add_argument("--clear_model", action="store_true")
    parser.add_argument("--whole_mix_step", type=int, default=6, help='the whole epoch number of decision stage')
    parser.add_argument("--warm_mix_step", type=int, default=1, help='first multiple epochs, in which the weight coefficient is not trained')
    parser.add_argument("--fit_epoch", type=int, default=0, help='not used')
    parser.add_argument("--last_dim_coe", type=float, default=0.0, help='not used')
    parser.add_argument("--select_temp", type=float, default=1.0, help='not used')
    parser.add_argument("--pretrain_adapter", action="store_true", help='not used')
    parser.add_argument("--mix_loss_norm", action="store_true", help='not used')
    parser.add_argument("--random_replay_batch", action="store_true", help='Set to True by default, create the order of replay batches randomly')
    parser.add_argument("--z_debug_start_from_trans", action="store_true", help='recover from one task training stage')
    parser.add_argument("--load_model_for_stage", action="store_true")
    parser.add_argument("--fake_mix_debug", action="store_true")
    parser.add_argument("--generate_after", action="store_true")
    parser.add_argument("--mix_loss_coe", type=float, default=1.0, help='not used')
    parser.add_argument("--partial_transfer", action="store_true", help='whether to fix unshared modules from old tasks')

    parser.add_argument('--z_train_epochs', nargs='+', type=int, default=[9, 9, 9, 9, 9], help='set task wise epochs')
    parser.add_argument('--z_train_lrs', nargs='+', type=float, default=[1.75e-4, 1.75e-4, 1.75e-4, 1.75e-4, 1.75e-4], help='set task wise learning rate')

    parser.add_argument("--layer_debug", action="store_true", help='this is for the module comparision in appendix')
    parser.add_argument("--layer_debug_cnt", type=int, default=-1)

    parser.add_argument("--adapterdrop", action="store_true", help='drop first three layers of adaptor')

    parser.add_argument("--partial_learn", action="store_true", help='only learn newly added modules? not used')

    parser.add_argument("--pseudo_ablation", action="store_true", help='pseudo replay ablation study')
    
    parser.add_argument("--deactivate_reuse_adapter", action="store_true", help='pseudo replay ablation study')
    parser.add_argument("--cm_lamol_flag", action="store_true", help='whether use lamol')
    parser.add_argument("--cm_unshared_adapter_layer_list", nargs='+', type=int, default=[], 
                                            help='For given layers, the adapter is unshared for each task.')
    parser.add_argument("--cm_mu", type=float, default=0.08, help='mu: weight for NULL head')

    parser.add_argument("--cm_drop_layers", nargs='+', type=int, default=[], 
                                                help='which layers in the list to drop. It is only useful when adapterdrop is True')

    parser.add_argument("--cm_kd", action="store_true", help='knowledge disllation flag')
    parser.add_argument("--cm_kd_term", type=float, default=0.5, help='weight in KD loss')
    parser.add_argument("--cm_kd_temperature", type=float, default=2, help='temperate for soft target calculation in KD loss')

    parser.add_argument("--cm_kl_v_flag", action="store_true", help='In the Transfer stae, whether to use the kl loss along layers/adapters.')
    parser.add_argument("--cm_v_kl_lambda", type=float, default=0.3, help='weight in calculating the vertical KD loss')
    parser.add_argument("--cm_v_kl_ep", type=int, default=0, help='weight in calculating the vertical KD loss')
    parser.add_argument("--cm_v_kl_typ", type=int, default=0, 
        help='on how to cal the kl/cos loss between adapters, option 0, 1, 2, 3. Go to cm_losses for detail.') 

    parser.add_argument("--cm_kl_h_flag", action="store_true", help='In the Transfer stae, whether to use the kl loss along layers/adapters.')
    parser.add_argument("--cm_h_kl_lambda", type=float, default=0.3, help='weight in calculating the vertical KD loss')
    parser.add_argument("--cm_h_kl_ep", type=int, default=0, help='weight in calculating the vertical KD loss')

    parser.add_argument("--cm_continue_from_task", type=str, default=None,
        help='If the program quit accendently, you can continue training from the given task.')
    parser.add_argument("--cm_ct_n_skip_Mix", action="store_true", help="continue from the transfer stage, ranther than the mix")

    parser.add_argument("--cm_bleu_for_nlg", action="store_true", help='use bleu rather than rouge for nlg tasks')
    parser.add_argument("--cm_test_tasks", nargs='+', default=["squad2"])
    parser.add_argument("--cm_debug", action="store_true")
    parser.add_argument("--cm_test_batch_size", type=int, default=-1, help='for dev/test batch size')
    parser.add_argument("--cm_fwt_dwt_flag", action="store_true", help='get fwt & dwt value')
    parser.add_argument("--cm_case_study_flag", action="store_true", help='generate samples for case study')
    parser.add_argument("--cm_case_study_test_task", nargs='+', default=[], help='which test to be tested for the case study part.')

 
    args = parser.parse_args()

    if args.debug:
        args.logging_steps = 1
        torch.manual_seed(0)
        torch.backends.cudnn.deterministric = True

    args.model_dir_root = os.path.join(args.model_dir_root, args.model_name, args.seq_train_type, 
        "{}_{}_{}_{}".format(args.id, "_".join(args.tasks), args.gen_lm_sample_percentage, args.seed) if "lll" in args.seq_train_type or "finetune" in args.seq_train_type or "llewc" in args.seq_train_type else "_".join(args.tasks)+"_seed_%d"%args.seed)
    logger.info("DIR ROOT CHANGED! {}".format(args.model_dir_root))

    if args.train_batch_size <= 0:
        args.train_batch_size = [28437]   
    if args.test_batch_size <= 0:
        args.test_batch_size = [28437]

    special_tokens = {"ans_token":'__ans__', "pad_token":'__pad__', "unk_token":'__unk__', "eos_token": '<|endoftext|>'}
    if args.use_sep:
        special_tokens["sep_token"] = '__sep__'

    model_class, tokenizer_class, config_class = MODEL_CLASSES[args.model_name]

    while True:
        try:
            tokenizer = tokenizer_class.from_pretrained('gpt2')
            break
        except ValueError:
            continue

    logger.info("TOKENIZER ORIGIN LEN: {}".format(len(tokenizer)))
    if not args.pretraining:
        tokenizer.add_tokens(list(special_tokens.values()))
    special_token_ids = {k:tokenizer.convert_tokens_to_ids(v) for k,v in special_tokens.items()}
    model_config = config_class.from_pretrained('gpt2')
    model_config.vocab_size = len(tokenizer)


    tokens_weight = torch.ones([model_config.vocab_size], dtype=torch.float).cuda()
    if not args.pretraining:
        tokens_weight[special_token_ids["ans_token"]] = args.tokens_weight
        if args.use_sep:
            tokens_weight[special_token_ids["sep_token"]] = args.tokens_weight

    args.max_len = model_config.n_positions - args.preseqlen

    data_attrs_path = os.path.join(BASE_DIR,"data_attrs.json")
    assert os.path.exists(data_attrs_path)
    with open(data_attrs_path, "r") as f:
        data_attrs = json.load(f)

    if args.seq_train_type in ["multitask", "multilm"]:
        args.n_train_epochs = {'_'.join(args.tasks): args.n_train_epochs}
    elif args.unbound:
        pass
    else:
        if "gem" in args.seq_train_type:
            args.memory_data = []
        if args.dynamic_epochs:
            data_sizes = {task: data_attrs[task]["train"]["data_size"] for task in args.tasks}
            max_total_data_size = max(data_sizes.values()) * args.n_train_epochs
            args.n_train_epochs = {d[0]: min(args.max_n_epochs, max_total_data_size//d[1]) for d in data_sizes.items()}
        else:
            args.n_train_epochs = {task: args.n_train_epochs for task in args.tasks}
    return args, model_config, model_class, tokenizer, config_class, special_token_ids, special_tokens, data_attrs, tokens_weight


class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
            last = self.last
        except AttributeError:
            last = record.relativeCreated

        delta = record.relativeCreated/1000 - last/1000
        record.relative = "{:.1f}".format(delta)
        record.uptime = str(datetime.timedelta(seconds=record.relativeCreated//1000))
        self.last = record.relativeCreated
        return True


def init_logging(filename):
    logging_format = "%(asctime)s - %(uptime)s - %(relative)ss - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(format=logging_format, filename=filename, filemode='a', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())

args, MODEL_CONFIG, MODEL_CLASS, TOKENIZER, CONFIG_CLASS, SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, DATA_ATTRS, TOKENS_WEIGHT = parse_args()


TASK_DICT = {
    "NULL_task": {
               "train": os.path.join(args.data_dir,"rnnlg.tv_to_squad-train-v2.0.json"),
               "eval": os.path.join(args.data_dir,"rnnlg.tv_to_squad-test-v2.0.json"),
               "test": os.path.join(args.data_dir,"rnnlg.tv_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "e2enlg": {
               "train": os.path.join(args.data_dir,"e2enlg_to_squad-train-v2.0.json"),
               "eval": os.path.join(args.data_dir,"e2enlg_to_squad-test-v2.0.json"),
               "test": os.path.join(args.data_dir,"e2enlg_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "rnnlg.tv": {
               "train": os.path.join(args.data_dir,"rnnlg.tv_to_squad-train-v2.0.json"),
               "eval": os.path.join(args.data_dir,"rnnlg.tv_to_squad-test-v2.0.json"),
               "test": os.path.join(args.data_dir,"rnnlg.tv_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "rnnlg.hotel": {
               "train": os.path.join(args.data_dir,"rnnlg.hotel_to_squad-train-v2.0.json"),
               "eval": os.path.join(args.data_dir,"rnnlg.hotel_to_squad-test-v2.0.json"),
               "test": os.path.join(args.data_dir,"rnnlg.hotel_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "rnnlg.rest": {
               "train": os.path.join(args.data_dir,"rnnlg.rest_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"rnnlg.rest_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"rnnlg.rest_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "rnnlg.laptop": {
               "train":os.path.join(args.data_dir,"rnnlg.laptop_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"rnnlg.laptop_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"rnnlg.laptop_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    # 
    "cnn_dailymail": {  # summarization, bleu;
               "train":os.path.join(args.data_dir,"cnn_dailymail_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"cnn_dailymail_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"cnn_dailymail_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "srl": {    # qa, bleu;
               "train":os.path.join(args.data_dir,"srl_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"srl_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"srl_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    # cls
    "sst": {   # semantic analysis, acc (binary classification);
               "train":os.path.join(args.data_dir,"sst_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"sst_to_squad-test-v2.0.json"),  
               # there is no dev, so I change 'dev' into 'test';
               "test":os.path.join(args.data_dir,"sst_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "ag": {     # cls, acc; "Is this sentence World, Sports, Business, or Sci/Tech?"
               "train":os.path.join(args.data_dir,"ag_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"ag_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"ag_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "dbpedia": { # reading comprehension, cls, 'acc';
                #    "Is this sentence Company, EducationalInstitution, Artist, Athlete, 
                #    OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, 
                #    Animal, Plant, Album, Film, or WrittenWork?"
               "train":os.path.join(args.data_dir,"dbpedia_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"dbpedia_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"dbpedia_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "yahoo": { # cls, acc;
               "train":os.path.join(args.data_dir,"yahoo_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"yahoo_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"yahoo_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "amazon": { # cls, acc;
               "train":os.path.join(args.data_dir,"amazon_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"amazon_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"amazon_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "yelp": {  # cls, acc; [very very long context;]
               "train":os.path.join(args.data_dir,"yelp_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"yelp_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"yelp_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "woz.en": { # dst, joint_goal_em;
               "train":os.path.join(args.data_dir,"woz.en_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"woz.en_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"woz.en_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "wikisql": { # code generation, lfem
               "train":os.path.join(args.data_dir,"wikisql_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"wikisql_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"wikisql_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    # nlg
    "TM19.restaurant.nlg": {
               "train":os.path.join(args.data_dir,'nlg_data',"TM19-['TM_restaurant']-NLG-train.json"),
               "eval":os.path.join(args.data_dir,'nlg_data',"TM19-['TM_restaurant']-NLG-dev.json"),
               "test":os.path.join(args.data_dir,'nlg_data',"TM19-['TM_restaurant']-NLG-test.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "TM19.movie.nlg": {
               "train":os.path.join(args.data_dir,'nlg_data',"TM19-['TM_movie']-NLG-train.json"),
               "eval":os.path.join(args.data_dir,'nlg_data',"TM19-['TM_movie']-NLG-dev.json"),
               "test":os.path.join(args.data_dir,'nlg_data',"TM19-['TM_movie']-NLG-test.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "TM20.restaurant.nlg": {
               "train":os.path.join(args.data_dir,'nlg_data',"TM20-['TM_restaurant']-NLG-train.json"),
               "eval":os.path.join(args.data_dir,'nlg_data',"TM20-['TM_restaurant']-NLG-dev.json"),
               "test":os.path.join(args.data_dir,'nlg_data',"TM20-['TM_restaurant']-NLG-test.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "TM20.hotel.nlg": {
               "train":os.path.join(args.data_dir,'nlg_data',"TM20-['TM_hotel']-NLG-train.json"),
               "eval":os.path.join(args.data_dir,'nlg_data',"TM20-['TM_hotel']-NLG-dev.json"),
               "test":os.path.join(args.data_dir,'nlg_data',"TM20-['TM_hotel']-NLG-test.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "TM20.flight.nlg": {
               "train":os.path.join(args.data_dir,'nlg_data',"TM20-['TM_flight']-NLG-train.json"),
               "eval":os.path.join(args.data_dir,'nlg_data',"TM20-['TM_flight']-NLG-dev.json"),
               "test":os.path.join(args.data_dir,'nlg_data',"TM20-['TM_flight']-NLG-test.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "TM20.movie.nlg": {
               "train":os.path.join(args.data_dir,'nlg_data',"TM20-['TM_movie']-NLG-train.json"),
               "eval":os.path.join(args.data_dir,'nlg_data',"TM20-['TM_movie']-NLG-dev.json"),
               "test":os.path.join(args.data_dir,'nlg_data',"TM20-['TM_movie']-NLG-test.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "MWOZ.restaurant.nlg": {
               "train":os.path.join(args.data_dir,'nlg_data',"MWOZ-['MWOZ_restaurant']-NLG-train.json"),
               "eval":os.path.join(args.data_dir,'nlg_data',"MWOZ-['MWOZ_restaurant']-NLG-dev.json"),
               "test":os.path.join(args.data_dir,'nlg_data',"MWOZ-['MWOZ_restaurant']-NLG-test.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "MWOZ.hotel.nlg": {
               "train":os.path.join(args.data_dir,'nlg_data',"MWOZ-['MWOZ_hotel']-NLG-train.json"),
               "eval":os.path.join(args.data_dir,'nlg_data',"MWOZ-['MWOZ_hotel']-NLG-dev.json"),
               "test":os.path.join(args.data_dir,'nlg_data',"MWOZ-['MWOZ_hotel']-NLG-test.json"),
               "n_train_epochs": args.n_train_epochs
    },
    "SGD.restaurant.nlg": {
               "train":os.path.join(args.data_dir,'nlg_data',"SGD-['sgd_restaurants']-NLG-train.json"),
               "eval":os.path.join(args.data_dir,'nlg_data',"SGD-['sgd_restaurants']-NLG-dev.json"),
               "test":os.path.join(args.data_dir,'nlg_data',"SGD-['sgd_restaurants']-NLG-test.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "SGD.hotel.nlg": {
               "train":os.path.join(args.data_dir,'nlg_data',"SGD-['sgd_hotels']-NLG-train.json"),
               "eval":os.path.join(args.data_dir,'nlg_data',"SGD-['sgd_hotels']-NLG-dev.json"),
               "test":os.path.join(args.data_dir,'nlg_data',"SGD-['sgd_hotels']-NLG-test.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "SGD.flight.nlg": {
               "train":os.path.join(args.data_dir,'nlg_data',"SGD-['sgd_flights']-NLG-train.json"),
               "eval":os.path.join(args.data_dir,'nlg_data',"SGD-['sgd_flights']-NLG-dev.json"),
               "test":os.path.join(args.data_dir,'nlg_data',"SGD-['sgd_flights']-NLG-test.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "SGD.movie.nlg": {
               "train":os.path.join(args.data_dir,'nlg_data',"SGD-['sgd_movies']-NLG-train.json"),
               "eval":os.path.join(args.data_dir,'nlg_data',"SGD-['sgd_movies']-NLG-dev.json"),
               "test":os.path.join(args.data_dir,'nlg_data',"SGD-['sgd_movies']-NLG-test.json"),
               "n_train_epochs": args.n_train_epochs 
    },
 
}

METRIC_DICT = {
    "e2enlg": 'bleu',
    "rnnlg.tv": 'bleu',
    "rnnlg.hotel": 'bleu',
    "rnnlg.rest": 'bleu',
    "rnnlg.laptop": 'bleu',
    "cnn_dailymail": 'bleu',       
    "srl": 'bleu',                
    "TM19.restaurant.nlg": 'bleu',
    "TM19.movie.nlg": 'bleu',
    "TM20.restaurant.nlg": 'bleu',
    "TM20.hotel.nlg": 'bleu',
    "TM20.flight.nlg": 'bleu',
    "TM20.movie.nlg": 'bleu',
    "MWOZ.restaurant.nlg": 'bleu',
    "MWOZ.hotel.nlg": 'bleu',
    "SGD.restaurant.nlg": 'bleu',
    "SGD.hotel.nlg": 'bleu',
    "SGD.flight.nlg": 'bleu',
    "SGD.movie.nlg": 'bleu',
   
}


TASK_TYPE_DICT = {
    "e2enlg": 'nlg',
    "rnnlg.tv": 'nlg',
    "rnnlg.hotel": 'nlg',
    "rnnlg.rest": 'nlg',
    "rnnlg.laptop": 'nlg',
    "cnn_dailymail": 'summarization',       
    "srl": 'nlg',            

    "TM19.restaurant.nlg": 'nlg',
    "TM19.movie.nlg": 'nlg',
    "TM20.restaurant.nlg": 'nlg',
    "TM20.hotel.nlg": 'nlg',
    "TM20.flight.nlg": 'nlg',
    "TM20.movie.nlg": 'nlg',
    "MWOZ.restaurant.nlg": 'nlg',
    "MWOZ.hotel.nlg": 'nlg',
    "SGD.restaurant.nlg": 'nlg',
    "SGD.hotel.nlg": 'nlg',
    "SGD.flight.nlg": 'nlg',
    "SGD.movie.nlg": 'nlg',
 
}

