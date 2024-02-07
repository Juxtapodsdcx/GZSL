import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup,AutoModelForMaskedLM

import sys
sys.path.append("../../")
from gzsl.classification.data_utils.dataset import get_dataset
from gzsl.classification.model.model import IntentClassifierNLI,SoftpromptIntentClassifierNLI
from gzsl.classification.model.losses import get_loss,get_ce_loss
from gzsl.classification.util.environment import seed_everything
from gzsl.classification.util.loops import train
from gzsl.classification.util.preprocessing import read_intent_info, read_decomposed_intents, \
    read_intent_similarity_matrix, read_split_data, read_uttr_similarity_matrix


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    seed_everything(cfg.experiment.seed)
    tensorboard_dir = os.path.join(os.getcwd(), 'tensorboard')
    checkpoints_dir = os.path.join(os.getcwd(), 'checkpoints')

    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    os.makedirs(checkpoints_dir, exist_ok=True)

    logging.info(f'Read and preprocesed data from {cfg.dataset.path}')
    data_path = Path(to_absolute_path(cfg.dataset.path))
    intent_info_path = Path(to_absolute_path(cfg.dataset.intent_info_path))
    intents, descriptions = read_intent_info(intent_info_path, cfg.dataset.description_type, cfg.dataset.name)
    concepts, actions = read_decomposed_intents(intent_info_path)
    zeroshot_intents, train_df, dev_df, test_df = read_split_data(data_path, intents, cfg.experiment.seed)
    logging.info(f'Unseen intents: {zeroshot_intents}')
    seen_intents = [intent for intent in intents if intent not in zeroshot_intents]

    train_intents = seen_intents if cfg.experiment.train_only_seen else intents

    if cfg.experiment.sim_matrix:
        if cfg.experiment.sampling_strategy == 'intents':
            similarity_matrix_intent = read_intent_similarity_matrix(intent_info_path, cfg.experiment.sim_matrix, train_intents)
        elif cfg.experiment.sampling_strategy == 'utterances':
            similarity_matrix_utter = read_uttr_similarity_matrix(data_path, cfg.experiment.sim_matrix, train_df.index.values)
        else:
            raise ValueError('Unexpected sampling strategy')
    else:
        similarity_matrix_intent = read_intent_similarity_matrix(intent_info_path, "intent_similarity/simcse", train_intents)
        similarity_matrix_utter = read_uttr_similarity_matrix(data_path, "sentence_bert_100.txt", train_df.index.values)



    tokenizer = AutoTokenizer.from_pretrained(to_absolute_path(cfg.model.base_model))
    train_loader = DataLoader(
        get_dataset(cfg, train_df, tokenizer, train_intents, descriptions, concepts, actions,"train",cfg.experiment.mlm_percent,
                    similarity_matrix_intent,similarity_matrix_utter,cfg.experiment.sampling_strategy, cfg.experiment.k_negative,
),
        batch_size=cfg.experiment.batch_size, shuffle=True
    )
    dev_loader = DataLoader(
        get_dataset(cfg, dev_df, tokenizer, intents, descriptions, concepts, actions,"evaluate"),
        batch_size=cfg.experiment.batch_size
    )

    logging.info('Init model...')
    device = torch.device('cuda:'+str(cfg.experiment.cuda) if torch.cuda.is_available() else 'cpu')

    logging.info("-"*50)
    logging.info(f'seed: {cfg.experiment.seed}')
    logging.info(f'device: {device}')
    logging.info(f'description_type: {cfg.dataset.description_type}')
    logging.info(f'embedding_params: {cfg.experiment.embedding_param}')
    logging.info(f'k_negs: {cfg.experiment.k_negative}')
    logging.info(f'mlm_percents: {cfg.experiment.mlm_percent}')
    logging.info("-"*50)

    base_model = AutoModelForMaskedLM.from_pretrained(to_absolute_path(cfg.model.base_model)).to(device)
    if hasattr(base_model, 'model'):
        base_model = base_model.model


    model = IntentClassifierNLI(base_model, hidden_size=cfg.model.embedding_dim, dropout=cfg.model.dropout).to(
            device)

    criterion = get_loss(cfg)
    CEcriterion = get_ce_loss(temperature=cfg.experiment.temperature)

    start_epoch = 1
    best_acc = 0.
    if cfg.checkpoint.saved_model:
        loaded_cp = torch.load(to_absolute_path(cfg.checkpoint.saved_model))
        start_epoch = loaded_cp['epoch'] + 1
        best_acc = loaded_cp['acc_val']
        model.load_state_dict(loaded_cp['model'].state_dict())
        logging.info(f'Loading model: {cfg.checkpoint.saved_model}, start epoch: {start_epoch}')

    accum_steps = cfg.experiment.accum_steps
    optimizer = AdamW(model.parameters(), lr=cfg.scheduler.lr)
    train_steps = len(train_loader) * cfg.experiment.epochs / cfg.experiment.accum_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.scheduler.warmup_steps * train_steps,
                                                num_training_steps=train_steps)
    OmegaConf.save(cfg, 'config.yaml')
    train(
        model, criterion,CEcriterion, optimizer, scheduler, len(intents),
        train_loader, dev_loader, device,
        writer, checkpoints_dir,
        print_every=cfg.log.print_every,
        n_epoch=cfg.experiment.epochs, accum_steps=accum_steps,
        save_from_epoch=cfg.checkpoint.save_from_epoch,
        start_epoch=start_epoch, best_acc=best_acc,
        model_type=cfg.model.model_type,
        embedding_param=cfg.experiment.embedding_param,
        mlm_param=cfg.experiment.mlm_param,

    )


if __name__ == '__main__':
    main()
