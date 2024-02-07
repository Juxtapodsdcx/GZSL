import hydra
from hydra.utils import to_absolute_path
import os
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from classification.data_utils.dataset import get_dataset
from classification.model.model import IntentClassifierNLI
from classification.util.environment import seed_everything
from classification.util.loops import encode
from classification.util.preprocessing import read_intent_info, read_decomposed_intents, read_split_data


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    logging.basicConfig(level=logging.INFO)
    seed_everything(cfg.experiment.seed)
    intent_info_path = Path(to_absolute_path(cfg.dataset.intent_info_path))
    intents, descriptions = read_intent_info(intent_info_path, cfg.dataset.description_type)
    concepts, actions = read_decomposed_intents(intent_info_path)

    logging.info(f'Read and preprocesed data from {cfg.dataset.path}')
    data_path = Path(to_absolute_path(cfg.dataset.path))
    zeroshot_intents, train_df, dev_df, test_df = read_split_data(data_path, intents, cfg.experiment.seed)
    logging.info(f'Unseen intents: {zeroshot_intents}')
    seen_intents = [intent for intent in intents if intent not in zeroshot_intents]

    tokenizer = AutoTokenizer.from_pretrained(to_absolute_path(cfg.model.base_model))
    train_loader = DataLoader(
        get_dataset(cfg, train_df, tokenizer, seen_intents, descriptions, concepts, actions),
        batch_size=cfg.experiment.batch_size
    )
    test_loader = DataLoader(
        get_dataset(cfg, test_df, tokenizer, intents, descriptions, concepts, actions),
        batch_size=cfg.experiment.batch_size
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = AutoModel.from_pretrained(to_absolute_path(cfg.model.base_model)).to(device)
    if hasattr(base_model, 'model'):
        base_model = base_model.model
    model = IntentClassifierNLI(base_model, hidden_size=cfg.model.embedding_dim, dropout=cfg.model.dropout).to(device)

    model_name = f'checkpoint_e{cfg.experiment.test_epoch}' if cfg.experiment.test_epoch else 'checkpoint_best_acc'
    model_path = os.path.join(os.getcwd(), 'checkpoints', model_name + '.pt')
    logging.info(f'Loading model: {model_path}')
    model.load_state_dict(torch.load(model_path)['model'].state_dict())

    logging.info('Start encoding...')
    y_train, feats_train = encode(model, train_loader, len(seen_intents), device, cfg.model.embedding_dim)
    true_labels_train = np.array([seen_intents[int(l)] for l in y_train])

    y_test, feats_test = encode(model, test_loader, len(intents), device, cfg.model.embedding_dim)
    true_labels_test = np.array([intents[int(l)] for l in y_test])

    encoding_dir = f'./encoding_{model_name}/'
    os.makedirs(encoding_dir, exist_ok=True)
    with open(f'{encoding_dir}/summary.json', 'w') as f:
        json.dump({
            'true_test': true_labels_test.tolist(),
            'y_test': y_test.tolist(),
            'true_train': true_labels_train.tolist(),
            'y_train': y_train.tolist(),
            'intents': intents,
            'seen_intents': seen_intents,
            'zeroshot_intents': zeroshot_intents
        }, f)

    np.save(f'./encoding_{model_name}/train.npy', feats_train)
    np.save(f'./encoding_{model_name}/test.npy', feats_test)


if __name__ == '__main__':
    main()
