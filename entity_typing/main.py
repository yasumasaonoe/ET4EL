#!/usr/bin/env python3
import datetime
import gc
import json
import logging
import pickle
import time
import torch
from torch import optim
from tqdm import tqdm

import config_parser
import constant
import data_utils
import models
import eval_metric
from data_utils import to_torch
from model_utils import get_gold_pred_str, get_eval_string, get_output_index


def get_data_gen(dataname, mode, args, char_vocab, elmo=None):
  data_path = constant.FILE_ROOT + dataname
  dataset = data_utils.ETDataset(data_path, vocab=char_vocab, elmo=elmo, args=args)
  if mode == 'train':
    data_gen = dataset.get_batch(args.batch_size, args.num_epoch, forever=False, eval_data=False)
  else:
    data_gen = dataset.get_batch(args.eval_batch_size, 1, forever=False, eval_data=True)
  return data_gen


def get_all_datasets(args):
  char_vocab = constant.CHAR_DICT
  elmo = data_utils.init_elmo()
  print('==> Embedding: ELMo')
  train_gen_list = []
  if args.mode == 'train':
    train_gen_list.append(get_data_gen(args.train_data,'train', args, char_vocab, elmo=elmo))
  return train_gen_list, elmo, char_vocab


def get_datasets(data_lists, args):
  data_gen_list = []
  char_vocab = constant.CHAR_DICT
  elmo = data_utils.init_elmo()
  print('==> Embedding: ELMo')
  for dataname, mode in data_lists:
    data_gen_list.append(get_data_gen(dataname, mode, args, char_vocab, elmo=elmo))
  return data_gen_list, elmo, char_vocab


def _train(args, device):
  print('==> Loading data generator... ')
  train_gen_list, elmo, char_vocab = get_all_datasets(args)

  if args.model_type == 'ETModel':
    print('==> ETModel')
    model = models.ETModel(args, constant.ANSWER_NUM_DICT[args.goal])
  else:
    print('ERROR: Invalid model type: -model_type ' + args.model_type)
    raise NotImplementedError

  model.to(device)
  total_loss = 0
  batch_num = 0
  best_macro_f1 = 0.
  start_time = time.time()
  init_time = time.time()

  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

  if args.load:
    load_model(args.reload_model_name, constant.EXP_ROOT, args.model_id, model, optimizer)

  for idx, m in enumerate(model.modules()):
    logging.info(str(idx) + '->' + str(m))

  while True:
    batch_num += 1
    for data_gen in train_gen_list:
      try:
        batch = next(data_gen)
        batch, _ = to_torch(batch, device)
      except StopIteration:
        logging.info('finished at ' + str(batch_num))
        print('Done!')
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   '{0:s}/{1:s}.pt'.format(constant.EXP_ROOT, args.model_id))
        return
      optimizer.zero_grad()
      loss, output_logits, _ = model(batch)
      loss.backward()
      total_loss += loss.item()
      optimizer.step()

      if batch_num % args.log_period == 0 and batch_num > 0:
        gc.collect()
        cur_loss = float(1.0 * loss.clone().item())
        elapsed = time.time() - start_time
        train_loss_str = ('|loss {0:3f} | at {1:d}step | @ {2:.2f} ms/batch'.format(cur_loss, batch_num,
                                                                                    elapsed * 1000 / args.log_period))
        start_time = time.time()
        print(train_loss_str)
        logging.info(train_loss_str)

      if batch_num % args.eval_period == 0 and batch_num > 0:
        output_index = get_output_index(output_logits, threshold=args.threshold)
        gold_pred_train = get_gold_pred_str(output_index, batch['y'].data.cpu().clone(), args.goal)
        #print(gold_pred_train[:10])
        accuracy = sum([set(y) == set(yp) for y, yp in gold_pred_train]) * 1.0 / len(gold_pred_train)
        train_acc_str = '==> Train accuracy: {0:.1f}%'.format(accuracy * 100)
        print(train_acc_str)
        logging.info(train_acc_str)

    if batch_num % args.eval_period == 0 and batch_num > args.eval_after:
      print('---- eval at step {0:d} ---'.format(batch_num))
      _, macro_f1 = evaluate_data(
        batch_num, args.dev_data, model, args, elmo, device, char_vocab, dev_type='original'
      )

      if best_macro_f1 < macro_f1:
        best_macro_f1 = macro_f1
        save_fname = '{0:s}/{1:s}_best.pt'.format(constant.EXP_ROOT, args.model_id)
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_fname)
        print(
          'Total {0:.2f} minutes have passed, saving at {1:s} '.format((time.time() - init_time) / 60, save_fname))

    if batch_num % args.save_period == 0 and batch_num > args.save_after:
      save_fname = '{0:s}/{1:s}_{2:d}.pt'.format(constant.EXP_ROOT, args.model_id, batch_num)
      torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_fname)
      print(
        'Total {0:.2f} minutes have passed, saving at {1:s} '.format((time.time() - init_time) / 60, save_fname))


def evaluate_data(batch_num, dev_fname, model, args, elmo, device, char_vocab, dev_type='original'):
  model.eval()
  dev_gen = get_data_gen(dev_fname, 'test', args, char_vocab, elmo=elmo)
  gold_pred = []
  eval_loss = 0.
  total_ex_count = 0
  for batch in tqdm(dev_gen):
    total_ex_count += len(batch['y'])
    eval_batch, annot_ids = to_torch(batch, device)
    loss, output_logits, _ = model(eval_batch)
    output_index = get_output_index(output_logits, threshold=args.threshold)
    gold_pred += get_gold_pred_str(output_index, eval_batch['y'].data.cpu().clone(), args.goal)
    eval_loss += loss.clone().item()
  eval_str = get_eval_string(gold_pred)
  _, _, _, _, _, macro_f1 = eval_metric.macro(gold_pred)
  eval_loss_str = 'Eval loss: {0:.7f} at step {1:d}'.format(eval_loss, batch_num)
  print('==> ' + dev_type + ' EVAL: seen ' + repr(total_ex_count) + ' examples.')
  print(eval_loss_str)
  print(gold_pred[:3])
  print('==> ' + dev_type + ' : ' + eval_str)
  logging.info(eval_loss_str)
  logging.info(eval_str)
  model.train()
  return eval_loss, macro_f1


def load_model(reload_model_name, save_dir, model_id, model, optimizer=None):
  if reload_model_name:
    model_file_name = '{0:s}/{1:s}.pt'.format(save_dir, reload_model_name)
  else:
    model_file_name = '{0:s}/{1:s}.pt'.format(save_dir, model_id)
  checkpoint = torch.load(model_file_name)
  model.load_state_dict(checkpoint['state_dict'])
  if optimizer:
    optimizer.load_state_dict(checkpoint['optimizer'])
  else:
    total_params = 0
    # Log params
    for k in checkpoint['state_dict']:
      elem = checkpoint['state_dict'][k]
      param_s = 1
      for size_dim in elem.size():
        param_s = size_dim * param_s
      print(k, elem.size())
      total_params += param_s
    param_str = ('Number of total parameters..{0:d}'.format(total_params))
    logging.info(param_str)
    print(param_str)
  logging.info("Loading old file from {0:s}".format(model_file_name))
  print('Loading model from ... {0:s}'.format(model_file_name))


def _test(args, device):
  assert args.load
  test_fname = args.eval_data
  data_gens, _, _ = get_datasets([(test_fname, 'test')], args)
  if args.model_type == 'ETModel':
    print('==> ETModel')
    model = models.ETModel(args, constant.ANSWER_NUM_DICT[args.goal])
  else:
    print('Invalid model type: -model_type ' + args.model_type)
    raise NotImplementedError
  model.to(device)
  model.eval()
  load_model(args.reload_model_name, constant.EXP_ROOT, args.model_id, model)
  if args.multi_gpu:
    model = torch.nn.DataParallel(model)
    print("==> use", torch.cuda.device_count(), "GPUs.")
  for name, dataset in [(test_fname, data_gens[0])]:
    print('Processing... ' + name)
    total_gold_pred = []
    total_annot_ids = []
    total_probs = []
    total_ys = []
    for batch_num, batch in enumerate(dataset):
      if batch_num % 10 == 0:
        print(batch_num)
      if not isinstance(batch, dict):
        print('==> batch: ', batch) 
      eval_batch, annot_ids = to_torch(batch, device)
      if args.multi_gpu:
        output_logits = model(eval_batch)
      else:
        _, output_logits, _ = model(eval_batch)
      output_index = get_output_index(output_logits, threshold=args.threshold)
      output_prob = model.sigmoid_fn(output_logits).data.cpu().clone().numpy()
      y = eval_batch['y'].data.cpu().clone().numpy()
      gold_pred = get_gold_pred_str(output_index, y, args.goal)
      total_probs.extend(output_prob)
      total_ys.extend(y)
      total_gold_pred.extend(gold_pred)
      total_annot_ids.extend(annot_ids)
    pickle.dump({'gold_id_array': total_ys, 'pred_dist': total_probs},
                open(constant.OUT_ROOT + '{0:s}.pkl'.format(args.model_id), "wb"))
    print(len(total_annot_ids), len(total_gold_pred))
    with open(constant.OUT_ROOT + '{0:s}.json'.format(args.model_id), 'w') as f_out:
      output_dict = {}
      counter = 0
      for a_id, (gold, pred) in zip(total_annot_ids, total_gold_pred):
        output_dict[a_id] = {"gold": gold, "pred": pred}
        counter += 1
      json.dump(output_dict, f_out)
    logging.info('processing: ' + name)
  print('Done!')


if __name__ == '__main__':
  device = torch.device("cuda")
  config = config_parser.parser.parse_args()
  print('-- Args ' + '-' * 72)
  for k, v in vars(config).items():
      print('{:>20} : {:}'.format(k, v))
  print('-' * 80)
  torch.cuda.manual_seed(config.seed)
  logging.basicConfig(
    filename=constant.OUT_ROOT + "/" + config.model_id + datetime.datetime.now().strftime("_%m-%d_%H") + config.mode +
             '.txt',
    level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
  logging.info(config)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  if config.model_type == 'ETModel':
    config.elmo = True
  else:
    print('ERROR: Invalid model type {}'.format(config.model_type))
    raise NotImplementedError
  if config.mode == 'train':
    print('==> mode: train')
    _train(config, device)
  elif config.mode == 'test':
    print('==> mode: test')
    _test(config, device)
  else:
    raise ValueError("ERROR: Invalid value for 'mode': {}".format(config.mode))
