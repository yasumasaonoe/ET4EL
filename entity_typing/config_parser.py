import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model_id", help="Identifier for model")
# Data
parser.add_argument("-train_data", help="Train data", default="train/wiki_et_conll_1k_random/train_*.json")
parser.add_argument("-dev_data", help="Dev data", default="validation/dev_wiki_et_conll_1k_random.json")
parser.add_argument("-eval_data", help="Test data", default="")
parser.add_argument("-num_epoch", help="The number of epoch", default=5000, type=int)
parser.add_argument("-batch_size", help="The batch size", default=100, type=int)
parser.add_argument("-eval_batch_size", help="The batch size", default=100, type=int)
parser.add_argument("-goal", help="category vocab.", default="conll_60k", choices=["conll_60k", "unseen_60k"])
parser.add_argument("-seed", help="Pytorch random Seed", default=1888)
parser.add_argument("-gpu", help="Using gpu or cpu", default=False, action="store_true")
parser.add_argument("-max_context_length", help="The max length of context", default=50, type=int)
parser.add_argument("-drop_stopwords", help="Drop stop words.", default=False, action='store_true')
parser.add_argument("-replace_numbers", help="Replace numbers.", default=False, action='store_true')

# learning
parser.add_argument("-mode", help="Whether to train or test", default="train", choices=["train", "test"])
parser.add_argument("-learning_rate", help="start learning rate", default=0.001, type=float)
parser.add_argument("-mention_dropout", help="drop out rate for mention", default=0.5, type=float)
parser.add_argument("-input_dropout", help="drop out rate for sentence", default=0.2, type=float)
parser.add_argument("-loss_neg_weight", help="weight on negative examples in loss", default=0.1, type=float)
parser.add_argument("-multi_gpu", help="Use multi GPUs.", default=False, action='store_true')

# Model
parser.add_argument("-model_type", default="ETModel", choices=["ETModel"])
parser.add_argument("-enhanced_mention", help="Use attention and cnn for mention representation", default=False, action='store_true')
parser.add_argument("-dim_hidden", help="The number of hidden dimension.", default=100, type=int)
parser.add_argument("-rnn_dim", help="The number of RNN dimension.", default=100, type=int)
parser.add_argument("-mention_lstm", help="Using LSTM for mention embedding.", default=False, action='store_true')
parser.add_argument("-custom_loss", help="Using custom loss.", default=False, action='store_true')
parser.add_argument("-elmo", help="Using ELMo.", default=False, action='store_true')
parser.add_argument("-threshold", help="threshold", default=0.5, type=float)
parser.add_argument("-annonym_mention", help="Annonymize mention span.", default=False, action='store_true')

# Save / log related
parser.add_argument("-save_period", help="How often to save", default=1000, type=int)
parser.add_argument("-eval_period", help="How often to run dev", default=1000, type=int)
parser.add_argument("-log_period", help="How often to save", default=1000, type=int)
parser.add_argument("-eval_after", help="Eval after X updates", default=1000, type=int)
parser.add_argument("-save_after", help="Save after Y updates", default=1000, type=int)

parser.add_argument("-load", help="Load existing model.", action='store_true')
parser.add_argument("-reload_model_name", help="")
