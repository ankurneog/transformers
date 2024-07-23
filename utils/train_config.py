from utils.device import Device

class TrainConfig:

    def __init__(self,
            device : str = Device("cpu"),
            train_batch_size=None,
            eval_batch_size=None,
            epoch=None,
            eval_iter=None, # evaluate interval for  training vs validation loss 
            log_interval=None, # logging interval , usually every 100th iteration
            weight_decay=None, # weight decay regularization to prevent over-fitting
            lr_decay_interval=None,
            activation_checkpointing=False, # discard recompute activations during backward pass , trading off compute for memory 
            gradient_acc_steps=False ,# to accomodate large batch sizes
            learning_rate=False,
            do_train = False,
            do_eval = False
    ):
        self.device = device
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.epoch = epoch
        self.eval_iter = eval_iter
        self.log_interval = log_interval
        self.weight_decay = weight_decay
        self.lr_decay_interval = lr_decay_interval
        self.activation_checkpointing = activation_checkpointing
        self.gradient_acc_steps = gradient_acc_steps
        self.learning_rate = learning_rate
        self.do_train = do_train
        self.do_eval = do_eval