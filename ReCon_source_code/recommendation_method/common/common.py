import os
import copy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from data_handler.data_common import get_data
from recommendation_method.common.custom_early_stopping import LogicalOREarlyStopping
from pytorch_lightning.callbacks.callback import Callback


class NegSamplingCallback(Callback):
	def __init__(self, train_loader, val_loader) -> None:
		super().__init__()
		self.train_loader = train_loader
		self.val_loader = val_loader

	def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
		"""Called when the val epoch begins."""
		self.val_loader.dataset.ng_sample()
		# print('validation loader ng_sample')

	def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
		"""Called when the train epoch begins."""
		self.train_loader.dataset.ng_sample()
		# print('train loader ng_sample')


def train(args, config, r_model, ModelClass, model_name, model_file_path, epochs, patience, ot_param_dict=None, dependent_models=None, monitor=None):
    	# min_delta = 0.001
	min_delta = 0.0000001
	if monitor is None:
		monitor = args.monitor
	monitor_metrics = monitor.split(',')
	
	user_num,item_num,top_k_list_for_rec_performance,top_k_list,train_loader,test_loader_classification, test_loader_ranking, val_loader = get_data(args, config)
	# initialize trainer  
	if args.early_stop:
		if ',' in monitor:
			callbacks = [LogicalOREarlyStopping(monitor=monitor_metrics, min_delta=min_delta, 
	patience=patience, verbose=True, mode=args.mode)]
		else:
			print("EarlyStopping initialized with patience %d" % patience)
			callbacks = [EarlyStopping(monitor=monitor, min_delta=min_delta, 
	patience=patience, verbose=True, mode=args.mode)]
	else:
		callbacks = []      
	callbacks.append(NegSamplingCallback(train_loader, val_loader))
	checkpoint_callbacks = list()
	for m in monitor_metrics:
		checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=m, mode=args.mode)
		checkpoint_callbacks.append(checkpoint_callback)
		callbacks.append(checkpoint_callback)

	r_model = r_model.cuda()
	if args.gradient_clipping > 0:
		gradient_clip_val = args.gradient_clipping
	else:
		gradient_clip_val = None
	print(callbacks)
	trainer = pl.Trainer(callbacks=callbacks, num_sanity_val_steps=0, enable_checkpointing=True, gradient_clip_val=gradient_clip_val, precision=args.precision,
		    default_root_dir=os.path.join(args.model_path, 'model_checkpoints', model_file_path.split('/')[-1].replace('.pth', '')),
			logger=None, min_epochs=args.min_epochs, max_epochs=epochs, 
			accelerator='gpu', devices=1, check_val_every_n_epoch=10)

	# print(r_model.state_dict())

	# train    
	r_model.train()
	r_model = r_model.cuda()
	try:
		trainer.fit(r_model, train_loader, val_loader)
	except Exception as e:
		print(e)
		if 'Input contains NaN' not in str(e):
			raise e

	# ff multiple metrics are used
	best_checkpoint_epoch = -1
	best_checkpoint_path = ''
	for checkpoint_callback in checkpoint_callbacks:
		checkpoint_epoch = int(checkpoint_callback.best_model_path.split('epoch=')[1].split('-step')[0])
		if checkpoint_epoch > best_checkpoint_epoch:
			print(checkpoint_callback.monitor, checkpoint_callback.best_model_score)
			best_checkpoint_epoch = checkpoint_epoch
			best_checkpoint_path = checkpoint_callback.best_model_path


	best_model = ModelClass.load_from_checkpoint(best_checkpoint_path)
	if model_name == 'cne':
		best_model.init_part2(args.use_ot, ot_param_dict, dependent_models, user_num, item_num)


	return best_model
