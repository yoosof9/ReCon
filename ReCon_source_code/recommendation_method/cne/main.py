import copy
import torch
import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch.backends.cudnn as cudnn
import os
import time
import argparse
from ot_local.ot_plugin import OTPlugin
from data_handler.config import DatasetConfig
import evaluate
from ot_local.ot_evaluation import OTEvaluation
from data_handler.data_common import get_data
from recommendation_method.common.common import train
from recommendation_method.cne.model import IdentityModel


def get_ot_param_dict(args, train_loader, user_num, item_num):
	ot_plugin = OTPlugin(user_num, item_num, args.sinkhorn_gamma, args.sinkhorn_maxiter)
	user2, item2 = None, None
	if (args.use_ot and args.ot_method == 'all') or (not args.use_ot and user_num+item_num < 9000):
		user2 = torch.LongTensor([int(u / item_num) for u in range(0, user_num * item_num)])
		item2 = torch.LongTensor([int(i % item_num) for i in range(0, user_num * item_num)])
		if torch.cuda.is_available():
			user2 = user2.cuda()
			item2 = item2.cuda()

	if args.use_ot == 1:
		ot_param_dict = {"ot_plugin": ot_plugin, "lambda_p": args.lambda_p, 'train_dataset': train_loader.dataset}
		if args.ot_method == 'all':
			if torch.cuda.is_available():
				user2 = user2.cuda()
				item2 = item2.cuda()
			ot_param_dict.update({"user2": user2, "item2": item2})
		else:
			ot_param_dict.update({"user2": None, "item2": None})
	else:
		ot_param_dict = {"ot_plugin": ot_plugin, "lambda_p": args.lambda_p, 'train_dataset': train_loader.dataset, "user2": user2, "item2": item2}
	return user2, item2, ot_param_dict, ot_plugin


parser = argparse.ArgumentParser()
parser.add_argument("--path_to_config", 
	type=str,
	default='recommendation_method/cne/config_career_builder_small_recon.yml', 
	help="model path for data")
parser.add_argument("--base_path_to_config", 
	type=str,
	default='recommendation_method/cne/config_career_builder_small_main.yml', 
	help="model path for data")

parser.add_argument("--batch_size", 
	type=int, 
	default=2048, 
	help="batch size for training")
parser.add_argument("--base_batch_size", 
	type=int, 
	default=2048, 
	help="batch size for training")

parser.add_argument("--min_epochs", 
	type=int,
	default=0,  
	help="training epoches")
parser.add_argument("--epochs", 
	type=int,
	default=50000,  
	help="training epoches")
parser.add_argument("--base_epochs", 
	type=int,
	default=50000,  
	help="training epoches")
parser.add_argument("--base_training_repeat", 
	type=int,
	default=1, 
	help="# repeating training")
parser.add_argument("--base_monitor", 
	type=str,
	default='valid_auc', 
	help="valid_loss, valid_auc")

parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--out", 
	type=int,
	default=1,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
parser.add_argument("--use_ot", 
	type=int,
	default=1, 
	help="0 or 1")
parser.add_argument("--lambda_p", 
	type=float,
	default=0.005, 
	help="regularization weight")
parser.add_argument("--sinkhorn_gamma", 
	type=float,
	default=10.0, 
	help="sinkhorn gamma")
parser.add_argument("--sinkhorn_maxiter", 
	type=int,
	default=100, 
	help="sinkhorn max iterations")
parser.add_argument("--dataset", 
	type=str,
	default='career_builder', 
	help="")
parser.add_argument("--main_path", 
	type=str,
	default='Data/', 
	help="main path for data")
parser.add_argument("--model_path", 
	type=str,
	default='models/', 
	help="model path for data")
parser.add_argument("--figures_path", 
	type=str,
	default='figures/', 
	help="path to save the figures")
parser.add_argument("--use_pretrained", 
	type=int,
	default=0, 
	help="not used")
parser.add_argument("--early_stop", 
	type=int,
	default=1, 
	help="do early stopping?")
parser.add_argument("--retraining_patience", 
	type=int,
	default=1, 
	help="retraining patience in early stopping")
parser.add_argument("--base_retraining_patience", 
	type=int,
	default=1, 
	help="retraining patience in early stopping")

parser.add_argument("--patience", 
	type=int,
	default=200, 
	help="patience in early stopping")
parser.add_argument("--base_patience", 
	type=int,
	default=200, 
	help="patience in early stopping")

parser.add_argument("--monitor", 
	type=str,
	default='valid_minus_congestion', 
	help="valid_loss, valid_auc")

parser.add_argument("--mode", 
	type=str,
	default='max', 
	help="min or max")
parser.add_argument("--ot_method", 
	type=str,
	default='all', 
	help="")
parser.add_argument("--cluster_path", 
	type=str,
	default='', 
	help="model path for data")
parser.add_argument("--training_repeat", 
	type=int,
	default=1, 
	help="# repeating training")
parser.add_argument("--train_if_found", 
	type=int,
	default=0, 
	help="whether to continue training if the model is found or not")
parser.add_argument("--popularity", 
	type=int,
	default=1, 
	help="whether to add popularity model or not")
parser.add_argument("--gradient_clipping", 
	type=float,
	default=0, 
	help="gradient clipping")
parser.add_argument("--precision", 
	type=int,
	default=32, 
	help="mixed precision")

if __name__ == "__main__":
	args = parser.parse_args()
	print(args)
	config = DatasetConfig('cne', args.model_path, args.main_path, args.dataset)

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	cudnn.benchmark = True

	user_num,item_num,top_k_list_for_rec_performance,top_k_list,train_loader,test_loader_classification, test_loader_ranking, val_loader = get_data(args, config)
	print(user_num)
	print(item_num)

	print("load config")
	cne_config = OmegaConf.load(args.path_to_config)
	base_cne_config = OmegaConf.load(args.base_path_to_config)

	popularity_epochs = 50
	popularity_patience = 5
	popularity_model_file_path = '{}popularity_{}_{}_{}_{}_{}_{}_{}.pth'.format(args.model_path, args.dataset, args.base_batch_size, popularity_epochs, args.base_monitor, args.early_stop, popularity_patience, args.num_ng)
	base_model_file_path = '{}cne_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_dim_{}_lr_{}_gradient_clip_{}_precision_{}.pth'.format(args.model_path, 0, args.dataset, args.base_batch_size, args.base_epochs, args.base_monitor, args.base_training_repeat, args.base_retraining_patience, args.early_stop, args.popularity, base_cne_config.interaction.parameters.weight_decay, args.base_patience, args.num_ng, base_cne_config.interaction.parameters.dim, base_cne_config.interaction.parameters.learning_rate, args.gradient_clipping, args.precision)
	if args.use_ot:
		model_file_path = '{}cne_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_dim_{}_lr_{}_gradient_clip_{}_precision_{}.pth'.format(args.model_path, args.use_ot, args.use_pretrained, args.dataset, args.batch_size, args.epochs, args.monitor, args.training_repeat, args.lambda_p, args.sinkhorn_gamma, args.sinkhorn_maxiter, args.retraining_patience, args.early_stop, args.popularity, cne_config.interaction.parameters.weight_decay, args.patience, args.num_ng, cne_config.interaction.parameters.dim, cne_config.interaction.parameters.learning_rate, args.gradient_clipping, args.precision)
	else:
		model_file_path = '{}cne_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_dim_{}_lr_{}_gradient_clip_{}_precision_{}.pth'.format(args.model_path, 0, args.dataset, args.batch_size, args.epochs, args.monitor, args.training_repeat, args.retraining_patience, args.early_stop, args.popularity, cne_config.interaction.parameters.weight_decay, args.patience, args.num_ng, cne_config.interaction.parameters.dim, cne_config.interaction.parameters.learning_rate, args.gradient_clipping, args.precision)


	if args.popularity:
		popularity_model = IdentityModel(user_num ,item_num, None, cne_config.popularity.parameters)
		if torch.cuda.is_available():
			popularity_model = popularity_model.cuda()

		########################### CREATE MODEL POPULARITY #################################
		if not os.path.exists(popularity_model_file_path):
			print("popularity model not found")
			popularity_model = train(args, config, popularity_model, IdentityModel, model_name='popularity', model_file_path=model_file_path, epochs=popularity_epochs, patience=popularity_patience, monitor=args.base_monitor)
			if args.out:
				if not os.path.exists(args.model_path):
					os.mkdir(args.model_path)
				popularity_model.zero_grad()
				torch.save(popularity_model.state_dict(), 
					popularity_model_file_path)

		else:
			print("popularity model found")
			state = torch.load(popularity_model_file_path)
			popularity_model.load_state_dict(state)



	class_ = IdentityModel
	if args.popularity:
		if torch.cuda.is_available():
			popularity_model.cuda()
		d = {'embeddings': {'emb_u': popularity_model.emb_u.weight.detach(), 'emb_i': popularity_model.emb_i.weight.detach()}, 
		'emb_operator': popularity_model._emb_operator}
		dependent_models = {'popularity': d}
	else:
		dependent_models = dict()
	
	cne_model = class_(user_num ,item_num, dependent_models, cne_config.interaction.parameters, ot_param_dict=dict(), use_ot=args.use_ot)
	if torch.cuda.is_available():
		cne_model.cuda()

	user2, item2, ot_param_dict, ot_plugin = get_ot_param_dict(args, train_loader, user_num, item_num)

	if not os.path.exists(model_file_path) or args.train_if_found:
		print("model not found")



		########################### CREATE MODEL CNE #################################
		start_time = time.time()
		if args.use_ot == 1:
			if args.use_pretrained:	
				# raise("errrrrr pretraining")
				state = torch.load(base_model_file_path)

				cne_model.load_state_dict(state)
				cne_model.update_cne_parameters(cne_config.interaction.parameters)


				cne_model.init_part2(args.use_ot, ot_param_dict, dependent_models, user_num, item_num)
			else:
				cne_model = class_(user_num ,item_num, dependent_models, cne_config.interaction.parameters, use_ot=args.use_ot, ot_param_dict=ot_param_dict, )
		else:
			cne_model = class_(user_num ,item_num, dependent_models, cne_config.interaction.parameters, use_ot=args.use_ot, ot_param_dict=ot_param_dict)
		if os.path.exists(model_file_path):
			try:
				state = torch.load(model_file_path)
				cne_model.load_state_dict(state)
			except Exception as e:
				cne_model = class_.load_from_checkpoint(model_file_path)
			cne_model.init_part2(args.use_ot, ot_param_dict, dependent_models, user_num, item_num)

		if torch.cuda.is_available():
			cne_model = cne_model.cuda()


		########################### TRAINING CNE #####################################
		cne_model = train(args, config, cne_model, class_, model_name='cne', ot_param_dict=ot_param_dict, dependent_models=dependent_models, model_file_path=model_file_path, epochs=args.epochs, patience=args.patience)
		if torch.cuda.is_available():
			cne_model = cne_model.cuda()
		end_time = time.time()
		total_time = end_time - start_time
		print("The total time is: " + 
			time.strftime("%H: %M: %S", time.gmtime(total_time)))

		if args.out:
			if not os.path.exists(args.model_path):
				os.mkdir(args.model_path)
			cne_model.zero_grad()
			torch.save(cne_model.state_dict(), 
				model_file_path)


	else:
		print("model found")
		try:
			state = torch.load(model_file_path)
			cne_model.load_state_dict(state)
		except Exception as e:
			print(e)
			cne_model = class_.load_from_checkpoint(model_file_path)

		cne_model.init_part2(args.use_ot, ot_param_dict, dependent_models, user_num, item_num)
		if torch.cuda.is_available():
			cne_model.cuda()


	cne_model.eval()
	if torch.cuda.is_available():
		cne_model = cne_model.cuda()
	cne_model.eval()

	evaluate.evaluate_print(cne_model, train_loader, test_loader_classification, test_loader_ranking, top_k_list_for_rec_performance, user_num, item_num, validation_dataset=val_loader.dataset)

	cne_model.eval()
	if torch.cuda.is_available():
		cne_model = cne_model.cuda()
	cne_model.eval()


	prediction2 = None
	if args.use_ot and args.ot_method == 'all' or (not args.use_ot and user_num+item_num < 9000):
		if torch.cuda.is_available():
			user2 = user2.cuda()
			item2 = item2.cuda()
		prediction2 = cne_model(user2, item2)
	ot_eval = OTEvaluation(user_num, item_num)

	for top_k in top_k_list:
		if top_k > user_num/2 or top_k > item_num/2:
			continue
		print("top_k: %d" % top_k)
		ot_eval.compute_metrics(cne_model, prediction2, top_k, compute_imbalance=True, lorenz_file_path_prefix=None)

