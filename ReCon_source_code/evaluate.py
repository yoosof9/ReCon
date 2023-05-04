import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ndcg_score
from sklearn.metrics._ranking import type_of_target




def get_hit(pos_items, pred_items):
	for i in pos_items:
		if i in pred_items:
			return 1
	return 0


def get_precision(actual, predicted, k):
	act_set = set(actual)
	pred_set = set(predicted[:k])
	result = len(act_set & pred_set) / float(k)
	return result


def get_recall(actual, predicted, k):
	act_set = set(actual)
	pred_set = set(predicted[:k])
	result = len(act_set & pred_set) / float(len(act_set))
	return result


def compute_all_metrics(model, dataloader_classification, dataset_ranking, top_k, is_training=False, eval_ng_num_classification=1, eval_ng_num_ranking=1000, full_negatives=True, train_val_datasets=None):
	if type(model) is dict:
		auc = 0
	else:
		if is_training:
			dataloader_classification.dataset.ng_sample(evaluation=True, eval_ng_num=eval_ng_num_classification)
		true_values = []
		pred_values = []
		for user, item, label in dataloader_classification:
			if torch.cuda.is_available():
				user = user.cuda()
				item = item.cuda()
			predictions = model(user, item)
			true_values = true_values + label.cpu().detach().numpy().tolist()
			pred_values = pred_values + predictions.cpu().detach().numpy().tolist()
		
		auc = roc_auc_score(true_values, pred_values)
	HR, precision, recall, ndcg = compute_ranking_metrics(model, dataset_ranking, top_k, is_training, eval_ng_num_ranking, full_negatives, train_val_datasets=train_val_datasets)
	return HR, precision, recall, auc, ndcg




def compute_ranking_metrics(model, dataset_ranking, top_k, is_training=False, eval_ng_num_ranking=1000, full_negatives=True, train_val_datasets=None):
	HR= []
	precision_list, recall_list = [], []
	NDCG = []

	train_val_dataframes = [d.get_dataframe() for d in train_val_datasets]

	if dataset_ranking is not None:
		if is_training:
			dataset_ranking.ng_sample(evaluation=True, eval_ng_num=eval_ng_num_ranking, eval_full_negatives=full_negatives)
			
		df = dataset_ranking.get_dataframe()
		for user, grouped_df in df.groupby('user'):
			items = grouped_df['item']
			exclude_ids = set()
			if full_negatives:
				for ddf in train_val_dataframes:
					exclude_ids = exclude_ids.union(set(ddf[(ddf['user'] == user) & (ddf['label'] == 1)]['item'].tolist()))

			if type(model) is dict:
				recommends = model[user][:top_k]
				if not full_negatives:
					predictions = [1 if i in recommends else 0 for i in items]
					for idx, jj in recommends:
						predictions[jj] = 1/(idx+1)
				else:
					predictions = [1/(recommends.index(i)+1) if i in recommends else 0 for i in range(0, dataset_ranking.num_item) if i not in exclude_ids]
			else:
				
				if not full_negatives:
					items_t = torch.LongTensor(items.tolist())
				else:
					items_t = torch.LongTensor([jj for jj in range(0, dataset_ranking.num_item) if jj not in exclude_ids])

				user_t = torch.full((len(items_t), ), int(user))

				if torch.cuda.is_available():
					user_t = user_t.cuda()
					items_t = items_t.cuda()
				predictions = model(user_t, items_t)
				if torch.cuda.is_available():
					predictions = predictions.cuda()

				if top_k < len(predictions):
					_, indices = torch.topk(predictions, top_k)
					recommends = torch.take(
							items_t, indices).cpu().numpy().tolist()
				else:
					recommends = items_t.tolist()
					random.shuffle(recommends)
				recommends = recommends[:top_k]
				predictions = predictions.cpu().detach().numpy().tolist()
			pos_items = grouped_df[grouped_df['label'] == 1]['item'].unique().tolist()

			if not full_negatives:
				ndcg_y_true = np.asarray([[int(jj in pos_items) for jj in items.tolist()]])
			else:
				ndcg_y_true = np.asarray([[int(jj in pos_items) for jj in range(0, dataset_ranking.num_item) if jj not in exclude_ids]])
			ndcg_y_score = np.asarray([predictions])
			NDCG.append(ndcg_score(ndcg_y_true, ndcg_y_score,
			k=top_k))
			if len(pos_items) > 0:
				HR.append(get_hit(pos_items, recommends))
				precision_list.append(get_precision(pos_items, recommends, top_k))
				recall_list.append(get_recall(pos_items, recommends, top_k))
		
	return np.mean(HR), np.mean(precision_list), np.mean(recall_list), np.mean(NDCG)

		
def evaluate_print(model, train_loader, test_loader_classification, test_loader_ranking, top_k_list_for_rec_performance, user_num, item_num, validation_dataset, full_negatives=True):
	print(user_num, item_num)
	return_dict = dict()
	for loader_text, loader_classification, loader_ranking, is_training, train_val_datasets in [
		("train_loader", train_loader, train_loader if len(train_loader.dataset) < 50000 else None, True, [validation_dataset]), 
		("test_loader", test_loader_classification, test_loader_ranking, False, [train_loader.dataset, validation_dataset])]:
		
		return_dict[loader_text] = dict()
		for top_k in top_k_list_for_rec_performance:
			if top_k > user_num/2 or top_k > item_num/2:
				continue
			HR, precision, recall, auc, ndcg = compute_all_metrics(model, loader_classification, loader_ranking.dataset if loader_ranking is not None else None, top_k, is_training=is_training, eval_ng_num_classification=1, eval_ng_num_ranking=1000, full_negatives=full_negatives, train_val_datasets=train_val_datasets)
			return_dict[loader_text][top_k] = {'HR': HR, 'precision': precision, 'recall': recall, 'auc': auc, 'ndcg': ndcg}
			print("loader: {} top_k: {:d} HR: {:.3f} Recall: {:.3f} Precision: {:.3f} NDCG: {:.3f} Auc: {:.3f}".format(loader_text, top_k, np.mean(HR), np.mean(recall), np.mean(precision), np.mean(ndcg), auc))
	return return_dict
		
