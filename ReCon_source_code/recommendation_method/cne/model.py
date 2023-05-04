from multiprocessing.sharedctypes import Value
import numpy as np
import torch
import pytorch_lightning as pl
from torch import sigmoid
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits
from sklearn.metrics import roc_auc_score
from ot_local.ot_evaluation import OTEvaluation



class Model(pl.LightningModule):
	def __init__(self, params):
		super().__init__()        
		self.update_cne_parameters(params)

	
	def update_cne_parameters(self, params):
		self._learning_rate, self._weight_decay = params["learning_rate"], params["weight_decay"]        
		self._optimizer = getattr(torch.optim, params["optimizer"])
		self._param_init_std = params["param_init_std"]
		self._emb_operator = getattr(torch, params["embedding_operator"])
		self.scheduler = params.get('scheduler')

		

	def forward(self):
		pass


	def training_step(self, batch, batch_idx):
		uids, iids, target = batch
		pred = self(uids, iids)        
		loss = binary_cross_entropy(pred.view(-1, 1).float(), target.view(-1, 1).float()).sum()                
		self.log("train_loss", loss, on_epoch=True)
		return loss


	def validation_step(self, batch, batch_idx):
		uids, iids, target = batch
		pred = self(uids, iids)
		auc = roc_auc_score(target.cpu(), pred.cpu())
		loss = binary_cross_entropy(pred.view(-1, 1).float(), target.view(-1, 1).float()).sum()                
		self.log("valid_auc", auc, prog_bar=True)
		self.log("valid_loss", loss, prog_bar=True)


	def test_step(self, batch, batch_idx, dataloader_idx):
		uids, iids, target = batch
		pred = self(uids, iids)


	def configure_optimizers(self):
		print(f"Config optimizer with learning rate {self._learning_rate}")
		optimizer = self._optimizer(self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)        
		return optimizer
		

class IdentityModel(Model):
	def __init__(self, user_num, item_num, dependent_models, params, use_ot=False, ot_param_dict=None):
		super().__init__(params)
		self.save_hyperparameters()

		self.init_part2(use_ot, ot_param_dict, dependent_models, user_num, item_num)

		self.emb_u = torch.nn.Embedding(self.n_user, params["dim"], dtype=torch.float32)
		self.emb_u.weight.data.normal_(std=self._param_init_std)
		
		self.emb_i = torch.nn.Embedding(self.n_item, params["dim"], dtype=torch.float32)
		self.emb_i.weight.data.normal_(std=self._param_init_std)        
		

			

		self.b = torch.nn.parameter.Parameter(torch.DoubleTensor(1))
		torch.nn.init.normal_(self.b, std=self._param_init_std)
		torch.nn.init.constant_(self.b, -4.0)

	def init_part2(self, use_ot, ot_param_dict, dependent_models, user_num, item_num):
		self.ot_eval = OTEvaluation(user_num, item_num)
		self.use_ot = use_ot
		if ot_param_dict is not None:
			for key, val in ot_param_dict.items():
					setattr(self, key, val)
		
		self.n_user, self.n_item = user_num, item_num
		self.ot_eval = OTEvaluation(user_num, item_num)
		if dependent_models and "popularity" in dependent_models:
    		# self._disable_required_gradient(dependent_models)
			self.lambda_u, self.lambda_i = dependent_models["popularity"]["embeddings"]["emb_u"],  dependent_models["popularity"]["embeddings"]["emb_i"]
			self.lambda_operator = dependent_models["popularity"]["emb_operator"]

	
	def training_step(self, batch, batch_idx):
		uids, iids, target = batch
		pred = self(uids, iids, training=True)
		auc = roc_auc_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy())
		loss = binary_cross_entropy_with_logits(pred.view(-1, 1).float(), target.view(-1, 1).float()).sum()
		self.log("train_first_loss", loss, prog_bar=True)
		if self.use_ot:
			prediction2 = self(self.user2, self.item2)
			sinkhorn_loss = self.ot_plugin.get_sinkhorn_loss(prediction2)
			self.log("train_second_loss", sinkhorn_loss, prog_bar=True)
			loss = loss + self.lambda_p*sinkhorn_loss
		self.log("train_auc", auc, prog_bar=True)
		self.log("train_loss", loss, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		uids, iids, target = batch
		pred = self(uids, iids, training=True)
		auc = roc_auc_score(target.cpu(), pred.cpu())
		loss = binary_cross_entropy_with_logits(pred.view(-1, 1).float(), target.view(-1, 1).float()).sum()                
		if self.use_ot:
			prediction2 = self(self.user2, self.item2)
			sinkhorn_loss = self.ot_plugin.get_sinkhorn_loss(prediction2)
			loss = loss + self.lambda_p*sinkhorn_loss

		self.log("valid_auc", auc, prog_bar=True)
		self.log("valid_loss", loss, prog_bar=True)
		return {'valid_loss': loss, 'valid_auc': auc}
	
	
	def validation_epoch_end(self, outputs):
		if self.use_ot:
			auc_agg = np.mean([o.get('valid_auc') for o in outputs])
			prediction2 = self(self.user2, self.item2)
			congestion, _, _ = self.ot_eval.compute_congestion_coverage_items(prediction2.reshape((self.n_user, self.n_item)), 1, None, do_print=False)
			auc_congestion = auc_agg + self.lambda_p* (-congestion)
			self.log("valid_minus_congestion", -congestion, prog_bar=True)
			self.log("valid_auc_congestion", auc_congestion, prog_bar=True)
			
		else:
			return super().validation_epoch_end(outputs)
	

	def forward(self, uids, iids, training=False):
		emb_sum = self._emb_operator(self.emb_u(uids), self.emb_i(iids)).sum(1)
		if hasattr(self, "lambda_u"):
			emb_sum += self.lambda_operator(self.lambda_u[uids], self.lambda_i[iids]).sum(1)
		sums = emb_sum + self.b
		if training:
			return sums
		else:
			return sigmoid(sums)
