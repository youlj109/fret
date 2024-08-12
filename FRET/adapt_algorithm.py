import numpy as np
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"


class ERM(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model.eval()
        self.featurizer = model.featurizer
        self.classifier = model.classifier
        
    @torch.no_grad()
    def forward(self, x):
        z = self.featurizer(x)
        p = self.classifier(z)
        return p  
    
        #outputs = self.model.predict(x)
        #return outputs



class BN(nn.Module):
    def __init__(self, model, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.steps = steps
        self.episodic = episodic
        assert self.steps>=0, 'steps must be non-negative'
        if self.steps==0:
            self.model.eval()
    
    @torch.no_grad()
    def forward(self, x):
        if self.steps>0:
            for _ in range(self.steps):
                outputs = self.model.predict(x)
        else:
            outputs = self.model.predict(x)
        return outputs


class Tent(nn.Module):
    """
    ICLR,2021
    Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    
    @torch.enable_grad()
    def forward_and_adapt(self,x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        outputs = model.predict(x)
        # adapt
        loss = softmax_entropy(outputs).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
        return outputs


class PseudoLabel(nn.Module):
    def __init__(self, model, optimizer, beta=0.9,steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.beta = beta  #threshold for selecting pseudo labels
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    
    @torch.enable_grad()
    def forward_and_adapt(self,x, model, optimizer):
        # forward
        outputs = model.predict(x)
        # adapt        
        scores = F.softmax(outputs,1)
        py,y_prime = torch.max(scores,1)
        mask = py > self.beta
        loss = F.cross_entropy(outputs[mask],y_prime[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return outputs

        
class T3A(nn.Module):
    """
    T3A, NeurIPS 2021
    """
    def __init__(self,model,filter_K=100,steps=1,episodic=False):
        super().__init__()
        self.model = model.eval()
        self.featurizer = model.featurizer
        self.classifier = model.classifier
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.filter_K = filter_K
        
        if hasattr(self.classifier, 'fc') and hasattr(self.classifier.fc, 'weight'):
            warmup_supports = self.classifier.fc.weight.data
        else:
            warmup_supports = self.classifier.weight.data
        self.num_classes = warmup_supports.size()[0]
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float()
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
             
    @torch.no_grad() 
    def forward(self,x):
        z = self.featurizer(x)
        p = self.classifier(z)
        yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)

        self.supports = self.supports.to(z.device)
        self.labels = self.labels.to(z.device)
        self.ent = self.ent.to(z.device)
        self.supports = torch.cat([self.supports,z])
        self.labels = torch.cat([self.labels,yhat])
        self.ent = torch.cat([self.ent,ent])
        
        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent # 置信度
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels
        

class TSD(nn.Module):
    """
    Test-time Self-Distillation (TSD)
    CVPR 2023
    """
    def __init__(self,model,optimizer,lam=0,filter_K=100,steps=1,episodic=False):
        super().__init__()
        self.model = model
        self.featurizer = model.featurizer
        self.classifier = model.classifier
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.filter_K = filter_K
        
        if hasattr(self.classifier, 'fc') and hasattr(self.classifier.fc, 'weight'):
            warmup_supports = self.classifier.fc.weight.data.detach()
        else:
            warmup_supports = self.classifier.weight.data.detach()
        self.num_classes = warmup_supports.size()[0]
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float()
        self.warmup_scores = F.softmax(warmup_prob,1)
                
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
        self.scores = self.warmup_scores.data
        self.lam = lam
        
        
    def forward(self,x):
        z = self.featurizer(x)
        p = self.classifier(z)
                       
        yhat = F.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)
        scores = F.softmax(p,1)

        with torch.no_grad():
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.scores = self.scores.to(z.device)
            self.supports = torch.cat([self.supports,z])
            self.labels = torch.cat([self.labels,yhat])
            self.ent = torch.cat([self.ent,ent])
            self.scores = torch.cat([self.scores,scores])
        
            supports, labels = self.select_supports()
            supports = F.normalize(supports, dim=1)
            weights = (supports.T @ (labels))
                
        dist,loss = self.prototype_loss(z,weights.T,scores,use_hard=False)

        loss_local = topk_cluster(z.detach().clone(),supports,self.scores,p,k=3)
        loss += self.lam*loss_local
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return p

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        self.scores = self.scores[indices]
        
        return self.supports, self.labels
    
    def prototype_loss(self,z,p,labels=None,use_hard=False,tau=1):
        #z [batch_size,feature_dim]
        #p [num_class,feature_dim]
        #labels [batch_size,]        
        z = F.normalize(z,1)
        p = F.normalize(p,1)
        dist = z @ p.T / tau
        if labels is None:
            _,labels = dist.max(1)
        if use_hard:
            """use hard label for supervision """
            #_,labels = dist.max(1)  #for prototype-based pseudo-label
            labels = labels.argmax(1)  #for logits-based pseudo-label
            loss =  F.cross_entropy(dist,labels)
        else:
            """use soft label for supervision """
            loss = softmax_kl_loss(labels.detach(),dist).sum(1).mean(0)  #detach is **necessary**
            #loss = softmax_kl_loss(dist,labels.detach()).sum(1).mean(0) achieves comparable results
        return dist,loss
        

def topk_labels(feature,supports,scores,k=3):
    feature = F.normalize(feature,1)
    supports = F.normalize(supports,1)
    sim_matrix = feature @ supports.T  #B,M
    _,idx_near = torch.topk(sim_matrix,k,dim=1)  #batch x K
    scores_near = scores[idx_near]  #batch x K x num_class
    soft_labels = torch.mean(scores_near,1)  #batch x num_class
    soft_labels = torch.argmax(soft_labels,1)
    return soft_labels
    

def topk_cluster(feature,supports,scores,p,k=3):
    #p: outputs of model batch x num_class
    feature = F.normalize(feature,1)
    supports = F.normalize(supports,1)
    sim_matrix = feature @ supports.T  #B,M
    topk_sim_matrix,idx_near = torch.topk(sim_matrix,k,dim=1)  #batch x K
    scores_near = scores[idx_near].detach().clone()  #batch x K x num_class
    diff_scores = torch.sum((p.unsqueeze(1) - scores_near)**2,-1)
    
    loss = -1.0* topk_sim_matrix * diff_scores
    return loss.mean()
    
    
def knn_affinity(X,knn):
    #x [N,D]
    N = X.size(0)
    X = F.normalize(X,1)
    dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
    n_neighbors = min(knn + 1, N)
    knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]
    W = torch.zeros(N, N, device=X.device)
    W.scatter_(dim=-1, index=knn_index, value=1.0)
    return W
    
       
def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

    
def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    return kl_div        
        

def get_distances(X, Y, dist_type="cosine"):
    """
    Args:
        X: (N, D) tensor
        Y: (M, D) tensor
    """
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances


@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank):
    pred_probs = []
    K = 4
    for feats in features.split(64):
        distances = get_distances(feats, features_bank,"cosine")
        _, idxs = distances.sort()
        idxs = idxs[:, : K]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs


class FRET(nn.Module):
    """
    Feature Redundancy Elimination for Test Time Adaptation
    """
    def __init__(self, model, optimizer, lam=[1,1,1], filter_K=100, k = 1, label_consistance=True):
        super().__init__()
        self.model = model
        self.featurizer = model.featurizer
        self.classifier = model.classifier
        self.optimizer = optimizer
        self.filter_K = filter_K
        self.k = k
        self.label_consistance = label_consistance

        if hasattr(self.classifier, 'fc') and hasattr(self.classifier.fc, 'weight'):
            warmup_supports = self.classifier.fc.weight.data.detach()
        else:
            warmup_supports = self.classifier.weight.data.detach()
        

        self.num_classes = warmup_supports.size()[0]
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)    
        
        self.warmup_ent = softmax_entropy(warmup_prob)  
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float() 
        self.warmup_scores = F.softmax(warmup_prob,1)  
                
        self.supports = self.warmup_supports.data
        self.ent = self.warmup_ent.data
        self.output_labels = self.warmup_labels.data
        self.output_scores = self.warmup_scores.data
        self.lam = lam


    def forward(self,x):
        Device_ = x.device
        x_augment = self.augment_image(x)
        feature_embeddings = self.featurizer(x)
        faeture_embeddings_augment = self.featurizer(x_augment)
        feature_embeddings_positive, feature_embeddings_negative = self.Compute_presentation(feature_embeddings)

        p_positive = self.classifier(feature_embeddings_positive)
        p_negative = self.classifier(feature_embeddings_negative)
        p = self.classifier(feature_embeddings)    
            
        yhat = F.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)
        scores = F.softmax(p,1)

        with torch.no_grad():
            self.supports = self.supports.to(Device_) 
            self.ent = self.ent.to(Device_)
            self.output_labels = self.output_labels.to(Device_)       
            self.output_scores = self.output_scores.to(Device_)       
            
            self.supports = torch.cat([self.supports,feature_embeddings])
            self.output_labels = torch.cat([self.output_labels,yhat])
            self.ent = torch.cat([self.ent,ent])
            self.output_scores = torch.cat([self.output_scores,scores])

            k = self.k
            percentile_k_entropy = torch.quantile(softmax_entropy(scores), k)

            selected_indices = (softmax_entropy(scores) <= percentile_k_entropy).nonzero(as_tuple=True)[0]
            suports_center = self.Compute_suports_center()
            
        Pseudo_label = self.Compute_pseudo_label(feature_embeddings, suports_center.T)


        loss_pseudo_label_positive = self.pseudo_label_positive_loss(Pseudo_label = Pseudo_label[selected_indices], p_positive = p_positive[selected_indices], label_consistance=self.label_consistance)
        loss_embeddings = self.feature_embedding_loss(feature_embeddings, faeture_embeddings_augment, feature_embeddings_negative, tau=1)
        loss_pseudo_label_nagative = self.pseudo_label_negative_loss(Pseudo_label = Pseudo_label[selected_indices],p_positive = p_positive[selected_indices],p_negative = p_negative[selected_indices], label_consistance=self.label_consistance)
        loss_minimize_ent = self.ent_loss(Pseudo_label = Pseudo_label[selected_indices],p_positive = p_positive[selected_indices],scores = scores[selected_indices], label_consistance=self.label_consistance)
        
        loss = loss_pseudo_label_positive + self.lam[0]*loss_embeddings + self.lam[1]*loss_minimize_ent+  self.lam[2]*loss_pseudo_label_nagative

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return p
    

    
    def augment_image(self, image):
        transform = transforms.GaussianBlur(kernel_size=(3, 3))
        augmented_image = transform(image)
        return augmented_image
    
    
    def Compute_presentation(self, X):
        X_norm = F.normalize(X, dim=0)

        d = X_norm.shape[1]

        M_F = torch.matmul(X_norm.T, X_norm)

        M_m = torch.eye(d, device=X.device)

        M_I = M_m 
        
        M_O = M_F - M_m

        D_I = torch.diag(torch.sum(torch.abs(M_I), axis=1))
        D_O = torch.diag(torch.sum(torch.abs(M_O), axis=1))

        D_I_inv_sqrt = torch.inverse(torch.sqrt(D_I))
        result_I = X @ D_I_inv_sqrt @ M_I @ D_I_inv_sqrt

        D_O_inv_sqrt = torch.inverse(torch.sqrt(D_O))
        result_O = X @ D_O_inv_sqrt @ M_O @ D_O_inv_sqrt

        return result_I, result_O

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.output_labels.argmax(dim=1).long() 
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
        
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])

        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.output_labels = self.output_labels[indices]
        self.ent = self.ent[indices]
        self.output_scores = self.output_scores[indices]
            
        return self.supports, self.output_labels
    
    def Compute_suports_center(self):
        supports, labels = self.select_supports() 
        supports = F.normalize(supports, dim=1)
        supports_center = (supports.T @ (labels)) 
        return supports_center

    def Compute_pseudo_label(self, X, supports_center):
        X = F.normalize(X,1)
        supports_center = F.normalize(supports_center,1)
        Pseudo_label = X @ supports_center.T 
        return Pseudo_label

    def feature_embedding_loss(self, X, X_positive, X_negative, tau):
        X = F.normalize(X,1)
        X_positive = F.normalize(X_positive,1)
        X_negative = F.normalize(X_negative,1)
        numerator_1 = torch.exp(F.cosine_similarity(X, X_positive) / tau)
        fenmu_1 = torch.exp(F.cosine_similarity(X, X_negative) / tau) + numerator_1
        loss = (-torch.log(numerator_1 / fenmu_1)).mean()
        return loss

    def pseudo_label_positive_loss(self, Pseudo_label, p_positive, label_consistance=True):
        scores = F.softmax(p_positive, 1)

        if label_consistance==True:
            indices = Pseudo_label.argmax(1) == scores.argmax(1)
        else:
            indices = torch.LongTensor(list(range(len(Pseudo_label))))        
        loss = self.softmax_kl_loss(scores[indices].detach(),Pseudo_label[indices]).sum(1).mean(0)
        return loss
    
    def softmax_kl_loss(self, input_logits, target_logits):
        """Takes softmax on both sides and returns KL divergence

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
        if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

        kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
        return kl_div  

    def pseudo_label_negative_loss(self, Pseudo_label, p_positive, p_negative, label_consistance=True):
        if label_consistance==True:
            indices = (Pseudo_label.argmax(1) == p_positive.argmax(1))
        else:
            indices = torch.LongTensor(list(range(len(Pseudo_label)))) 
        
        label_p_negative = F.softmax(p_negative,1)
        loss = softmax_kl_loss(label_p_negative[indices].detach(),1-Pseudo_label[indices]).sum(1).mean(0)  

        
        return loss
    
    def ent_loss(self, Pseudo_label, p_positive, scores,label_consistance=True):
        if label_consistance==True:
            indices = (Pseudo_label.argmax(1) == p_positive.argmax(1))
        else:
            indices = torch.LongTensor(list(range(len(Pseudo_label)))) 

        loss = softmax_entropy(scores[indices]).mean()        
        return loss



class EnergyModel(nn.Module):
    """
    2024CVPR TEA: Test-time Energy Adaptation
    """
    def __init__(self, model):
        super(EnergyModel, self).__init__()
        self.f = model

    def classify(self, x):
        penult_z = self.f.predict(x)
        return penult_z
    
    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1), logits
        else:
            return torch.gather(logits, 1, y[:, None]), logits
        
def init_random(bs, im_sz=32, n_ch=3):
    return torch.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)

def sample_p_0(reinit_freq, replay_buffer, bs, im_sz, n_ch, device, y=None):
    if len(replay_buffer) == 0:
        return init_random(bs, im_sz=im_sz, n_ch=n_ch), []
    buffer_size = len(replay_buffer)
    inds = torch.randint(0, buffer_size, (bs,))
    # if cond, convert inds to class conditional inds

    buffer_samples = replay_buffer[inds]
    random_samples = init_random(bs, im_sz=im_sz, n_ch=n_ch)
    choose_random = (torch.rand(bs) < reinit_freq).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples.to(device), inds

def sample_q(f, replay_buffer, n_steps, sgld_lr, sgld_std, reinit_freq, batch_size, im_sz, n_ch, device, y=None):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    f.eval()
    # get batch size
    bs = batch_size if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(reinit_freq=reinit_freq, replay_buffer=replay_buffer, bs=bs, im_sz=im_sz, n_ch=n_ch, device=device ,y=y)
    init_samples = deepcopy(init_sample)
    x_k = torch.autograd.Variable(init_sample, requires_grad=True)
    # sgld
    for k in range(n_steps):
        f_prime = torch.autograd.grad(f(x_k, y=y)[0].sum(), [x_k], retain_graph=True)[0]
        x_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)
    f.train()
    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples, init_samples.detach()

class Energy(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, 
                 buffer_size=10000, sgld_steps=20, sgld_lr=1, sgld_std=0.01, reinit_freq=0.05, if_cond=False, 
                 n_classes=10, im_sz=32, n_ch=3): 
        super().__init__()

        self.energy_model=EnergyModel(model)
        self.replay_buffer = init_random(buffer_size, im_sz=im_sz, n_ch=n_ch)
        self.replay_buffer_old = deepcopy(self.replay_buffer)
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.sgld_steps = sgld_steps
        self.sgld_lr = sgld_lr
        self.sgld_std = sgld_std
        self.reinit_freq = reinit_freq
        self.if_cond = if_cond
        
        self.n_classes = n_classes
        self.im_sz = im_sz
        self.n_ch = n_ch
        
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.energy_model, self.optimizer)
        
    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.energy_model, self.optimizer, 
                                        self.replay_buffer, self.sgld_steps, self.sgld_lr, self.sgld_std, self.reinit_freq,
                                        if_cond=self.if_cond, n_classes=self.n_classes)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.energy_model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        
@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, energy_model, optimizer, replay_buffer, sgld_steps, sgld_lr, sgld_std, reinit_freq, if_cond=False, n_classes=10):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    batch_size=x.shape[0]
    n_ch = x.shape[1]
    im_sz = x.shape[2]
    device = x.device
    
    if if_cond == 'uncond':
        x_fake, _ = sample_q(energy_model, replay_buffer, 
                             n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq, 
                             batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=None)
    elif if_cond == 'cond':
        y = torch.randint(0, n_classes, (batch_size,)).to(device)
        x_fake, _ = sample_q(energy_model, replay_buffer, 
                             n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq, 
                             batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=y)

    # forward
    out_real = energy_model(x)
    energy_real = out_real[0].mean()
    energy_fake = energy_model(x_fake)[0].mean()

    # adapt
    loss = (- (energy_real - energy_fake)) 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    outputs = energy_model.classify(x)

    return outputs

def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data



class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class SAR(nn.Module):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
        2023 ICLR "TOWARDS STABLE TEST-TIME ADAPTATION IN DYNAMIC WILD WORLD"
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, margin_e0=0.4*math.log(1000), reset_constant_em=0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "SAR requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.margin_e0 = margin_e0  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
    def forward(self, x ):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, ema, reset_flag = forward_and_adapt_sar(x, self.model, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema)
            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss
        
        return outputs
    
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)
        self.ema = None

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_sar(x, model, optimizer, margin,reset_constant,ema):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    optimizer.zero_grad()
    # forward
    outputs = model.predict(x)
    # adapt
    # filtering reliable samples/gradients for further adaptation; first time forward
    entropys = softmax_entropy(outputs)
    filter_ids_1 = torch.where(entropys < margin)
    entropys = entropys[filter_ids_1]
    loss = entropys.mean(0)
    loss.backward()

    optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
    entropys2 = softmax_entropy(model.predict(x))
    entropys2 = entropys2[filter_ids_1]  # second time forward  
    filter_ids_2 = torch.where(entropys2 < margin)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
    loss_second = entropys2[filter_ids_2].mean(0)
    if not np.isnan(loss_second.item()):
        ema = update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

    # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
    loss_second.backward()
    optimizer.second_step(zero_grad=True)

    # perform model recovery
    reset_flag = False
    if ema is not None:
        if ema < 0.2:
            print("ema < 0.2, now reset the model")
            reset_flag = True

    return outputs, ema, reset_flag


class EATA(nn.Module):   
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    ICML 2022  'Efficient Test-Time Model Adaptation without Forgetting'
    """
    def __init__(self, model, optimizer, steps=1, fishers=None, fisher_alpha=2000.0, episodic=False, e_margin=math.log(1000)/2-1, d_margin=0.05):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "EATA requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = e_margin # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = d_margin # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)

        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)

        self.fishers = fishers # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        self.fisher_alpha = fisher_alpha # trade-off \beta for two losses (Eqn. 8) 

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, num_counts_2, num_counts_1, updated_probs = forward_and_adapt_eata(x, self.model, self.optimizer, self.fishers, self.e_margin, self.current_model_probs, fisher_alpha=self.fisher_alpha, num_samples_update=self.num_samples_update_2, d_margin=self.d_margin)
            self.num_samples_update_2 += num_counts_2
            self.num_samples_update_1 += num_counts_1
            self.reset_model_probs(updated_probs)
        
        return outputs
    
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_eata(x, model, optimizer, fishers, e_margin, current_model_probs, fisher_alpha=50.0, d_margin=0.05, scale_factor=2, num_samples_update=0):
    # forward
    outputs = model.predict(x)
    # adapt
    entropys = softmax_entropy(outputs)
    # filter unreliable samples
    filter_ids_1 = torch.where(entropys < e_margin)
    ids1 = filter_ids_1
    ids2 = torch.where(ids1[0]>-0.1)
    entropys = entropys[filter_ids_1] 
    # filter redundant samples
    if current_model_probs is not None: 
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
        entropys = entropys[filter_ids_2]
        ids2 = filter_ids_2
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
    else:
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
    coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
    # implementation version 1, compute loss, all samples backward (some unselected are masked)
    entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
    loss = entropys.mean(0)

    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
        loss += ewc_loss
    if x[ids1][ids2].size(0) != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs

def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


scaler = GradScaler()
scaler_adv = GradScaler()

class TIPI(nn.Module):
    '''
        CVPR2023 'Test Time Adaptation with Transformation Invariance'
    '''
    def __init__(self, model, lr_per_sample=0.00025/64, optim='SGD', epsilon=2/255, random_init_adv=False, reverse_kl=True,  tent_coeff=0.0, use_test_bn_with_large_batches=False):
        super(TIPI, self).__init__()

        self.lr_per_sample = lr_per_sample
        self.epsilon = epsilon
        self.random_init_adv = random_init_adv
        self.reverse_kl = reverse_kl
        self.tent_coeff = tent_coeff
        self.use_test_bn_with_large_batches = use_test_bn_with_large_batches
        self.large_batch_threshold = 64

        configure_multiple_BN(model,["main","adv"]) 
        self.model = model
        params, _ = collect_params_TIPI(self.model)

        if optim == 'SGD':
            self.optimizer = torch.optim.SGD(params, lr=lr_per_sample,
                                        momentum=0.9,
                                        weight_decay=0.0)
        elif optim == 'Adam':
            self.optimizer = torch.optim.Adam(params, lr=lr_per_sample,
                                        betas=(0.9, 0.999),
                                        weight_decay=0.0)
        else:
            raise NotImplementedError

    def forward(self, x):
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr_per_sample * x.shape[0]
       
        with autocast():
            self.model.train()
            use_BN_layer(self.model,'main')

            delta = torch.zeros_like(x)
            delta.requires_grad_()
            pred = self.model.predict(x+delta)

            use_BN_layer(self.model,'adv')
            if self.random_init_adv:
                delta = (torch.rand_like(x)*2-1) * self.epsilon
                delta.requires_grad_()
                pred_adv = self.model(x+delta)
            else:
                pred_adv = pred

            loss = KL(pred.detach(), pred_adv, reverse=self.reverse_kl).mean()
            grad = torch.autograd.grad(scaler_adv.scale(loss), [delta], retain_graph=(self.tent_coeff!=0.0) and (not self.random_init_adv))[0]
            delta = delta.detach() + self.epsilon*torch.sign(grad.detach())
            delta = torch.clip(delta,-self.epsilon,self.epsilon)
            x_adv = x + delta
            #x_adv = torch.clip(x_adv,0.0,1.0)

            pred_adv = self.model.predict(x_adv)
            loss = KL(pred.detach(), pred_adv, reverse=self.reverse_kl)
            ent = - (pred.softmax(1) * pred.log_softmax(1)).sum(1)
            
            if self.tent_coeff != 0.0:
                loss = loss + self.tent_coeff*ent
            
            loss = loss.mean()
            
            self.optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            #scaler_adv.update()
            #self.optimizer.step()

            use_BN_layer(self.model,'main')
            with torch.no_grad():
                if self.use_test_bn_with_large_batches and x.shape[0] > self.large_batch_threshold:
                    pred = self.model.predict(x)
                else:
                    self.model.eval()
                    pred = self.model.predict(x)

        return pred
    
class MultiBatchNorm2d(nn.Module):
    def __init__(self, bn, BN_layers=['main']):
        super(MultiBatchNorm2d, self).__init__()
        self.weight = bn.weight
        self.bias = bn.bias
        self.BNs = nn.ModuleDict()
        self.current_layer = 'main'
        for l in BN_layers:
            m = deepcopy(bn)
            m.weight = self.weight
            m.bias = self.bias
            try:
                self.BNs[l] = m
            except Exception:
                import pdb; pdb.set_trace()
    def forward(self,x):
        assert self.current_layer in self.BNs.keys()
        return self.BNs[self.current_layer](x)
    
def configure_multiple_BN(net, BN_layers=['main']):
    for attr_str in dir(net):
        m = getattr(net, attr_str)
        if type(m) == nn.BatchNorm2d:
            new_bn = MultiBatchNorm2d(m, BN_layers)
            setattr(net, attr_str, new_bn)
    for n, ch in net.named_children():
        if type(ch) != MultiBatchNorm2d:
            configure_multiple_BN(ch, BN_layers)

def collect_params_TIPI(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, MultiBatchNorm2d)\
                or isinstance(m,nn.GroupNorm)\
                or isinstance(m,nn.InstanceNorm2d)\
                or isinstance(m,nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def use_BN_layer(net, BN_layer='main'):
    for m in net.modules():
        if isinstance(m, MultiBatchNorm2d):
            m.current_layer = BN_layer


def KL(logit1,logit2,reverse=False):
    if reverse:
        logit1, logit2 = logit2, logit1
    p1 = logit1.softmax(1)
    logp1 = logit1.log_softmax(1)
    logp2 = logit2.log_softmax(1) 
    return (p1*(logp1-logp2)).sum(1)