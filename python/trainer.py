import sys
from math import ceil
import time
import torch
import torch.nn.functional as F

UPDATE_TEMPLATE = "{} / {} steps ({:0.3f}%) | {:0.3f}ms/stp | eta {:0.3f}min" \
    " | avg. nll {:0.3f}"
FINISH_TEMPLATE = "{} / {} steps ({:0.3f}%) | {:0.3f}ms/stp |" \
    " elapsed {:0.3f}min | avg. nll {:0.3f}\n"
START_TIME = 0


def optimizer_from_args(args, parameters):
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=args.lr)
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(parameters, lr=args.lr)
    elif args.optimizer == "adadelta":
        optimizer = torch.optim.Adadelta(parameters, lr=args.lr)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
    else:
        raise Exception("Unkown optimizer: {}".format(args.optimizer))
    return optimizer


def _update_progress(step, max_steps, avg_nll):

    if step == 1:
        global START_TIME
        START_TIME = time.perf_counter()

    time_per_step_ms = 1000 * (time.perf_counter() - START_TIME) /  step
    eta = (max_steps - step) * time_per_step_ms / 1000 / 60 

    if step < max_steps:
        msg = UPDATE_TEMPLATE.format(
            step, max_steps, 100 * step / max_steps, time_per_step_ms, eta, 
            avg_nll)
        sys.stdout.write("\r")
        sys.stdout.write(msg)
        sys.stdout.flush()
    else:
        elapsed = time_per_step_ms * step / 1000 / 60
        msg = FINISH_TEMPLATE.format(
            step, max_steps, 100 * step / max_steps, time_per_step_ms,
            elapsed, avg_nll)
        sys.stdout.write("\r")
        sys.stdout.write(msg)
        sys.stdout.flush()


def minimize_criterion(crit, data_train, data_valid, max_epochs,
                       save_best_model=None, show_progress=True):

    train_scores = []
    valid_scores = []
    
    best_score = float("inf")
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):

        if show_progress:
            print("\n === Epoch {} ===".format(epoch))

        print("train split ...")
        train_obj = train_epoch_(crit, data_train, show_progress=show_progress)
        print(crit.status_msg())

        print("valid split ...")
        valid_obj = eval_(crit, data_valid, show_progress=show_progress)
        print(crit.status_msg())

        if valid_obj < best_score:
            best_score = valid_obj
            best_epoch = epoch
        
            if save_best_model:
                print("Writing model to {} ...".format(save_best_model))
                torch.save(crit.model, save_best_model)

        train_scores.append(train_obj)
        valid_scores.append(valid_obj)

    return {"valid_nll": valid_scores, "train_nll": train_scores}        

def train_epoch_(crit, dataset, show_progress=True):
    
    max_steps = ceil(dataset.size / dataset.batch_size)
    crit.reset()
    crit.model.train()

    for step, batch in enumerate(dataset.iter_batch(), 1):
        crit.minimize(batch)
        if show_progress:
            _update_progress(step, max_steps, crit.avg_loss)
    return crit.avg_loss
           
def eval_(crit, dataset, show_progress=True):
    
    max_steps = ceil(dataset.size / dataset.batch_size)
    crit.reset()
    crit.model.eval()

    for step, batch in enumerate(dataset.iter_batch(), 1):
        
        crit.compute_loss(batch)
        if show_progress:
            _update_progress(step, max_steps, crit.avg_loss)
 
    return crit.avg_loss

def train(model, data_train, data_valid, optimizer, max_epochs, 
          show_progress=True, best_model_path=None):


    best_validation_score = float("inf")
    best_model = None 

    

    for epoch in range(1, max_epochs + 1):
        if show_progress:
            print("\n === Epoch {} ===".format(epoch))

        train_nll = train_epoch(
            model, data_train, optimizer, show_progress=show_progress)
        if show_progress:
            print(" training avg. nll {}".format(train_nll))

        valid_nll = eval_epoch(
            model, data_valid, show_progress=show_progress)
        if show_progress:
            print(" validation avg. nll {}".format(valid_nll))
        if valid_nll < best_validation_score and best_model_path:
            print("new best model!")
            best_validation_score = valid_nll
            torch.save(model, best_model_path)
            




def train_epoch(model, dataset, optimizer, show_progress=True):

    model.train()
    max_steps = ceil(dataset.size / dataset.batch_size)

    total_examples = 0
    total_nll = 0
    avg_nll = 0

    for step, batch in enumerate(dataset.iter_batch(), 1):

        if show_progress:
            _update_progress(step, max_steps, avg_nll)

        optimizer.zero_grad()

        logits = model(batch)
        logits_flat = logits.view(
            logits.size(0) * logits.size(1), logits.size(2))
        tgt_out = batch.target_out
        tgt_out_flat = tgt_out.t().view(tgt_out.size(0) * tgt_out.size(1))        
        loss = F.cross_entropy(logits_flat, tgt_out_flat, ignore_index=0)
        total_nll += loss.data[0] * batch.target_in.data.gt(0).sum()
        total_examples += batch.target_in.data.gt(0).sum()
        avg_nll = total_nll / total_examples

        loss.backward()            
        optimizer.step()

    return avg_nll


def eval_epoch(model, dataset, show_progress=True):

    model.eval()
    max_steps = ceil(dataset.size / dataset.batch_size)

    total_examples = 0
    total_nll = 0
    avg_nll = 0

    for step, batch in enumerate(dataset.iter_batch(), 1):

        if show_progress:
            _update_progress(step, max_steps, avg_nll)

        logits = model(batch)
        logits_flat = logits.view(
            logits.size(0) * logits.size(1), logits.size(2))
        tgt_out = batch.target_out
        tgt_out_flat = tgt_out.t().view(tgt_out.size(0) * tgt_out.size(1))        
        loss = F.cross_entropy(logits_flat, tgt_out_flat, ignore_index=0)
        total_nll += loss.data[0] * batch.target_in.data.gt(0).sum()
        total_examples += batch.target_in.data.gt(0).sum()
        avg_nll = total_nll / total_examples

    return avg_nll
