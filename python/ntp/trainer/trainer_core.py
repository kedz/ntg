import os
import math
import sys
import torch

def optimize_criterion(criterion, model, optimizer, training_data, 
                       validation_data=None, max_epochs=10, 
                       save_model=None,
                       show_progress=True):
    '''
    Minimize/maximize model criterion w.r.t. to the training data using the
    supplied optimizer. If validation data is provided, track learning 
    progress on this data set.
    '''

    if save_model is not None:
        model_dir = os.path.dirname(save_model)
        if model_dir != '' and not os.path.exists(model_dir):
            os.makedirs(model_dir)


    result_data = {"training": []}
    if validation_data is not None:
        result_data["validation"] = []
    
    for epoch in range(1, max_epochs + 1):

        if show_progress:
            print("\n === Epoch {} ===".format(epoch))
            print("   * == Training == ")

        train_epoch(criterion, model, optimizer, training_data)
        criterion.checkpoint("training")

        if show_progress:
            print(criterion.report(indent="     "))

        if validation_data is not None:

            if show_progress:
                print("\n  * == Validation ==")

            eval(criterion, model, validation_data)
            criterion.checkpoint("validation")
            
            if show_progress:

                best_epoch, obj = criterion.find_best_checkpoint("validation")
                if best_epoch == epoch and save_model is not None:
                    torch.save(model, save_model)
                print(criterion.report(indent="     "))
                print("\n     Best epoch: {} obj: {}\n".format(
                    best_epoch, obj))


        if show_progress:
            print("")

def train_epoch(criterion, model, optimizer, dataset, step_callback=None):
    max_steps = math.ceil(dataset.size / dataset.batch_size)
    criterion.reset()
    model.train()

    if step_callback is None:
        step_callback = default_step_callback

    for step, batch in enumerate(dataset.iter_batch(), 1):
        batch_loss = criterion.minimize(batch, model, optimizer)
        step_callback(step, max_steps, batch_loss, criterion)

def eval(criterion, model, dataset, step_callback=None):
    max_steps = math.ceil(dataset.size / dataset.batch_size)
    criterion.reset()
    model.eval()

    if step_callback is None:
        step_callback = default_step_callback

    for step, batch in enumerate(dataset.iter_batch(), 1):
        batch_loss = criterion.compute_loss(batch, model)
        step_callback(step, max_steps, batch_loss, criterion)


def default_step_callback(step, max_steps, batch_loss, criterion):
    sys.stdout.write("\r")
    sys.stdout.write(" " * 79)
    sys.stdout.write("\r")
    sys.stdout.write(
        "\t{} / {} | obj: {:0.4f}".format(step, max_steps, criterion.avg_loss))
    sys.stdout.flush()
    if step == max_steps:
        print("\n")

