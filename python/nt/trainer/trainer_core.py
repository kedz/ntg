import math
import sys

def optimize_criterion(criterion, model, optimizer, training_data, 
                       validation_data=None, max_epochs=10, 
                       show_progress=True):
    '''
    Minimize/maximize model criterion w.r.t. to the training data using the
    supplied optimizer. If validation data is provided, track learning 
    progress on this data set.
    '''

    result_data = {"training": []}
    if validation_data is not None:
        result_data["validation"] = []
    
    for epoch in range(1, max_epochs + 1):

        if show_progress:
            print("\n === Epoch {} ===".format(epoch))
        train_epoch(criterion, model, optimizer, training_data)
        print(criterion.result_dict())

        if show_progress:
            print("")
        if validation_data is not None:
            eval(criterion, model, validation_data)
            print(criterion.result_dict())

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
        "{} / {} | obj: {:0.4f}".format(step, max_steps, criterion.avg_loss))
    sys.stdout.flush()

#def epoch_start_callback(epoch, model, criterion
