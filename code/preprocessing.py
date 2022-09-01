import numpy as np
import torch

def evaluate(model, features, labels, verbose=False):
    if len(features) > 100:
        predictions = []
        i = 0
        
        # really simple batching
        batch_size = 50000
        while True:
            inp = features[i:(i+batch_size)]
            if len(inp) == 0:
                break
            predictions.append(model.predict(inp).reshape(-1))
            i += batch_size
            if i > len(features):
                break

        if isinstance(predictions[0], torch.Tensor):
            predictions = torch.cat(predictions)
        else:
            # assume numpy
            predictions = np.concatenate(predictions)

    else:
        predictions = model.predict(features)

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)
    errors = abs(predictions - labels)
    num_stops_wrong = predictions-labels
    num_stops_wrong = len(num_stops_wrong[num_stops_wrong == 1])
    num_gos_wrong = predictions-labels
    num_gos_wrong = len(num_gos_wrong[num_gos_wrong == -1])
    mape = 100 * np.sum(errors)/len(labels)
    accuracy = 100 - mape
    if verbose:
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))
        print('Number Stops Wrong {:0.2f}'.format(num_stops_wrong))
        print('Number Gos Wrong {:0.2f}'.format(num_gos_wrong))
        return accuracy
    else:
        return accuracy, num_stops_wrong, num_gos_wrong
    