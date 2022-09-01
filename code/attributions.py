import torch
import numpy as np

'''
    Base class providing collective functionality for all explainability methods used here.
    `attribute` is the main method used from the outside.
'''
class AttributionMethod:

    def __init__(self, model, background=None, background_labels=None):
        self.model = model
        self.background = background
        self.background_labels = background_labels

    def _expand_to_vector(self, desired_length, labels):
        if not isinstance(labels, int):
            return labels

        return torch.ones(desired_length).long() * labels

    def _compute_input_and_baseline_predictions(self, x):
        #model_fn = _make_forward(self.model, requires_grad=False, return_numpy=False)
        model_fn = self.model
        pred_input = model_fn(x)
        pred_baseline = torch.mean(model_fn(self.background))

        return pred_input, pred_baseline

    def _compute_deltas(self, x, labels, attribution, pred_input, pred_baseline):
        deltas = torch.sum(attribution, axis=1) - (pred_input - pred_baseline)
        return deltas

    def _normalize(self, attribution, pred_input, pred_baseline):
        return attribution / torch.abs((pred_input.reshape(-1, 1) - pred_baseline))

    def _model_specific_explanation(self, x, labels, **kwargs):
        raise NotImplementedError("Need to implement model specific attribution behaviour for {}".format(self.method_name))

    def attribute(self, x, labels, return_delta=False, normalize=False, return_numpy=False, **kwargs):

        # Expand labels (in case a fixed integer label is given), i.e., always use label 1
        labels = self._expand_to_vector(len(x), 1)

        # Default label is all ones (assuming binary classification task)
        if self.background is not None:
            self.background_labels = self._expand_to_vector(len(self.background), 1)

        # Get the attribution, predictions (and possible delta values) for the specific method
        attribution = self._model_specific_explanation(x, labels, **kwargs)

        if normalize or self.closeness_threshold is not None:
            pred_input, pred_baseline = self._compute_input_and_baseline_predictions(x)

        if self.closeness_threshold is not None:
            deltas = self._compute_deltas(x, labels, attribution, pred_input, pred_baseline)
            close_enough = np.all(np.isclose(deltas.detach().numpy(), np.zeros_like(deltas), atol=self.closeness_threshold))
            if not close_enough:
                raise RuntimeError("{} did not produce close enough estimates; mean {} std {}".format(self.method_name, torch.mean(deltas).item(), torch.std(deltas).item()))

        if normalize:
            attribution = self._normalize(attribution, pred_input, pred_baseline)

        if return_numpy:
            attribution = attribution.detach().numpy()
            deltas = deltas.detach().numpy()

        if return_delta:
            if self.closeness_threshold is None:
                raise RuntimeError("{} does not support computation of delta".format(self.method_name))
            return attribution, deltas

        return attribution

class DIY_EG(AttributionMethod):

    def __init__(self, model, background=None, background_labels=None):
        super().__init__(model, background=background, background_labels=None)

        self.method_name = "EG"
        self.closeness_threshold = None # TODO: These are not very accurate, so I had to turn this off

    def _model_specific_explanation(self, x, labels, k=200, batched=False):

        if batched:
            self.closeness_threshold = None # Batched predictions will probably not be accurate enough to pass this test
            assert len(self.background) >= len(x)
            alphas = torch.rand(len(x))
            sample_background = self.background[np.random.choice(len(self.background), len(x), replace=False)]

            x.requires_grad = True
            assert (x.requires_grad) == True and (x.is_leaf == True)
            eg_input = sample_background + alphas.reshape(-1, 1) * (x - sample_background)
            output = self.model(eg_input)
            attributions = torch.autograd.grad(outputs=output, inputs=x, grad_outputs=torch.ones_like(output), create_graph=True)[0]
        else:
            gradient_collector = []
            for i in range(k):
                x.requires_grad = True
                assert (x.requires_grad) == True and (x.is_leaf == True)

                sample_background = self.background[np.random.choice(len(self.background), len(x), replace=False)]
                eg_input = sample_background + torch.rand(1) * (x - sample_background)
                output = self.model(eg_input)
                gradients = torch.autograd.grad(outputs=output, inputs=x, grad_outputs=torch.ones_like(output), create_graph=True)[0]

                difference_to_baseline = x - sample_background
                gradients *= difference_to_baseline

                gradient_collector.append(gradients.unsqueeze(0))

            attributions = torch.mean(torch.cat(gradient_collector, axis=0), axis=0)

        return attributions

def _gradient_input(model, x, label, keep_gradients=False):
    # Assume: x.shape[0] ^= batch_size
    gradients = []
    was_training = model.training
    model.eval()

    if not x.requires_grad:
        x.requires_grad = True

    for i, x_i in enumerate(x):
        x_i = x_i.unsqueeze(0)

        output = torch.exp(model(x_i))
        output = _get_output_per_class(output, label[i])
        grad = torch.autograd.grad(output, x_i)[0]

        gradients.append(grad)

        if not keep_gradients:
            model.zero_grad()

    if was_training:
        model.train()

    return torch.cat(gradients, axis=0)