"""
Utility functions for intervening on the concept bottleneck methods
"""
import numpy as np
import torch


def intervene_CBM(model, config, batch, intervention_concept_ids):
    """
    Makes predictions for the given batch for the target variable under the specified intervention.
    Specified concepts are replaced with the ground-truth values.

    @param model: concept bottleneck model
    @param config: configuration parameters
    @param batch: a batch of data
    @param intervention_concept_ids: indices of the concepts to be intervened on
    @return: model's predictions for the target variable before and after the intervention, respectively
    """

    batch_images, target_true, batch_names = batch["images"], batch["label"], batch["file_names"]
    batch_images = batch_images.to(device=config["device"], dtype=torch.float32)
    target_true = target_true.to(device=config["device"], dtype=torch.float32)
    concepts_true = batch["concepts"].to(config["device"])
    ex_feat = batch["features"].to(config["device"])
    if config['dataset'] == 'app':
        batch_names = np.array(list(map(list, zip(*batch_names))), dtype=object)

    if config['dataset'] == 'app':
        mask = torch.tensor(batch_names != "padding.bmp").to(config["device"])
    elif config['dataset'] == 'mawa':
        mask = torch.ones((batch_images.shape[0], config['num_views'])).to(config['device'])
    elif config['dataset'] == 'synthetic':
        mask = torch.ones((batch_images.shape[0], config['num_views'])).to(config['device'])

    with torch.no_grad():
        # Forward pass with the ground-truth concept values
        if config['model'] == 'SSMVCBM':
            concepts_pred, _, attn_weights, _, _, target_pred, _ = model(batch_images, mask, ex_feat)
            concepts_pred_int, _, attn_weights_int, _, _, target_pred_int, _ = model(
                batch_images, mask, ex_feat, intervention_concept_ids, concepts_true)
        elif config['model'] == 'MVCBM':
            concepts_pred, target_pred, _, attn_weights = model(batch_images, mask, ex_feat)
            concepts_pred_int, target_pred_int, _, attn_weights_int = model(
                batch_images, mask, ex_feat, intervention_concept_ids, concepts_true)
        elif config['model'] == 'CBM':
            concepts_pred, target_pred, _, attn_weights = model(batch_images, mask, ex_feat)
            concepts_pred_int, target_pred_int, _, attn_weights_int = model(
                batch_images, mask, ex_feat, intervention_concept_ids, concepts_true)
        target_pred = target_pred.squeeze(1)
        target_pred_int = target_pred_int.squeeze(1)

    return target_pred, target_pred_int
