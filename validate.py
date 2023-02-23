"""
Utility functions for validating MVCBM and SSMVCBM models
"""
import numpy as np
import torch
from utils.metrics import calc_cMetrics, calc_confusion, calc_tMetrics


def validate_epoch_mvcbm(epoch, config, model, dataloader, loss_fn, fold=None, roc=None, prc=None):
    """
    Run a validation round for an MVCBM
    """
    model.eval()

    t_pred_total = torch.Tensor()
    t_true_total = torch.Tensor()
    t_probs_total = torch.Tensor()
    c_pred_total = torch.Tensor()
    c_true_total = torch.Tensor()
    c_probs_total = torch.Tensor()
    all_img_codes = []

    total = 0
    val_target_loss = 0
    val_concepts_loss = [0] * config["num_concepts"]
    val_summed_concepts_loss = 0
    val_total_loss = 0
    all_cMetrics = []

    for k, batch in enumerate(dataloader):
        batch_images, target_true, batch_names, batch_img_codes = \
            batch["images"], batch["label"], batch["file_names"], batch["img_code"]
        batch_images = batch_images.to(device=config["device"], dtype=torch.float32)
        target_true = target_true.to(device=config["device"], dtype=torch.float32)
        if config['dataset'] == 'app':
            batch_names = np.array(list(map(list, zip(*batch_names))), dtype=object)

        if config['dataset'] == 'app':
            mask = torch.tensor(batch_names != "padding.bmp").to(config["device"])
        elif config['dataset'] == 'mawa':
            mask = torch.ones((batch_images.shape[0], config['num_views'])).to(config['device'])
        elif config['dataset'] == 'synthetic':
            mask = torch.ones((batch_images.shape[0], config['num_views'])).to(config['device'])

        concepts_true = batch["concepts"].to(config["device"])
        clinical_feat = batch["features"].to(config["device"])
        all_img_codes.extend(batch_img_codes)
        total += batch_images.size(0)

        with torch.no_grad():
            if model.name in ["MVCBM", "CBM", "Dummy"]:
                # Forward pass
                concepts_pred, target_pred_probs, target_pred_logits, attn_weights = model(
                    batch_images, mask, clinical_feat)
                target_pred_probs = target_pred_probs.squeeze(1)
                target_pred_logits = target_pred_logits.squeeze(1)
                # Calculate the loss
                loss_fn.target_class_weight = None
                loss_fn.target_sample_weight = None
                loss_fn.c_weights = None
                target_loss, concepts_loss, summed_concepts_loss, total_loss = loss_fn(
                    concepts_pred, concepts_true, target_pred_probs, target_pred_logits, target_true)
                val_target_loss += target_loss.item() * batch_images.size(0)
                for concept_idx in range(len(val_concepts_loss)):
                    val_concepts_loss[concept_idx] += concepts_loss[concept_idx].item() * batch_images.size(0)
                val_summed_concepts_loss += summed_concepts_loss.item() * batch_images.size(0)
                val_total_loss += total_loss.item() * batch_images.size(0)
                # Predict
                if model.num_classes == 2:
                    t_predicted = torch.where(target_pred_probs > 0.5, 1, 0).cpu()
                else:
                    t_predicted = torch.argmax(target_pred_probs, 1).cpu()

                c_predicted = torch.where(concepts_pred > 0.5, 1, 0).cpu()
            else:
                # Forward pass
                target_pred = model(concepts_true, clinical_feat).squeeze(1)
                # Calculate loss
                loss_fn.target_class_weight = None
                loss_fn.target_sample_weight = None
                loss_fn.c_weights = None
                target_loss = loss_fn(target_pred, target_true)
                val_target_loss += target_loss.item() * batch_images.size(0)
                val_total_loss = val_target_loss
                # Predict
                if model.num_classes == 2:
                    t_predicted = torch.where(target_pred > 0.5, 1, 0).cpu()
                else:
                    t_predicted = torch.argmax(target_pred, 1).cpu()

            # If last batch_size == 1
            if not t_predicted.shape:
                t_predicted = t_predicted.unsqueeze(0)

            # Append the results
            t_pred_total = torch.cat((t_pred_total, t_predicted))
            t_true_total = torch.cat((t_true_total, target_true.cpu()))
            t_probs_total = torch.cat((t_probs_total, target_pred_probs.cpu()))
            if model.name in ["MVCBM", "CBM", "Dummy"]:
                c_pred_total = torch.cat((c_pred_total, c_predicted))
                c_true_total = torch.cat((c_true_total, concepts_true.cpu()))
                c_probs_total = torch.cat((c_probs_total, concepts_pred.cpu()))

    if model.name in ["MVCBM", "CBM", "Dummy"]:
        for concept_idx in range(len(val_concepts_loss)):

            if len(np.unique(c_true_total[:, concept_idx].numpy())) != 2:
                print(np.unique(c_true_total[:, concept_idx].numpy()))
                print(f"Concept {concept_idx} has only one unique outcome value in the validation set!")
            all_cMetrics.append(
                calc_cMetrics(c_true_total[:, concept_idx], c_probs_total[:, concept_idx], f"c{concept_idx}"))

    tMetrics = calc_tMetrics(t_true_total, t_probs_total)
    conf_matrix, FP_names, FN_names = calc_confusion(t_true_total, t_probs_total, all_img_codes)
    if fold is not None:
        roc.update(epoch + 1, fold, t_true_total, t_probs_total)
        prc.update(epoch + 1, fold, t_true_total, t_probs_total)

    model.train()

    return val_target_loss / total, [val_concepts_loss[i] / total for i in range(len(val_concepts_loss))], \
           val_summed_concepts_loss / total, val_total_loss / total, tMetrics, all_cMetrics, conf_matrix, FP_names, FN_names


def validate_epoch_ssmvcbm(epoch, mode, config, model, dataloader, loss_fn, beta, gamma, fold=None, roc=None, prc=None):
    """
    Run a validation round for an SSMVCBM
    """
    model.eval()

    pred_total = torch.Tensor()
    true_total = torch.Tensor()
    probs_total = torch.Tensor()

    all_img_codes = []

    total = 0
    val_loss = 0
    val_s_concepts_loss = [0]*config["num_s_concepts"] if mode == "sc" else None
    all_cMetrics = [] if mode == "sc" else None

    us_concepts_sample = torch.Tensor().to(device=config["device"])

    for k, batch in enumerate(dataloader):

        batch_images, target_true, batch_names, batch_img_codes = \
            batch["images"], batch["label"], batch["file_names"], batch["img_code"]
        batch_images = batch_images.to(device=config["device"], dtype=torch.float32)
        target_true = target_true.to(device=config["device"], dtype=torch.float32)
        if config['dataset'] == 'app':
            batch_names = np.array(list(map(list, zip(*batch_names))), dtype=object)

        if config['dataset'] == 'app':
            mask = torch.tensor(batch_names != "padding.bmp").to(config["device"])
        elif config['dataset'] == 'mawa':
            mask = torch.ones((batch_images.shape[0], config['num_views'])).to(config['device'])
        elif config['dataset'] == 'synthetic':
            mask = torch.ones((batch_images.shape[0], config['num_views'])).to(config['device'])

        concepts_true = batch["concepts"].to(config["device"])
        clinical_feat = batch["features"].to(config["device"])
        all_img_codes.extend(batch_img_codes)
        total += batch_images.size(0)

        with torch.no_grad():
            # Forward pass
            s_concepts_pred, us_concepts_pred, s_attn_weights, us_attn_weights, discr_concepts_pred, \
            target_pred_probs, target_pred_logits = model(batch_images, mask, clinical_feat)
            target_pred_probs = target_pred_probs.squeeze(1)
            target_pred_logits = target_pred_logits.squeeze(1)
            us_concepts_sample = torch.cat((us_concepts_sample, us_concepts_pred))

            # Calculate the loss
            loss_fn.target_class_weight = None
            loss_fn.target_sample_weight = None
            loss_fn.c_weights = None
            target_loss, s_concepts_loss, summed_s_concepts_loss, summed_discr_concepts_loss, summed_gen_concepts_loss, us_corr_loss = \
                loss_fn(s_concepts_pred, discr_concepts_pred, concepts_true, target_pred_probs, target_pred_logits,
                        target_true, us_concepts_sample)
            if mode == "t":
                val_loss += target_loss.item()*batch_images.size(0)
                if model.num_classes == 2:
                    predicted = torch.where(target_pred_probs > 0.5, 1, 0).cpu()
                else:
                    predicted = torch.argmax(target_pred_probs, 1).cpu()
                pred_total = torch.cat((pred_total, predicted))
                true_total = torch.cat((true_total, target_true.cpu()))
                probs_total = torch.cat((probs_total, target_pred_probs.cpu()))
            elif mode == "sc":
                for concept_idx in range(len(val_s_concepts_loss)):
                    val_s_concepts_loss[concept_idx] += s_concepts_loss[concept_idx].item()*batch_images.size(0)
                val_loss += summed_s_concepts_loss.item()*batch_images.size(0)
                predicted = torch.where(s_concepts_pred > 0.5, 1, 0).cpu()
                pred_total = torch.cat((pred_total, predicted))
                true_total = torch.cat((true_total, concepts_true.cpu()))
                probs_total = torch.cat((probs_total, s_concepts_pred.cpu()))
            elif mode == "usc":
                val_loss += target_loss.item()*batch_images.size(0) + beta*summed_gen_concepts_loss.item()*batch_images.size(0)
            else:
                val_loss += summed_discr_concepts_loss.item()*batch_images.size(0)

    val_loss = val_loss/total

    if mode == "t":
        val_s_concepts_loss_norm = None
        all_cMetrics = None
        us_cov = None
        tMetrics = calc_tMetrics(true_total, probs_total)
        conf_matrix, FP_names, FN_names = calc_confusion(true_total, probs_total, all_img_codes)
        if fold is not None:
            roc.update(epoch+1, fold, true_total, probs_total)
            prc.update(epoch+1, fold, true_total, probs_total)

    elif mode == "sc":
        tMetrics = None
        conf_matrix, FP_names, FN_names = None, None, None
        us_cov = None
        for concept_idx in range(len(val_s_concepts_loss)):
            if len(np.unique(true_total[:, concept_idx].numpy())) != 2:
                print(np.unique(true_total[:, concept_idx].numpy()))
                print(f"Concept {concept_idx} has only one unique outcome value in the validation set!")
            all_cMetrics.append(calc_cMetrics(true_total[:, concept_idx], probs_total[:, concept_idx], f"sc{concept_idx}"))
        val_s_concepts_loss_norm = [val_s_concepts_loss[i]/total for i in range(len(val_s_concepts_loss))]

    elif mode == "usc":
        val_loss += gamma*us_corr_loss.item()
        us_cov = torch.cov(us_concepts_sample.T).cpu()
        val_s_concepts_loss_norm = None
        tMetrics = None
        all_cMetrics = None
        conf_matrix = None
        FP_names = None
        FN_names = None

    else:
        val_s_concepts_loss_norm = None
        tMetrics = None
        all_cMetrics = None
        conf_matrix = None
        FP_names = None
        FN_names = None
        us_cov = None
    model.train()

    return val_loss, val_s_concepts_loss_norm, tMetrics, all_cMetrics, conf_matrix, FP_names, FN_names, us_cov
