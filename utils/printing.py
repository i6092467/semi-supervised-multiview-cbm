"""
Utility functions for printing the output
"""


def print_epoch_val_scores(config, mode, target_loss, concepts_loss, summed_concepts_loss, total_loss, tMetrics,
						   all_cMetrics, print_all_c=True):
	print(f"(Mode = {mode}) Epoch validation scores:")

	print(f"        Concepts summed loss: {summed_concepts_loss}")
	if print_all_c:
		for concept_idx in range(len(concepts_loss)):
			print(f"            Concept {concept_idx} loss: {concepts_loss[concept_idx]}")

	print(f"        Concepts averaged metrics")
	print(
		f"            Accuracy: {sum([all_cMetrics[i].accuracy for i in range(len(all_cMetrics))]) / config['num_concepts']} "
		f"F1-macro: {sum([all_cMetrics[i].f1_macro for i in range(len(all_cMetrics))]) / config['num_concepts']}")
	print(
		f"            AUROC: {sum([all_cMetrics[i].auroc for i in range(len(all_cMetrics))]) / config['num_concepts']} "
		f"AUPR: {sum([all_cMetrics[i].aupr for i in range(len(all_cMetrics))]) / config['num_concepts']}")

	print("        Target")
	print(f"            Target loss: {target_loss}")
	print(f"            PPV: {tMetrics.ppv} NPV: {tMetrics.npv} ")
	print(f"            Sensitivity: {tMetrics.sensitivity} Specificity: {tMetrics.specificity}")
	print(f"            Accuracy: {tMetrics.accuracy} Balanced accuracy: {tMetrics.balanced_accuracy}")
	print(f"            F1_1: {tMetrics.f1_1} F1_0: {tMetrics.f1_0} F1_macro: {tMetrics.f1_macro}")
	print(f"            AUROC: {tMetrics.auroc} AUPR: {tMetrics.aupr}")

	print(f"        Total loss: {total_loss}")


def print_epoch_val_scores_(config, mode, val_loss, val_s_concepts_loss, tMetrics, all_cMetrics, print_all_c=True):
	print(f"(Mode = {mode}) Epoch validation scores:")

	if mode == "sc":
		print(f"        Concepts summed loss: {val_loss}")
		if print_all_c:
			for concept_idx in range(len(val_s_concepts_loss)):
				print(f"            Concept {concept_idx} loss: {val_s_concepts_loss[concept_idx]}")

		print(f"        Concepts averaged metrics")
		print(f"            Accuracy: {sum([all_cMetrics[i].accuracy for i in range(len(all_cMetrics))]) / config['num_s_concepts']} "
			  f"F1-macro: {sum([all_cMetrics[i].f1_macro for i in range(len(all_cMetrics))]) / config['num_s_concepts']}")
		print(f"            AUROC: {sum([all_cMetrics[i].auroc for i in range(len(all_cMetrics))]) / config['num_s_concepts']} "
			  f"AUPR: {sum([all_cMetrics[i].aupr for i in range(len(all_cMetrics))]) / config['num_s_concepts']}")

		if print_all_c:
			for concept_idx in range(len(all_cMetrics)):
				print(f"        Concept {concept_idx}")
				print(
					f"            Accuracy: {all_cMetrics[concept_idx].accuracy} F1-macro: {all_cMetrics[concept_idx].f1_macro}")
				print(f"            AUROC: {all_cMetrics[concept_idx].auroc} AUPR: {all_cMetrics[concept_idx].aupr}")

	elif mode == "t":
		print("        Target")
		print(f"            Target loss: {val_loss}")
		print(f"            PPV: {tMetrics.ppv} NPV: {tMetrics.npv} ")
		print(f"            Sensitivity: {tMetrics.sensitivity} Specificity: {tMetrics.specificity}")
		print(f"            Accuracy: {tMetrics.accuracy} Balanced accuracy: {tMetrics.balanced_accuracy}")
		print(f"            F1_1: {tMetrics.f1_1} F1_0: {tMetrics.f1_0} F1_macro: {tMetrics.f1_macro}")
		print(f"            AUROC: {tMetrics.auroc} AUPR: {tMetrics.aupr}")
