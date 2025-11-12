from nowcastpnn.utils.metrics import pnn_PIs, evaluate_PIs
from nowcastpnn.utils.train_utils import set_seeds


def calculate_metrics(model, test_loader, random_split, dow) -> dict:
    intervals = pnn_PIs(
        model=model,
        test_loader=test_loader,
        random_split=random_split,
        dow=dow
    )
    return evaluate_PIs(intervals_dict=intervals, test_loader=test_loader)
