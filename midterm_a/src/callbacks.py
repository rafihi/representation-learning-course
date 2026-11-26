"""Lightning callbacks."""
import lightning as L

from .evaluation import extract_features, knn_scores, topk_acc


class KNNMonitor(L.Callback):
    """Every `every` epochs (and at the last epoch) compute k-NN top-1 on the held-out VALIDATION split and log it.
    Never touches the test set."""

    def __init__(self, X_train, y_train, X_val, y_val, num_classes, every=5, k=20, T=0.1):
        self.d = (X_train, y_train, X_val, y_val)
        self.num_classes, self.every, self.k, self.T = num_classes, every, k, T

    def on_train_epoch_end(self, trainer, pl_module):
        e = trainer.current_epoch + 1
        if self.every <= 0 or (e % self.every != 0 and e != trainer.max_epochs):
            return
        Xtr, ytr, Xva, yva = self.d
        f_tr, f_va = extract_features(pl_module.encoder, Xtr), extract_features(pl_module.encoder, Xva)
        acc = topk_acc(knn_scores(f_tr, ytr, f_va, self.num_classes, self.k, self.T), yva)["top1"]
        pl_module.log("val_knn_top1", acc, prog_bar=True)
