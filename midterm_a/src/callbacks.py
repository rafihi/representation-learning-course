"""Lightning callbacks."""
import lightning as L

from .evaluation import extract_features, knn_scores, topk_acc


class KNNMonitor(L.Callback):
    """Every `every` epochs (and at the last epoch) compute k-NN top-1 on the held-out VALIDATION split and log it.
    Never touches the test set."""

    def __init__(self, X_train, y_train, X_val, y_val, num_classes, every=5, k=20, T=0.1, weight="exp"):
        self.d = (X_train, y_train, X_val, y_val)
        self.num_classes, self.every, self.k, self.T, self.weight = num_classes, every, k, T, weight

    def on_train_epoch_end(self, trainer, pl_module):
        e = trainer.current_epoch + 1
        if self.every <= 0 or (e % self.every != 0 and e != trainer.max_epochs):
            return
        Xtr, ytr, Xva, yva = self.d
        f_tr, f_va = extract_features(pl_module.encoder, Xtr), extract_features(pl_module.encoder, Xva)
        scores = knn_scores(f_tr, ytr, f_va, self.num_classes, self.k, self.T, weight=self.weight)
        pl_module.log("val_knn_top1", topk_acc(scores, yva)["top1"], prog_bar=True)


class KeepBestState(L.Callback):
    """Keeps (in CPU memory) the weights of the step with the best validation metric.
    Used by fine-tuning so that the reported model is chosen on VALIDATION, never on test."""

    def __init__(self, monitor="val_top1"):
        self.monitor = monitor
        self.best, self.best_step, self.state = float("-inf"), None, None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        v = trainer.callback_metrics.get(self.monitor)
        if v is None:
            return
        v = float(v)
        if v > self.best:
            self.best, self.best_step = v, int(trainer.global_step)
            self.state = {k: t.detach().cpu().clone() for k, t in pl_module.state_dict().items()}
