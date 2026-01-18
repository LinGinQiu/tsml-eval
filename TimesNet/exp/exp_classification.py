import os
import pdb
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from docutils.nodes import classifier
from torch import optim
from torch.fx.experimental.unification.unification_tools import first
from torchaudio.functional import resample

from TimesNet.data_provider.data_factory import data_provider
from TimesNet.exp.exp_basic import Exp_Basic
from TimesNet.utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy

warnings.filterwarnings("ignore")


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        train_data, train_loader = self._get_data(flag="train")
        test_data, test_loader = self._get_data(flag="test")
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.data.shape[-1]

        model = super()._build_model()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, self.data_zip, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.view(-1).long().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="test")

        path = self.args.results_path + "/checkpoints/" + setting
        if not os.path.exists(path):
            os.makedirs(path)
        time_start = time.time()
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.view(-1).long())
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}"
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)

            print(
                "Epoch: {}, Steps: {} | Train Loss: {:.3f} Vali Loss: {:.3f} Vali Acc: {:.3f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, val_accuracy
                )
            )
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            self.log_vram_usage(epoch + 1)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        time_fit = time.time() - time_start
        self.time_fit = time_fit
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.args.results_path + "/checkpoints/" + setting,
                        "checkpoint.pth",
                    )
                )
            )

        preds = []
        trues = []
        test_start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print("test shape:", preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(
            preds - preds.max(dim=1, keepdim=True).values, dim=1
        )  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        from TimesNet.utils.results_writing import write_classification_results

        test_probs = probs.cpu().numpy()
        test_preds = self.classes_[np.argmax(test_probs, axis=1)]
        y_test = trues
        classifier_name = self.args.classifier_name
        dataset_name = self.args.dataset
        results_path = self.args.results_path
        resample_id = self.args.resample_id
        first_comment = self.first_comment
        second = str(setting)
        test_acc = accuracy
        fit_time = self.time_fit
        test_time = time.time() - test_start_time
        n_classes = self.args.num_class
        write_classification_results(
            test_preds,
            test_probs,
            y_test,
            classifier_name,
            dataset_name,
            results_path,
            full_path=False,
            first_line_classifier_name=(
                f"{classifier_name} ({type(classifier).__name__})"
            ),
            split="TEST",
            resample_id=resample_id,
            time_unit="MILLISECONDS",
            first_line_comment=first_comment,
            parameter_info=second,
            accuracy=test_acc,
            fit_time=fit_time,
            predict_time=test_time,
            benchmark_time=None,
            memory_usage=None,
            n_classes=n_classes,
            train_estimate_method="N/A",
            train_estimate_time=-1,
            fit_and_estimate_time=None,
        )

        # result save
        folder_path = self.args.results_path + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(f"accuracy:{accuracy}")
        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(setting + "  \n")
        f.write(f"accuracy:{accuracy}")
        f.write("\n")
        f.write("\n")
        f.close()
        return
