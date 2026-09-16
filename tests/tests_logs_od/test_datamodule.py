# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

# import pytest
# from pathlib import Path
# from torch.utils.data import DataLoader, WeightedRandomSampler
# from adf.logs.autoencoder.datamodule import LogDataModule
#
#
# class DummyWindowDataset:
#     def __init__(self, data_sel=None, win_ind_sel=None):
#         self.data_sel = data_sel
#         self.win_ind_sel = win_ind_sel
#
#     @classmethod
#     def from_filenames(
#         cls, data_filename, ind_filename, data_dim, win_size, return_memmap
#     ):
#         instance = cls()
#         instance.data_filename = data_filename
#         instance.ind_filename = ind_filename
#         instance.data_dim = data_dim
#         instance.win_size = win_size
#         instance.return_memmap = return_memmap
#         return instance
#
#     def __eq__(self, other):
#         return isinstance(other, DummyWindowDataset) and self.__dict__ == other.__dict__
#
#
# class DummyWindowIterableDataset:
#     def __init__(self, iterator, input_dim, window_size):
#         self.iterator = iterator
#         self.input_dim = input_dim
#         self.window_size = window_size
#
#     def __eq__(self, other):
#         return (
#             isinstance(other, DummyWindowIterableDataset)
#             and self.iterator == other.iterator
#             and self.input_dim == other.input_dim
#             and self.window_size == other.window_size
#         )
#
#
# @pytest.fixture(autouse=True)
# def patch_datasets(monkeypatch):
#     import adf.logs.autoencoder.datamodule as dm
#
#     monkeypatch.setattr(dm, "WindowDataset", DummyWindowDataset)
#     monkeypatch.setattr(dm, "WindowIterableDataset", DummyWindowIterableDataset)
#
#
# @pytest.fixture
# def dummy_iterator():
#     return iter(
#         [(Path("dummy_data_1"), Path("dummy_ind_1")), (None, Path("dummy_ind_2"))]
#     )
#
#
# @pytest.fixture
# def dummy_backup_files():
#     return {"data": Path("backup_data"), "ind": Path("backup_ind")}
#
#
# def test_setup_on_first_iteration(dummy_iterator):
#     dm = LogDataModule(
#         iterator=dummy_iterator,
#         input_dim=10,
#         window_size=5,
#         num_workers=0,
#         batch_size=2,
#         iteration_phase="on_first_iteration",
#     )
#     dm.setup("fit")
#     assert isinstance(dm.train_dataset, DummyWindowIterableDataset)
#     assert dm.val_dataset == []
#
#
# def test_setup_on_sync_backup(dummy_backup_files):
#     dummy_iter = iter([])
#     dm = LogDataModule(
#         iterator=dummy_iter,
#         input_dim=20,
#         window_size=10,
#         num_workers=0,
#         batch_size=4,
#         iteration_phase="on_sync_backup",
#         backup_files=dummy_backup_files,
#     )
#     dm.setup("fit")
#     ds_train = dm.train_dataset
#     assert hasattr(ds_train, "data_filename")
#     assert ds_train.data_filename == dummy_backup_files["data"]
#     assert ds_train.ind_filename == dummy_backup_files["ind"]
#     assert ds_train.data_dim == 20
#     assert ds_train.win_size == 10
#
#     dm.scores = [1.0, 1.0]
#     sampler = dm.sampler
#     assert isinstance(sampler, WeightedRandomSampler)
#     assert sampler.num_samples == 2
#
#
# def test_on_train_dataloader_fetch_with_data():
#     it = iter([(Path("dummy_data_valid"), Path("dummy_ind_valid"))])
#     dm = LogDataModule(
#         iterator=it,
#         input_dim=15,
#         window_size=3,
#         num_workers=0,
#         batch_size=2,
#         iteration_phase="on_save_samples",
#     )
#     dm.data_sel = "dummy_data_sel"
#     dm.win_ind_sel = "dummy_win_ind_sel"
#     dm.on_train_dataloader_fetch()
#     assert isinstance(dm.train_dataset, DummyWindowDataset)
#     assert dm.train_dataset.data_sel == "dummy_data_sel"
#     assert dm.train_dataset.win_ind_sel == "dummy_win_ind_sel"
#     ds_val = dm.val_dataset
#     assert hasattr(ds_val, "data_filename")
#     assert ds_val.data_filename == Path("dummy_data_valid")
#     assert ds_val.ind_filename == Path("dummy_ind_valid")
#     assert ds_val.return_memmap is True
#
#
# def test_on_train_dataloader_fetch_with_none():
#     it = iter([(None, Path("dummy_ind_none"))])
#     dm = LogDataModule(
#         iterator=it,
#         input_dim=15,
#         window_size=3,
#         num_workers=0,
#         batch_size=2,
#         iteration_phase="on_save_samples",
#     )
#     dm.data_sel = "dummy_data_sel"
#     dm.win_ind_sel = "dummy_win_ind_sel"
#     dm.val_dataset = "dummy_val_dataset"
#     dm.on_train_dataloader_fetch()
#     assert dm.test_dataset == "dummy_val_dataset"
#     assert dm.val_dataset == []
#
#
# def test_on_val_dataloader_fetch_on_save_samples():
#     it = iter([(Path("dummy_data_val"), Path("dummy_ind_val"))])
#     dm = LogDataModule(
#         iterator=it,
#         input_dim=12,
#         window_size=4,
#         num_workers=0,
#         batch_size=2,
#         iteration_phase="on_save_samples",
#     )
#     dm.on_val_dataloader_fetch()
#     ds_val = dm.val_dataset
#     assert hasattr(ds_val, "data_filename")
#     assert ds_val.data_filename == Path("dummy_data_val")
#     assert ds_val.ind_filename == Path("dummy_ind_val")
#     assert ds_val.return_memmap is False
#
#
# def test_on_val_dataloader_fetch_on_sync_backup(dummy_backup_files):
#     dummy_iter = iter([])
#     dm = LogDataModule(
#         iterator=dummy_iter,
#         input_dim=12,
#         window_size=4,
#         num_workers=0,
#         batch_size=2,
#         iteration_phase="on_sync_backup",
#         backup_files=dummy_backup_files,
#     )
#     dm.on_val_dataloader_fetch()
#     ds_val = dm.val_dataset
#     assert hasattr(ds_val, "data_filename")
#     assert ds_val.data_filename == dummy_backup_files["data"]
#     assert ds_val.ind_filename == dummy_backup_files["ind"]
#     assert ds_val.return_memmap is False
#
#
# def test_train_dataloader(dummy_iterator):
#     it = iter([(Path("dummy_data"), Path("dummy_ind"))])
#     dm = LogDataModule(
#         iterator=it,
#         input_dim=8,
#         window_size=2,
#         num_workers=0,
#         batch_size=2,
#         iteration_phase="on_first_iteration",
#     )
#     dm.setup("fit")
#     dl = dm.train_dataloader()
#     assert isinstance(dl, DataLoader)
#     assert dl.dataset == dm.train_dataset
#
#
# def test_val_dataloader(dummy_backup_files):
#     dummy_iter = iter([])
#     dm = LogDataModule(
#         iterator=dummy_iter,
#         input_dim=8,
#         window_size=2,
#         num_workers=0,
#         batch_size=2,
#         iteration_phase="on_sync_backup",
#         backup_files=dummy_backup_files,
#     )
#     dm.setup("fit")
#     dm.on_val_dataloader_fetch()
#     dm.scores = [1.0, 1.0]
#     dl = dm.val_dataloader()
#     assert isinstance(dl, DataLoader)
#     assert dl.dataset == dm.val_dataset
#
#
# def test_test_dataloader(dummy_backup_files):
#     dummy_iter = iter([])
#     dm = LogDataModule(
#         iterator=dummy_iter,
#         input_dim=8,
#         window_size=2,
#         num_workers=0,
#         batch_size=2,
#         iteration_phase="on_save_samples",
#         backup_files=dummy_backup_files,
#     )
#     dm.setup("test")
#     dl = dm.test_dataloader()
#     assert isinstance(dl, DataLoader)
#     assert dl.dataset == dm.test_dataset
