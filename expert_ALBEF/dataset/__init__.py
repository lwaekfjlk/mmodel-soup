import torch
from dataset.bi_cls_dataset import bi_cls_dataset
from dataset.caption_dataset import (pretrain_dataset, re_eval_dataset,
                                     re_train_dataset)
from dataset.grounding_dataset import grounding_dataset
from dataset.irfl_dataset import irfl_test_dataset, irfl_train_dataset
from dataset.mustard_dataset import mustard_test_dataset, mustard_train_dataset
from dataset.nlvr_dataset import nlvr_dataset
from dataset.nycartoon_dataset import (nycartoon_test_dataset,
                                       nycartoon_train_dataset)
from dataset.randaugment import RandomAugment
from dataset.sarc_detect_dataset import (sarc_detect_test_dataset,
                                         sarc_detect_train_dataset)
from dataset.urfunny_dataset import urfunny_test_dataset, urfunny_train_dataset
from dataset.ve_dataset import ve_dataset
from dataset.vqa_dataset import vqa_dataset
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


def create_dataset(dataset, config):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    pretrain_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config["image_res"], scale=(0.2, 1.0), interpolation=Image.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            RandomAugment(
                2,
                7,
                isPIL=True,
                augs=[
                    "Identity",
                    "AutoContrast",
                    "Equalize",
                    "Brightness",
                    "Sharpness",
                    "ShearX",
                    "ShearY",
                    "TranslateX",
                    "TranslateY",
                    "Rotate",
                ],
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config["image_res"], scale=(0.5, 1.0), interpolation=Image.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            RandomAugment(
                2,
                7,
                isPIL=True,
                augs=[
                    "Identity",
                    "AutoContrast",
                    "Equalize",
                    "Brightness",
                    "Sharpness",
                    "ShearX",
                    "ShearY",
                    "TranslateX",
                    "TranslateY",
                    "Rotate",
                ],
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_res"], config["image_res"]), interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if dataset == "pretrain":
        dataset = pretrain_dataset(config["train_file"], pretrain_transform)
        return dataset

    elif dataset == "re":
        train_dataset = re_train_dataset(
            config["train_file"], train_transform, config["image_root"]
        )
        val_dataset = re_eval_dataset(
            config["val_file"], test_transform, config["image_root"]
        )
        test_dataset = re_eval_dataset(
            config["test_file"], test_transform, config["image_root"]
        )
        return train_dataset, val_dataset, test_dataset

    elif dataset == "vqa":
        train_dataset = vqa_dataset(
            config["train_file"],
            train_transform,
            config["vqa_root"],
            config["vg_root"],
            split="train",
        )
        vqa_test_dataset = vqa_dataset(
            config["test_file"],
            test_transform,
            config["vqa_root"],
            config["vg_root"],
            split="test",
            answer_list=config["answer_list"],
        )
        return train_dataset, vqa_test_dataset

    elif dataset == "nlvr":
        train_dataset = nlvr_dataset(
            config["train_file"], train_transform, config["image_root"]
        )
        val_dataset = nlvr_dataset(
            config["val_file"], test_transform, config["image_root"]
        )
        test_dataset = nlvr_dataset(
            config["test_file"], test_transform, config["image_root"]
        )
        return train_dataset, val_dataset, test_dataset

    elif dataset == "ve":
        train_dataset = ve_dataset(
            config["train_file"], train_transform, config["image_root"]
        )
        val_dataset = ve_dataset(
            config["val_file"], test_transform, config["image_root"]
        )
        test_dataset = ve_dataset(
            config["test_file"], test_transform, config["image_root"]
        )
        return train_dataset, val_dataset, test_dataset

    elif dataset == "grounding":
        train_transform = transforms.Compose(
            [
                transforms.Resize(
                    (config["image_res"], config["image_res"]),
                    interpolation=Image.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2,
                    7,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Equalize",
                        "Brightness",
                        "Sharpness",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_dataset = grounding_dataset(
            config["train_file"], train_transform, config["image_root"], mode="train"
        )
        test_dataset = grounding_dataset(
            config["test_file"], test_transform, config["image_root"], mode="test"
        )
        return train_dataset, test_dataset

    elif dataset == "bi-cls":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    config["image_res"], scale=(0.5, 1.0), interpolation=Image.BICUBIC
                ),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2,
                    7,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Equalize",
                        "Brightness",
                        "Sharpness",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(
                    (config["image_res"], config["image_res"]),
                    interpolation=Image.BICUBIC,
                ),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                normalize,
            ]
        )
        dataset = load_dataset(*config["dataset_args"])
        train_dataset = bi_cls_dataset(dataset["train"], train_transform)
        val_dataset = bi_cls_dataset(dataset["validation"], test_transform)
        test_dataset = bi_cls_dataset(dataset["test"], test_transform)
        return train_dataset, val_dataset, test_dataset

    elif dataset == "sarc-detect":
        train_dataset = sarc_detect_train_dataset(
            config["train_file"], train_transform, config["image_root"]
        )
        val_dataset = sarc_detect_test_dataset(
            config["val_file"], test_transform, config["image_root"]
        )
        test_dataset = sarc_detect_test_dataset(
            config["test_file"], test_transform, config["image_root"]
        )
        return train_dataset, val_dataset, test_dataset

    elif dataset == "irfl":
        train_dataset = irfl_train_dataset(
            config["train_file"], train_transform, config["image_root"]
        )
        val_dataset = irfl_test_dataset(
            config["val_file"], test_transform, config["image_root"]
        )
        test_dataset = irfl_test_dataset(
            config["test_file"], test_transform, config["image_root"]
        )
        return train_dataset, val_dataset, test_dataset

    elif dataset == "nycartoon":
        train_dataset = nycartoon_train_dataset(
            config["train_file"], train_transform, config["image_root"]
        )
        val_dataset = nycartoon_test_dataset(
            config["val_file"], test_transform, config["image_root"]
        )
        test_dataset = nycartoon_test_dataset(
            config["test_file"], test_transform, config["image_root"]
        )
        return train_dataset, val_dataset, test_dataset

    elif dataset == "mustard":
        train_dataset = mustard_train_dataset(
            config["train_file"], train_transform, config["image_root"]
        )
        val_dataset = mustard_test_dataset(
            config["val_file"], test_transform, config["image_root"]
        )
        test_dataset = mustard_test_dataset(
            config["test_file"], test_transform, config["image_root"]
        )
        return train_dataset, val_dataset, test_dataset

    elif dataset == "urfunny":
        train_dataset = urfunny_train_dataset(
            config["train_file"], train_transform, config["image_root"]
        )
        val_dataset = urfunny_test_dataset(
            config["val_file"], test_transform, config["image_root"]
        )
        test_dataset = urfunny_test_dataset(
            config["test_file"], test_transform, config["image_root"]
        )
        return train_dataset, val_dataset, test_dataset


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return (
        torch.stack(image_list, dim=0),
        question_list,
        answer_list,
        torch.Tensor(weight_list),
        n,
    )


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
