import torch
from tqdm import tqdm

def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc="   Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs.squeeze(1), labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        loop.set_postfix(loss=loss.item())
    return train_loss / len(train_loader)


def train_segmentation(model, train_loader, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc="   Training", leave=False)
    for data in loop:
        if len(data)==3:
            images, labels, _ = data
        else:
            images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        if outputs.shape[1]>1:
            loss = loss_fn(outputs, labels)
        else:
            loss = loss_fn(outputs, labels.float())

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        loop.set_postfix(loss=loss.item())
    return train_loss / len(train_loader)


def train_two_step(model, train_loader, optimizer, loss_seg, loss_cls, device):
    model.train()
    train_loss = 0
    train_loss_segmentation = 0
    train_loss_classification = 0
    loop = tqdm(train_loader, desc="   Training", leave=False)

    for data in loop:

        images, mask, labels = data

        images, mask,  labels = images.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs_cls, outputs_seg = model(images)

        loss_segmentation = loss_seg(outputs_seg, mask)


        if outputs_cls.shape[1]>1:
            loss_classification = loss_cls(outputs_cls, labels)
        else:
            loss_classification = loss_cls(outputs_cls.squeeze(1), labels.float())

        loss = loss_segmentation * 2 + loss_classification

        loss.backward()
        optimizer.step()


        train_loss_classification += loss_classification.item()
        train_loss_segmentation += loss_segmentation.item()
        train_loss += loss.item()

        loop.set_postfix(loss=loss.item())
    return train_loss / len(train_loader), train_loss_segmentation / len(train_loader), train_loss_classification / len(train_loader)
