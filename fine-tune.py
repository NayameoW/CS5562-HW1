import os
import argparse
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np

from resnet_attack_todo import ResnetPGDAttacker

parser = argparse.ArgumentParser(description="Adversarially train a Resnet50 model")
parser.add_argument('--eps', type=float, help='maximum perturbation for PGD attack', default=8 / 255)
parser.add_argument('--alpha', type=float, help='step size for PGD attack', default=2 / 255)
parser.add_argument('--steps', type=int, help='number of steps for PGD attack', default=20)
parser.add_argument('--batch_size', type=int, help='batch size for PGD attack', default=100)
parser.add_argument('--batch_num', type=int, help='number of batches on which to run PGD attack', default=None)
parser.add_argument('--seed', type=int, help='set manual seed value for reproducibility, default 1234',
                    default=1234)
parser.add_argument('--epoch', type=int, help='number of epochs for training', default=10)
parser.add_argument('--resultsdir', type=str, help='name of the folder to save the results to', default='results')
parser.add_argument('--test', type=bool, help='set to True to run test mode', default=False)

args = parser.parse_args()

# Initialize hyperparameters
EPS = args.eps
ALPHA = args.alpha
STEPS = args.steps

BATCH_SIZE = args.batch_size
BATCH_NUM = args.batch_num
if BATCH_NUM is None:
    BATCH_NUM = 1281167 // BATCH_SIZE + 1
assert BATCH_NUM > 0

EPOCH = args.epoch

if args.seed:
    SEED = args.seed
    torch.manual_seed(SEED)
else:
    SEED = torch.seed()

RESULTS_DIR = args.resultsdir

test_mode = args.test

# Load the model
print("Loading model...")
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
preprocess = weights.transforms()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load the dataset
print("Loading dataset...")
ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True)

def preprocess_img(example):
    example['image'] = preprocess(example['image'])
    return example

# Filter out grayscale images
ds = ds.filter(lambda example: example['image'].mode == 'RGB')
# Preprocess function will be applied to images on-the-fly whenever they are being accessed in the loop
ds = ds.map(preprocess_img)
ds = ds.shuffle(seed=SEED)
# Only take desired portion of dataset
ds = ds.take(BATCH_NUM * BATCH_SIZE)


# PGD attack
dset_loader = DataLoader(ds, batch_size=BATCH_SIZE)
dset_classes = weights.meta["categories"]
attacker = ResnetPGDAttacker(model, dset_loader)

def adversarial_train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the layers to be trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    numbers = [2, 4, 6, 8, 10]
    for i in numbers:
        adv_images = torch.load('results/adv_images_eps_{}'.format(i))['adv_images']
        for epoch in range(epoch):
            print('Training on adversarial examples with epsilon = {}/255'.format(i))
            print('Epoch: {}'.format(epoch))
            for batch_idx, inputs in enumerate(train_loader):
                # print('Batch: %d Generating adversarial examples...' % batch_idx)
                # adv_images = pgd_attacker.pgd_attack(**inputs)
                
                start_id = batch_idx * BATCH_SIZE
                end_id = (batch_idx + 1) * BATCH_SIZE
                batch_adv_images = adv_images[start_id:end_id].clone().detach().to(device)

                optimizer.zero_grad()
                images = inputs['image'].clone().detach().to(device)
                labels = inputs['label'].clone().detach().to(device)
                output = model(images)
                adv_output = model(batch_adv_images)

                loss = criterion(adv_output, labels) + criterion(output, labels)

                loss.backward()
                optimizer.step()
            print('Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

def compute_accuracy(model, adv_images, labels):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adv_images = adv_images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = model(adv_images).softmax(1)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += predictions.size(0) 
  
    return correct / total


if test_mode:
    print("Running test mode...")
    loaded_data = torch.load('results/adv_images')
    adv_images = loaded_data['adv_images']
    labels = loaded_data['labels']
    model.load_state_dict(torch.load('fine-tuned-ResNet50.pth'))
    accuracy = compute_accuracy(model, adv_images, labels)
    print("Test accuracy: {:.2f}%".format(accuracy * 100))
else:
    adversarial_train(model, dset_loader, optimizer, criterion, EPOCH)
    torch.save(model.state_dict(), 'fine-tuned-ResNet50.pth')